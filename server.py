from dotenv import load_dotenv
import os
import sys
import asyncio
import gc
import json
import faiss
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header, Form
import uvicorn
from insightface.utils import storage
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()
app = FastAPI()

# Configuration
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "17866"))
detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
model_load_delay = int(os.getenv("MODEL_LOAD_DELAY", "60"))
idle_timeout = int(os.getenv("IDLE_TIMEOUT", "1200"))
match_threshold = float(os.getenv("MATCH_THRESHOLD", "0.65"))

# Model storage
storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'

# Globals for model and indexing
tface = None
unload_task = None
index = None
id_map = []  # maps vector index -> username

# Persistent storage file
db_file = os.getenv("FACE_DB_FILE", "face_db.json")

async def init_face_analysis():
    global tface
    if tface is None:
        providers = ['CPUExecutionProvider'] if detector_backend in ("onnx", "cpu") else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logging.info(f"Loading model {recognition_model} using {providers}")
        fa = FaceAnalysis(providers=providers, allowed_modules=['detection','recognition'], name=recognition_model)
        fa.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640,640))
        tface = fa

async def delayed_load():
    await asyncio.sleep(model_load_delay)
    await init_face_analysis()

@app.on_event("startup")
async def startup():
    asyncio.create_task(delayed_load())
    load_face_db()
    build_index()

def unload_model():
    global tface
    if tface:
        logging.info("Unloading model to free memory")
        del tface; tface = None
        gc.collect()

async def schedule_unload():
    await asyncio.sleep(idle_timeout)
    unload_model()

@app.middleware("http")
async def activity_check(req, call_next):
    global unload_task
    if unload_task:
        unload_task.cancel()
    unload_task = asyncio.create_task(schedule_unload())
    return await call_next(req)

async def verify_header(key: str = Header(...)):
    if key != api_auth_key:
        raise HTTPException(401, "Invalid API key")
    return key

@app.get("/")
async def root():
    return {"title":"unofficial face recognition api","link":"https://mtmt.tech/docs/advanced/facial_api"}

# Load and persist face DB
face_db = {}

def load_face_db():
    global face_db
    if os.path.exists(db_file):
        with open(db_file,'r') as f:
            face_db = json.load(f)
    else:
        face_db = {}

def save_face_db():
    with open(db_file,'w') as f:
        json.dump(face_db, f)

# Build FAISS index for fast similarity search
def build_index():
    global index, id_map
    all_vecs = []
    id_map = []
    for user, vecs in face_db.items():
        for v in vecs:
            all_vecs.append(v)
            id_map.append(user)
    if all_vecs:
        dim = len(all_vecs[0])
        index = faiss.IndexFlatIP(dim)
        mat = np.array(all_vecs).astype('float32')
        faiss.normalize_L2(mat)
        index.add(mat)
    else:
        index = None

# Registration
@app.post("/v1/vision/face/register")
async def register_face(key: str = Depends(verify_header), image: UploadFile = File(...), name: str = Form(...)):
    await init_face_analysis()
    img = load_image_bytes(await image.read(), image.content_type)
    faces = tface.get(img)
    if not faces:
        return {"success":False, "error":"No face detected"}
    vecs = [face.normed_embedding.tolist() for face in faces]
    face_db.setdefault(name, []).extend(vecs)
    save_face_db()
    build_index()
    return {"success":True}

# List
@app.post("/v1/vision/face/list")
async def list_faces(key: str = Depends(verify_header)):
    return {"success":True, "faces":list(face_db.keys())}

# Delete
@app.post("/v1/vision/face/delete")
async def delete_face(key: str = Depends(verify_header), name: str = Form(...)):
    if name in face_db:
        face_db.pop(name)
        save_face_db(); build_index()
        return {"success":True}
    return {"success":False, "error":"name not found"}

# Recognition using FAISS
@app.post("/v1/vision/face/recognize")
async def recognize(key: str = Depends(verify_header), image: UploadFile = File(...), min_conf: float = None):
    await init_face_analysis()
    img = load_image_bytes(await image.read(), image.content_type)
    faces = tface.get(img)
    result = []
    for face in faces:
        conf = float(face.det_score)
        if min_conf and conf < min_conf: continue
        emb = face.normed_embedding.astype('float32')
        faiss.normalize_L2(emb.reshape(1,-1))
        if index:
            D, I = index.search(emb.reshape(1,-1),1)
            sim = float(D[0][0]); user = id_map[I[0][0]] if sim>=match_threshold else "unknown"
        else:
            sim, user = 0.0, "unknown"
        x1,y1,x2,y2 = map(int, face.bbox)
        result.append({"x_min":x1,"y_min":y1,"x_max":x2,"y_max":y2,"confidence":conf,"userid":user})
    return {"success":True, "predictions":result}

# Match two images
@app.post("/v1/vision/face/match")
async def match_faces(key: str = Depends(verify_header), image1: UploadFile = File(...), image2: UploadFile = File(...)):
    await init_face_analysis()
    img1 = load_image_bytes(await image1.read(), image1.content_type)
    img2 = load_image_bytes(await image2.read(), image2.content_type)
    f1 = tface.get(img1); f2 = tface.get(img2)
    if not f1 or not f2: raise HTTPException(400,"No face detected")
    e1,e2 = f1[0].normed_embedding, f2[0].normed_embedding
    sim = float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
    return {"success":True, "similarity":sim}

# Image loader
def load_image_bytes(buffer, ct):
    im = Image.open(BytesIO(buffer))
    if getattr(im, 'is_animated', False): im.seek(0)
    rgb = im.convert('RGB')
    arr = np.array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h,w = bgr.shape[:2]
    if w>10000 or h>10000: raise HTTPException(400,"Image too large")
    return bgr

# Restart
@app.post("/restart")
async def restart(key: str = Depends(verify_header)):
    os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__":
    uvicorn.run("server:app",host="0.0.0.0",port=http_port)
