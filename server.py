from dotenv import load_dotenv
import os
import sys
import asyncio
import gc
import json
import base64
import requests
import faiss
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header, Form, Body, Query, Request
from pydantic import BaseModel
import uvicorn
from insightface.utils import storage
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()
app = FastAPI()

# Config
API_KEY = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
PORT = int(os.getenv("HTTP_PORT", 17866))
BACKEND = os.getenv("DETECTOR_BACKEND", "onnx")
MODEL = os.getenv("RECOGNITION_MODEL", "buffalo_l")
THRESH = float(os.getenv("DETECTION_THRESH", 0.65))
LOAD_DELAY = int(os.getenv("MODEL_LOAD_DELAY", 60))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 1200))
MATCH_THRESH = float(os.getenv("MATCH_THRESHOLD", 0.65))
DB_FILE = os.getenv("FACE_DB_FILE", "face_db.json")

storage.BASE_REPO_URL = os.getenv(
    "MODEL_BASE_URL",
    "https://ghproxy.com/https://github.com/kqstone/"
    "mt-photos-insightface-unofficial/releases/download/models"
)
# Globals
face_model = None
unload_task = None
index = None
id_map = []  # index->userid
face_db = {}

# Payload models
class RecognizePayload(BaseModel):
    url: str = None
    image: str = None  # base64
    min_confidence: float = None

class MatchPayload(BaseModel):
    image1: str = None
    image2: str = None

class DetectPayload(BaseModel):
    url: str = None
    image: str = None  # base64
    min_confidence: float = None

async def init_model():
    global face_model
    if face_model is None:
        if BACKEND in ('onnx','cpu'):
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logging.info(f"Loading {MODEL} with {providers}")
        fa = FaceAnalysis(providers=providers,
                         allowed_modules=['detection','recognition'],
                         name=MODEL)
        fa.prepare(ctx_id=0, det_thresh=THRESH, det_size=(640,640))
        face_model = fa

async def delayed_init():
    await asyncio.sleep(LOAD_DELAY)
    await init_model()

@app.on_event("startup")
def on_startup():
    asyncio.create_task(delayed_init())
    load_db(); build_index()

# Unload
def unload_model():
    global face_model
    if face_model:
        logging.info("Unloading model")
        del face_model; face_model=None; gc.collect()

async def idle_unload():
    await asyncio.sleep(IDLE_TIMEOUT)
    unload_model()

@app.middleware("http")
async def refresh_idle_timer(req, call_next):
    global unload_task
    if unload_task: unload_task.cancel()
    unload_task = asyncio.create_task(idle_unload())
    return await call_next(req)

# DB persistence
def load_db():
    global face_db
    if os.path.exists(DB_FILE):
        with open(DB_FILE,'r') as f: face_db = json.load(f)
    else:
        face_db = {}

def save_db():
    with open(DB_FILE,'w') as f: json.dump(face_db,f)

# FAISS index
def build_index():
    global index, id_map
    vecs=[]; id_map=[]
    for user, vs in face_db.items():
        for v in vs:
            vecs.append(v); id_map.append(user)
    if vecs:
        mat = np.array(vecs).astype('float32')
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1]); index.add(mat)
    else:
        index = None

# Utility to load image bytes from file, url, or base64
def extract_image_bytes(file: UploadFile = None, b64: str = None, url: str = None):
    if url:
        resp = requests.get(url)
        if resp.status_code != 200: raise HTTPException(400, "URL fetch failed")
        data = resp.content
    elif b64:
        try: data = base64.b64decode(b64)
        except: raise HTTPException(400, "Invalid base64")
    elif file:
        data = file.file.read()
    else:
        raise HTTPException(400, "No image provided")
    try:
        im = Image.open(BytesIO(data))
        if getattr(im,'is_animated',False): im.seek(0)
        arr = np.array(im.convert('RGB'))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except:
        raise HTTPException(400, "Invalid image format")
    h,w = img.shape[:2]
    if w>10000 or h>10000: raise HTTPException(400, "Image too large")
    return img

# DeepStack-compatible endpoints
@app.get("/v1/vision/face/recognize")
@app.post("/v1/vision/face/recognize")
async def recognize(
                    image: UploadFile = File(None),
                    payload: RecognizePayload = Body(None)):
    await init_model()
    url = payload.url if payload else None
    b64 = payload.image if payload else None
    min_conf = payload.min_confidence if payload else None
    img = extract_image_bytes(image if image else None, b64, url)
    faces = face_model.get(img)
    preds = []
    for f in faces:
        c = float(f.det_score)
        if min_conf and c < min_conf: continue
        emb = f.normed_embedding.astype('float32')
        faiss.normalize_L2(emb.reshape(1,-1))
        if index:
            D,I = index.search(emb.reshape(1,-1),1)
            sim = float(D[0][0]); uid = id_map[I[0][0]] if sim>=MATCH_THRESH else "unknown"
        else: sim=0.0; uid="unknown"
        x1,y1,x2,y2 = map(int,f.bbox)
        preds.append({"x_min":x1,"y_min":y1,"x_max":x2,"y_max":y2,"confidence":c,"userid":uid})
    return {"success":True,"predictions":preds}

@app.post("/v1/vision/face/register")
async def register(
                   image: UploadFile = File(...),
                   userid: str = Form(...)):
    await init_model()
    img = extract_image_bytes(image,None,None)
    faces = face_model.get(img)
    if not faces: return {"success":False,"error":"No face detected"}
    vecs = [f.normed_embedding.tolist() for f in faces]
    face_db.setdefault(userid,[]).extend(vecs); save_db(); build_index()
    return {"success":True}

@app.post("/v1/vision/face/list")
async def list_faces():
    return {"success":True,"faces":list(face_db.keys())}

@app.post("/v1/vision/face/delete")
async def delete_face(userid: str = Form(...)):
    if userid in face_db:
        face_db.pop(userid); save_db(); build_index(); return {"success":True}
    return {"success":False,"error":"userid not found"}

@app.post("/v1/vision/face/match")
async def match(
                image1: UploadFile = File(None),
                image2: UploadFile = File(None),
                payload: MatchPayload = Body(None)):
    await init_model()
    if payload and payload.image1 and payload.image2:
        img1 = extract_image_bytes(None,payload.image1,None)
        img2 = extract_image_bytes(None,payload.image2,None)
    else:
        img1 = extract_image_bytes(image1,None,None)
        img2 = extract_image_bytes(image2,None,None)
    f1 = face_model.get(img1); f2 = face_model.get(img2)
    if not f1 or not f2: raise HTTPException(400, "No face detected")
    e1,e2 = f1[0].normed_embedding, f2[0].normed_embedding
    sim = float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
    return {"success":True,"similarity":sim}

# Object detection endpoint for agentdvr
@app.post("/v1/vision/detection")
async def detection(
                    image: UploadFile = File(None),
                    payload: DetectPayload = Body(None)):
    await init_model()
    url = payload.url if payload else None
    b64 = payload.image if payload else None
    min_conf = payload.min_confidence if payload else None
    img = extract_image_bytes(image if image else None, b64, url)
    faces = face_model.get(img)
    preds = []
    for f in faces:
        c = float(f.det_score)
        if min_conf and c < min_conf: continue
        x1,y1,x2,y2 = map(int,f.bbox)
        preds.append({"x_min":x1,"y_min":y1,"x_max":x2,"y_max":y2,"confidence":c,"label":"person"})
    return {"success":True,"predictions":preds}

@app.post("/restart")
async def restart():
    os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=PORT)
