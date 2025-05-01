import logging
import datetime
import pytz
import getpass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def china_time(*args):
    return datetime.datetime.now(pytz.timezone('Asia/Shanghai')).timetuple()

logging.Formatter.converter = china_time

from dotenv import load_dotenv
import os
import asyncio
import gc
import json
import base64
import requests
import faiss
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Query
from pydantic import BaseModel
import uvicorn
from insightface.utils import storage
from insightface.app import FaceAnalysis

load_dotenv()
app = FastAPI()

PORT = int(os.getenv("HTTP_PORT", 17866))
BACKEND = os.getenv("DETECTOR_BACKEND", "onnx")
MODEL = os.getenv("RECOGNITION_MODEL", "buffalo_l")
THRESH = float(os.getenv("DETECTION_THRESH", 0.65))
LOAD_DELAY = int(os.getenv("MODEL_LOAD_DELAY", 60))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 1200))
MATCH_THRESH = float(os.getenv("MATCH_THRESHOLD", 0.65))
DB_FILE = os.getenv("FACE_DB_FILE", "face_db.json")

# 根据当前用户设置模型路径
current_user = getpass.getuser()
if current_user == "root":
    model_path = os.path.join("/root/.insightface/models", MODEL)
else:
    model_path = os.path.join(os.path.expanduser("~"), ".insightface/models", MODEL)

storage.BASE_REPO_URL = os.getenv(
    "MODEL_BASE_URL",
    "https://github.com/kqstone/"
    "mt-photos-insightface-unofficial/releases/download/models"
)

face_model = None
unload_task = None
index = None
id_map = []
face_db = {}

class RecognizePayload(BaseModel):
    url: str = None
    image: str = None
    min_confidence: float = None

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(delayed_init())
    load_db(); build_index()

def unload_model():
    global face_model
    if face_model:
        logging.info("Unloading model")
        del face_model; face_model = None; gc.collect()

async def idle_unload():
    try:
        await asyncio.sleep(IDLE_TIMEOUT)
        unload_model()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.error(f"Error in idle unload: {str(e)}")

@app.middleware("http")
async def refresh_idle_timer(request, call_next):
    global unload_task
    if unload_task: 
        unload_task.cancel()
    response = await call_next(request)
    unload_task = asyncio.create_task(idle_unload())
    return response

async def init_model():
    global face_model
    if face_model is None:
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if BACKEND in ('onnx','cpu') else ['CUDAExecutionProvider','CPUExecutionProvider']
        logging.info(f"progress run in {providers}...")
        try:
            fa = FaceAnalysis(
                providers=providers,
                allowed_modules=['detection','recognition'],
                name=MODEL
            )
            fa.prepare(ctx_id=0, det_thresh=THRESH, det_size=(640,640))
            face_model = fa
            logging.info(f"模型 {MODEL} 加载成功")
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            logging.error(f"{model_path}模型不存在，请手动下载模型")
            logging.error(f"地址 https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models/{MODEL}.zip")
            logging.error(f"下载 {MODEL}.zip，解压至{model_path}")
            logging.error(f"如果是docker容器，则需要映射外部模型路径到 {model_path}")
            exit(1)

async def delayed_init(): await asyncio.sleep(LOAD_DELAY); await init_model()

def load_db():
    global face_db
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    face_db = json.loads(content)
                else:
                    face_db = {}
                    logging.warning("face_db.json is empty")
        else:
            face_db = {}
            logging.warning(f"face_db.json file {DB_FILE} not found")
    except json.JSONDecodeError as e:
        logging.error(f"face_db.json file format error: {str(e)}")
        face_db = {}
    except Exception as e:
        logging.error(f"load face_db.json file error: {str(e)}")
        face_db = {}

def save_db():
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(face_db, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"save face_db.json file error: {str(e)}")

def build_index():
    global index,id_map
    vecs=[]; id_map=[]
    for u,vs in face_db.items():
        for v in vs: vecs.append(v); id_map.append(u)
    if vecs:
        mat=np.array(vecs,dtype='float32'); faiss.normalize_L2(mat)
        index=faiss.IndexFlatIP(mat.shape[1]); index.add(mat)
    else: index=None

def extract_image_bytes(file: UploadFile=None, b64: str=None, url: str=None):
    if url:
        resp=requests.get(url)
        if resp.status_code!=200: raise HTTPException(400,"URL fetch失败")
        data=resp.content
    elif b64:
        b64=b64.strip().strip('"')
        if b64.startswith("data:") and ',' in b64: b64=b64.split(',',1)[1]
        data=base64.b64decode(b64)
    elif file:
        raw=file.file.read()
        try:
            d=json.loads(raw)
            img_b64=d.get('image','').strip().strip('"')
            if img_b64.startswith("data:"): img_b64=img_b64.split(',',1)[1]
            data=base64.b64decode(img_b64)
        except:
            data=raw
    else:
        raise HTTPException(400,"No image provided")
    sig=data[:4]
    if not (sig.startswith(b'\xff\xd8') or sig==b'\x89PNG'): raise HTTPException(400,"Invalid image format")
    return data

# Recognize
@app.post("/v1/vision/face/recognize")
async def recognize(image:UploadFile=File(None), payload:RecognizePayload=Body(None)):
    await init_model()
    data=extract_image_bytes(image if image else None, payload.image if payload else None, payload.url if payload else None)
    im=Image.open(BytesIO(data));
    if getattr(im,'is_animated',False): im.seek(0)
    img_cv=cv2.cvtColor(np.array(im.convert('RGB')),cv2.COLOR_RGB2BGR)

    faces=face_model.get(img_cv)
    preds=[]
    draw=ImageDraw.Draw(im)
    for f in faces:
        score=float(f.det_score)
        if payload and payload.min_confidence and score<payload.min_confidence: continue
        emb=f.normed_embedding.astype('float32'); faiss.normalize_L2(emb.reshape(1,-1))
        if index:
            D,I=index.search(emb.reshape(1,-1),1); sim=float(D[0][0]); uid=id_map[I[0][0]] if sim>=MATCH_THRESH else 'unknown'
        else: sim=0.0; uid='unknown'
        x1,y1,x2,y2=map(int,f.bbox)
        preds.append({'x_min':x1,'y_min':y1,'x_max':x2,'y_max':y2,'confidence':score,'userid':uid})
        draw.rectangle([x1,y1,x2,y2],outline='green',width=2)
    result={'success':True,'predictions':preds}
    return result
    
@app.post("/v1/vision/face/register")
async def register(
                   image: UploadFile = File(...),
                   userid: str = Form(...)):
    await init_model()
    data = extract_image_bytes(image, None, None)
    im = Image.open(BytesIO(data))
    if getattr(im, 'is_animated', False): im.seek(0)
    img_cv = cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR)
    faces = face_model.get(img_cv)
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

if __name__=="__main__": uvicorn.run(app,host="0.0.0.0",port=PORT)
