import logging
import logging.config

# 日志配置：全局生效，包括 uvicorn，带时间戳
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
logging.config.dictConfig(LOGGING_CONFIG)

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
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
from pydantic import BaseModel
import uvicorn
from insightface.utils import storage
from insightface.app import FaceAnalysis

# 加载环境变量
load_dotenv()
app = FastAPI()

# 配置
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

# 全局
face_model = None
unload_task = None
index = None
id_map = []
face_db = {}

# Payload
class RecognizePayload(BaseModel):
    url: str = None
    image: str = None
    min_confidence: float = None

# 启动卸载逻辑
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
    await asyncio.sleep(IDLE_TIMEOUT); unload_model()

@app.middleware("http")
async def refresh_idle_timer(request, call_next):
    global unload_task
    if unload_task: unload_task.cancel()
    unload_task = asyncio.create_task(idle_unload())
    return await call_next(request)

# 模型初始化
async def init_model():
    global face_model
    if face_model is None:
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if BACKEND in ('onnx','cpu') else ['CUDAExecutionProvider','CPUExecutionProvider']
        logging.info(f"Loading {MODEL} with {providers}")
        fa = FaceAnalysis(
            providers=providers,
            allowed_modules=['detection','recognition','genderage'],
            name=MODEL
        )
        fa.prepare(ctx_id=0, det_thresh=THRESH, det_size=(640,640))
        face_model = fa

async def delayed_init(): await asyncio.sleep(LOAD_DELAY); await init_model()

# DB

def load_db():
    global face_db
    if os.path.exists(DB_FILE): face_db = json.load(open(DB_FILE))
    else: face_db = {}

def save_db(): json.dump(face_db, open(DB_FILE,'w'))

# FAISS 索引

def build_index():
    global index,id_map
    vecs=[]; id_map=[]
    for u,vs in face_db.items():
        for v in vs: vecs.append(v); id_map.append(u)
    if vecs:
        mat=np.array(vecs,dtype='float32'); faiss.normalize_L2(mat)
        index=faiss.IndexFlatIP(mat.shape[1]); index.add(mat)
    else: index=None

# 图片提取

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
    # 签名校验
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
    draw=ImageDraw.Draw(im); font=ImageFont.load_default()
    for f in faces:
        score=float(f.det_score)
        if payload and payload.min_confidence and score<payload.min_confidence: continue
        emb=f.normed_embedding.astype('float32'); faiss.normalize_L2(emb.reshape(1,-1))
        if index:
            D,I=index.search(emb.reshape(1,-1),1); sim=float(D[0][0]); uid=id_map[I[0][0]] if sim>=MATCH_THRESH else 'unknown'
        else: sim=0.0; uid='unknown'
        x1,y1,x2,y2=map(int,f.bbox)
        # 年龄 & 性别 强制转换为Python类型
        age=int(float(f.age)) if hasattr(f,'age') else None
        gender=int(f.gender) if hasattr(f,'gender') else None
        preds.append({'x_min':x1,'y_min':y1,'x_max':x2,'y_max':y2,'confidence':score,'userid':uid,'age':age,'gender':gender})
        draw.rectangle([x1,y1,x2,y2],outline='green',width=2)
        label=f"{gender},{age}" if age!=None and gender!=None else ''
        draw.text((x1,y1-10),label,font=font,fill='green')
    result={'success':True,'predictions':preds}
    return result

if __name__=="__main__": uvicorn.run(app,host="0.0.0.0",port=PORT)
