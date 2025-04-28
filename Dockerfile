FROM python:3.8.10-buster
USER root

WORKDIR /app
COPY requirements.txt .

# 先升级 pip
RUN pip3 install --upgrade pip

# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

# 单独安装 faiss-cpu 或 faiss-gpu
RUN pip3 install faiss-cpu --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
# 单独安装 faiss-cpu 或 faiss-gpu
#RUN pip3 install faiss-gpu --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV RECOGNITION_MODEL=buffalo_l
ENV DETECTION_THRESH=0.65
ENV MATCH_THRESHOLD=0.65
EXPOSE 17866

VOLUME ["/root/.insightface/models"]

CMD [ "python3", "server.py" ]
