FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg needed by moviepy, and libgl/libglib for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy your model + config + scripts
COPY reelmaker.py deploy.prototxt res10_300x300_ssd_iter_140000_fp16.caffemodel handler.py ./

CMD ["python", "handler.py"]
