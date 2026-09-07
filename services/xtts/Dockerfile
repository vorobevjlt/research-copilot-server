FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_DATA_HOME=/data/models \
    VOICE_DATA_DIR=/data/voices

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY song_convert.py ./
COPY speech_convert.py ./

RUN useradd --create-home --uid 10001 xtts \
    && mkdir -p /data/models /data/voices /data/song-jobs \
    && chown -R xtts:xtts /data /app

USER xtts

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
