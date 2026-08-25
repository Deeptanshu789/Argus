# Node supervisor + Python inference sidecar in one image. They must co-locate:
# worker/ingest.ts spawns ml/sidecar.py as a child process.
FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/requirements.txt ml/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r ml/requirements.txt

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
CMD ["npm", "run", "worker"]
