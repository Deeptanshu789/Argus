# Node supervisor + Python inference sidecars in ONE image. They must
# co-locate: worker/ingest.ts spawns ml/sidecar.py as a child process, and that
# pipe is the only place Python and TypeScript meet.
FROM node:22-slim

# libgl1 and libglib2.0-0 are OpenCV's runtime; ffmpeg is what lets it open an
# RTSP or MJPEG stream, which is how a paired phone reaches the sidecar.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# THE PIP UPGRADE IS LOAD-BEARING, not housekeeping. Debian bookworm ships pip
# 23.0.1, which rejects the typing_extensions wheel over a name-normalisation
# bug ("expected 'typing-extensions', but metadata has 'typing_extensions'"),
# falls back to the source distribution, and then cannot fetch flit_core to
# build it because --index-url has restricted it to the torch index. The build
# fails several minutes in, with an error naming neither pip nor Debian.
RUN python3 -m pip install --break-system-packages --no-cache-dir --upgrade pip

# THE CPU TORCH INDEX IS NOT OPTIONAL EITHER. PyPI's default wheel bundles
# ~2.5 GB of CUDA runtime that a VPS without an NVIDIA card cannot use; the CPU
# wheel is ~200 MB and is what the sidecar actually runs.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY ml/requirements.txt ml/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r ml/requirements.txt

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

ENV NODE_ENV=production
# The sidecar is a system python here, not the repo's .venv.
ENV ARGUS_PYTHON=python3

CMD ["npm", "run", "worker"]
