# GPU build: CUDA 12.1 base so torch's cu121 wheels have a matching CUDA/cuDNN
# userspace. The NVIDIA driver itself is supplied by the GKE node
# (gpu-driver-version=default); this image provides everything above the driver.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Python 3.10 (Ubuntu 22.04 default) + pip. No compilers needed: every wheel
# below is prebuilt for cu121.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# General dependencies (tqdm included for the progress bars).
RUN python -m pip install --no-cache-dir \
    networkx==3.1 \
    numpy==1.23.5 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    scipy==1.11.4 \
    statsmodels==0.14.0 \
    Unidecode==1.3.7 \
    tqdm==4.66.1 \
    google-cloud-storage
    
# PyTorch + PyG, CUDA 12.1 builds. torch will report "2.1.0+cu121".
RUN python -m pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir torch-geometric==2.4.0 && \
    python -m pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy code last so code changes don't bust the heavy dependency layers.
COPY . .

# Non-root user; /code -> /app symlink keeps the original relative paths working.
RUN useradd -m appuser && mkdir -p /data /results && ln -s /app /code && chown -R appuser /app /data /results
USER appuser

CMD ["python", "train.py"]