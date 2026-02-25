FROM runpod/pytorch:2.6.0-py3.10-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ARG CONTROLCOM_REF=HEAD

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN set -eux; \
    mkdir -p /app/vendor; \
    git clone --depth 1 https://github.com/bcmi/ControlCom-Image-Composition /app/vendor/controlcom; \
    if [ "${CONTROLCOM_REF}" != "HEAD" ]; then \
      git -C /app/vendor/controlcom fetch --depth 1 origin "${CONTROLCOM_REF}"; \
      git -C /app/vendor/controlcom checkout FETCH_HEAD; \
    fi

COPY worker/ /app/worker/

ENV PYTHONPATH="/app/worker"
ENV MODEL_CACHE_DIR="/runpod-volume/models"
ENV HF_HOME="/runpod-volume/hf_cache"
ENV TRANSFORMERS_CACHE="/runpod-volume/hf_cache"

HEALTHCHECK --interval=30s --timeout=60s --start-period=60s --retries=3 \
  CMD python -c "from pathlib import Path; import torch; import transformers; import diffusers; import libcom; import cv2; assert Path('/app/vendor/controlcom/scripts/inference.py').exists(); assert torch.cuda.is_available(); assert torch.__version__.startswith('2.6.'); assert hasattr(cv2, 'ximgproc'); print('ok')"

CMD ["python", "-u", "/app/worker/handler.py"]
