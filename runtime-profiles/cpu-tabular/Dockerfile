# PYTHON_IMAGE must be a resolved Linux Python 3.12 image digest.
ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
COPY requirements.lock /requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /requirements.lock
USER 65534:65534
WORKDIR /workspace
