# PYTHON_IMAGE must be a resolved Linux Python 3.10 image digest.
ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=2
COPY requirements.lock /requirements.lock
# promise is source-only. Use the digest-pinned base's build tooling rather than
# fetching an unrecorded isolated build environment.
RUN python -m pip install --no-cache-dir --no-build-isolation --require-hashes -r /requirements.lock
USER 65534:65534
WORKDIR /workspace
