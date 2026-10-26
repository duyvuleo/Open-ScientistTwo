ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
USER 65534:65534
WORKDIR /tmp
