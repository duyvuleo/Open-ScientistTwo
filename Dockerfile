ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-worker.txt /requirements-worker.txt
RUN pip install --no-cache-dir -r /requirements-worker.txt
# Estimators are available, but the public fixture's full labels must not be
# recoverable through sklearn.datasets inside the candidate sandbox.
RUN python -c "import pathlib, shutil, sklearn.datasets; shutil.rmtree(pathlib.Path(sklearn.datasets.__file__).parent / 'data')"
USER 65534:65534
WORKDIR /tmp
CMD ["python", "/worker.py"]
