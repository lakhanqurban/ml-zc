FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
# Only the runtime requirements are installed - the notebook tooling lives in
# requirements-dev.txt and would add ~700 MB to the image.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# transformers.py is required to unpickle the CountryGrouper inside model.pkl
# healthcheck.py is used by the HEALTHCHECK below.
COPY features.py transformers.py serve.py predict.py healthcheck.py ./
COPY artifacts ./artifacts

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "healthcheck.py"]

# PORT defaults to 9696 locally; Render and Fly.io inject their own.
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-9696}"]
