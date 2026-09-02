# =============================================================
# Dockerfile - build recipe for the Tenet Clinical AI API.
# Build:  docker build -t tenet-clinical-ai .
# Run:    docker run -p 8000:8000 --env-file .env tenet-clinical-ai
# =============================================================

# 1. Base image: a small official Python. "slim" = minimal Linux + Python.
FROM python:3.11-slim

# 2. Don't buffer stdout/stderr (so logs appear immediately) and don't write
#    .pyc files (keeps the image clean).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Work inside /app. All following paths are relative to this.
WORKDIR /app

# 4. Install CPU-only PyTorch first (smaller than the default CUDA build).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 5. Copy ONLY requirements first, then install. (Docker caches this layer, so
#    changing app code later doesn't reinstall all the libraries.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the application code and sample data into the image.
COPY app/ ./app/
COPY data/ ./data/

# 7. Document the port the API listens on.
EXPOSE 8000

# 8. Start the API server. 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
