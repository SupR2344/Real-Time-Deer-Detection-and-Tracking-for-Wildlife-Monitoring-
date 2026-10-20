FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for OpenCV and media handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (standard cloud containers; for GPU use an nvidia/cuda base)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install (strictly binary wheels, no source compilation)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy source code and model
COPY . .

# Expose backend port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV YOLO_DEVICE=cpu

# Command to run FastAPI server (respects dynamic cloud $PORT e.g. on Railway/Render)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
