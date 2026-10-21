FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and media handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000) for Hugging Face Spaces & security
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install PyTorch CPU and pinned NumPy 1.26.4 together
RUN pip install --no-cache-dir numpy==1.26.4 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80 numpy==1.26.4

# Verify all key imports during build to guarantee zero runtime collisions
RUN python -c "import cv2; import torch; import ultralytics; import shapely; print('Build verification: all AI/CV imports OK!')"

# Copy source code and model
COPY --chown=user:user . .

# Ensure upload and output directories are writable by non-root user
RUN mkdir -p /app/videos/uploads /app/outputs /home/user/.cache && \
    chown -R user:user /app /home/user

USER user

# Expose backend port
EXPOSE 8000

# Set environment variables (defaults to 8000)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV YOLO_DEVICE=cpu

# Command to run FastAPI server (respects dynamic cloud $PORT injected by Render)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
