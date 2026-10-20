---
title: DeerGuard AI Backend
emoji: 🦌
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# DEERGUARD AI — AI-Based Deer Detection, Tracking, Trajectory Prediction & Collision Warning System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-green.svg)](https://pytorch.org/)
[![Ultralytics YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-orange.svg)](https://docs.ultralytics.com/)
[![React 18 TypeScript](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript%20%7C%20Vite-cyan.svg)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)

A comprehensive, production-style, real-time command center and edge safety monitoring system that detects deer near roadways and highway corridors, tracks individual animals across frames with persistent IDs using ByteTrack, predicts future movement trajectories using Kalman filtering, evaluates real-time collision hazards against configurable road-zone polygons, and triggers a Vision-Language Model (Salesforce BLIP) exclusively for genuine high-risk events.

---

## 1. Project Overview

Wildlife-vehicle collisions pose severe road safety and ecological concerns. DeerGuard AI delivers an authoritative, end-to-end computer vision and warning pipeline:
- **Continuous Detection**: Custom YOLOv11s model trained for deer recognition under diverse lighting conditions.
- **Multi-Object Tracking**: ByteTrack implementation tracking multiple deer simultaneously with persistent IDs and trajectory history.
- **Motion & Trajectory Estimation**: Per-track constant-velocity Kalman filter forecasting positions \(N=5\) frames ahead.
- **Road-Zone Collision Engine**: Spatial analysis against user-defined polygon geometry yielding deterministic risk states: `SAFE`, `CAUTION`, `HIGH_RISK`, `UNKNOWN`.
- **Vision-Language Intelligence**: Lazy-loaded BLIP image captioning running on high-risk crops with per-track cooldowns.
- **Unified Real-Time Dashboard**: Single-page command center showing annotated video feeds, interactive polygon editor, and live telemetry.

---

## 2. System Architecture

```
Camera / Video Source (Local MP4 / USB Webcam / IP Camera RTSP)
        ↓
  CameraManager (Unified capture & auto-reconnect)
        ↓
  YOLOv11 Custom Deer Detector (model/best.pt)
        ↓
  ByteTrack Multi-Object Tracker (Persistent IDs)
        ↓
  TrajectoryManager (Bounded historical positions)
        ↓
  Per-Track Kalman Filter (Velocity & future coordinate estimation)
        ↓
  Road-Zone Collision Engine (SAFE / CAUTION / HIGH_RISK / UNKNOWN)
        ↓
  [HIGH_RISK only] → BLIP Vision-Language Model (Scene captioning)
        ↓
  Authoritative RuntimeState (Thread-safe in-memory state)
        ↓
  FastAPI Backend (Port 8000) ─── Live MJPEG (/video/stream) & REST APIs
        ↓
  React 18 + Vite Command Center Dashboard (Port 5173 / Port 8000)
```

---

## 3. Hardware & Software Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 22.04+), or macOS.
- **Python**: Version 3.10 to 3.12 (Python 3.12 verified).
- **Node.js**: Version 18.x or 20.x (with npm).
- **GPU (Recommended)**: NVIDIA GPU with CUDA 12.x support (verified on NVIDIA GeForce RTX 3050 Laptop GPU with CUDA 12.4).
- **CPU Fallback**: Automatic CPU fallback supported if no CUDA GPU is detected.

---

## 4. Installation & Environment Setup

### 4.1 Clone / Open Workspace
```powershell
cd C:\Users\supri\Desktop\wiserli
```

### 4.2 Python Virtual Environment
```powershell
# Create virtual environment (if not already present)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate
```

### 4.3 Install Python Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA (if using NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all project requirements
pip install -r requirements.txt
```

---

## 5. Frontend Setup

```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## 6. Model Verification & Setup

Place your trained YOLOv11 model at:
```
model/best.pt
```
Verify the model and CUDA detection before running the application:
```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m = YOLO('model/best.pt'); print('Model OK:', m.names)"
```
Expected output:
```text
Model OK: {0: 'deer'}
```

---

## 7. Environment Configuration (.env)

Copy `.env.example` to `.env` or edit the existing `.env`:
```ini
HOST=0.0.0.0
PORT=8000

# Path to YOLO model
MODEL_PATH=model/best.pt
YOLO_CONFIDENCE=0.30
YOLO_IOU=0.50
YOLO_IMG_SIZE=640
YOLO_DEVICE=cuda

# Camera Source Configuration (video, webcam, or rtsp)
CAMERA_TYPE=video
CAMERA_SOURCE=videos/deer_test.mp4
CAMERA_TIMEOUT=10
CAMERA_RECONNECT_DELAY=5
CAMERA_RECONNECT_ATTEMPTS=5

# Road Zone Polygon (JSON coordinates or configure in UI)
# ROAD_ZONE=[[100,50],[500,50],[500,350],[100,350]]
ROAD_ZONE=""
ROAD_DISTANCE_THRESHOLD=20.0
CAUTION_DISTANCE=50.0

# Speed thresholds (pixels per frame)
CAUTION_SPEED=2.0
HIGH_RISK_SPEED=5.0
PREDICTION_HORIZON=5
MAX_HISTORY=30

# Vision-Language Model (BLIP)
VLM_ENABLED=true
VLM_MODEL=Salesforce/blip-image-captioning-base
VLM_COOLDOWN=30
```

---

## 8. Running the Application

### 8.1 Production Mode (Single Unified Server)
In production mode, FastAPI serves both the REST APIs, live MJPEG stream, and the compiled React dashboard at `http://localhost:8000`:

```powershell
# 1. Build frontend
cd frontend
npm run build
cd ..

# 2. Start backend
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Open your browser at:
- **Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 8.2 Development Mode (Hot-Reloading)
Run backend and frontend in separate terminals:

**Terminal 1 (Backend):**
```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## 9. Camera Source Instructions

### 9.1 Local Test Video
```text
Type: Local Video File
Source: videos/deer_test.mp4
```
Automatically loops video seamlessly for continuous testing.

### 9.2 USB / Laptop Webcam
```text
Type: USB / Integrated Webcam
Source: 0  (or 1 for external USB webcam)
```

### 9.3 IP Camera / RTSP
```text
Type: IP Camera / RTSP Stream
Source: rtsp://username:password@192.168.1.100:554/stream
```
Credentials in RTSP URLs are automatically masked in logs and telemetry payloads.

---

## 10. Configuring Road Zones

1. Open the dashboard at [http://localhost:8000](http://localhost:8000).
2. Click **Configure Road Zone** on the right panel.
3. Click directly on the video player to add polygon vertices along the road surface.
4. Use **Undo** to remove the last point or **Clear** to restart.
5. Click **Save Polygon**.
6. The Collision Engine will immediately transition to `ACTIVE` and evaluate real-time hazard status for every tracked deer!

---

## 11. Testing & Validation

### 11.1 Run Comprehensive Unit & Integration Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_system.py
```
Validates:
- Trajectory lifecycle and pruning
- Kalman filter 2D velocity and horizon prediction
- Polygon geometry intersection and distance
- Collision engine risk states (`SAFE`, `CAUTION`, `HIGH_RISK`, `UNKNOWN`)
- VLM lazy loading and cooldown guard
- Camera sources and CameraManager lifecycle
- Thread-safe RuntimeState mutations
- FastAPI REST endpoints

### 11.2 Run Smoke Tests
```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

### 11.3 Run End-to-End (E2E) Video Validation
```powershell
.\.venv\Scripts\python.exe scripts/e2e_validation.py --source videos/deer_test.mp4 --frames 120
```
Generates:
- `outputs/e2e_output.mp4`: Real annotated video showing bounding boxes, IDs, trajectories, and risk tags.
- `outputs/e2e_report.json`: Factual metrics on detection counts, track IDs, Kalman predictions, and final status.

---

## 12. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server uptime and health status |
| `GET` | `/runtime` | Authoritative system telemetry, FPS, components, risk counts |
| `GET` | `/tracks` | Active tracks with bbox, confidence, speed, position, prediction, risk |
| `GET` | `/source` | Current camera source configuration and connection status |
| `POST` | `/source` | Switch camera source (`type`: video/webcam/rtsp, `source`: string) |
| `POST` | `/source/stop` | Stop active camera stream |
| `GET` | `/road-zone` | Fetch configured road polygon vertices |
| `POST` | `/road-zone` | Save new road polygon (`points`, `frame_width`, `frame_height`) |
| `DELETE`| `/road-zone` | Clear active road polygon |
| `POST` | `/reset` | Reset tracking state and telemetry counters |
| `GET` | `/video/stream` | Multipart/x-mixed-replace live annotated MJPEG stream |

---

## 13. License

Academic and research use under the MIT License.
