# DeerGuard AI — Architecture & Technical Specification

## 1. System Overview

DeerGuard AI is a real-time, edge-capable wildlife detection, multi-object tracking, trajectory prediction, and collision warning system designed for road safety in proximity to forested and highway corridors.

The system is architected as an authoritative end-to-end pipeline where:
- The backend owns all stream processing, model inference, tracking, trajectory estimation, and risk logic.
- The frontend operates strictly as an interactive monitoring, telemetry, and control dashboard consuming verified runtime state.

---

## 2. End-to-End Pipeline Architecture

```
[ Camera Source ]
  (Video File / USB Webcam / IP Camera RTSP)
        ↓
  [ CameraManager ]
  (Status: CONNECTED / STREAMING / RECONNECTING)
        ↓
  [ Frame Capture & Preprocessing ]
        ↓
  [ YOLOv11 Deer Detector ]
  (model/best.pt, CUDA-accelerated with CPU fallback)
        ↓
  [ ByteTrack Multi-Object Tracker ]
  (Persistent Track IDs across frames)
        ↓
  [ TrajectoryManager ]
  (Bounded historical center points: P1 → P2 → ... → Pn)
        ↓
  [ Kalman Filter Prediction Manager ]
  (Constant-velocity 2D motion model: [x, y, vx, vy] → Future Position)
        ↓
  [ Road-Zone Collision Engine ]
  (Polygon intersection, boundary proximity, trajectory vector)
        ↓
  [ Risk Assessment: SAFE / CAUTION / HIGH_RISK / UNKNOWN ]
        ↓
  [ BLIP Vision-Language Model ]
  (Lazy-loaded, triggered ONLY on genuine HIGH_RISK events, per-track cooldown)
        ↓
  [ Authoritative RuntimeState ]
  (Thread-safe in-memory state: tracks, annotated JPEG, telemetry, events)
        ↓
  [ FastAPI Backend ] (Port 8000)
  (REST APIs: /tracks, /runtime, /source, /road-zone, /reset + /video/stream MJPEG)
        ↓
  [ React + TypeScript Command Center Dashboard ]
  (Live video overlay, interactive road polygon editor, telemetry cards, track inspector)
```

---

## 3. Core Modules & Responsibilities

### 3.1 Detection (`detection/detector.py`)
- Loads `model/best.pt` (custom YOLOv11s architecture).
- Inspects `model.names` metadata dynamically:
  - Validates class 0 (`deer`).
- Applies configurable `YOLO_CONFIDENCE` (default 0.3) and `YOLO_IOU` (default 0.5).
- Output: normalized detection objects with bounding boxes, confidences, class IDs, and labels.

### 3.2 Multi-Object Tracking (`bytetrack/adapter.py`, `bytetrack/`)
- Associates bounding boxes across consecutive frames using low- and high-confidence association cascades.
- Maintains persistent, stable track IDs.
- Recovers lost tracks within configurable buffer windows (`lost_track_buffer=30`).

### 3.3 Trajectory History (`trajectory/manager.py`)
- Tracks temporal progression of deer centers using a bounded history ring-buffer (`MAX_HISTORY=30`).
- Prunes stale or inactive track records to maintain bounded memory footprint.

### 3.4 Kalman Prediction (`prediction/kalman_filter.py`, `prediction/manager.py`)
- State vector: \([x, y, v_x, v_y]^T\).
- Measurement vector: \([x, y]^T\).
- Forecasts future position after configurable horizon (\(N=5\) frames).
- Derives instant scalar velocity \(\|v\| = \sqrt{v_x^2 + v_y^2}\) in pixels per frame.

### 3.5 Collision Engine (`collision/engine.py`, `collision/geometry.py`)
- Evaluates spatial proximity and trajectory vector relative to road-zone polygon:
  - **UNKNOWN**: When road zone polygon is missing or not configured. Never treated as SAFE.
  - **HIGH_RISK**: Deer is currently inside road polygon, predicted inside road polygon, trajectory line directly intersects road polygon with speed, or within high-risk distance threshold.
  - **CAUTION**: Deer is within caution distance threshold of road boundary or heading toward road.
  - **SAFE**: Deer is far from road polygon and trajectory does not approach road.

### 3.6 Vision-Language Model (`vlm/manager.py`)
- Uses `Salesforce/blip-image-captioning-base`.
- Lazy-loaded: model weights are loaded into GPU/CPU memory ONLY when a genuine `HIGH_RISK` event occurs.
- Per-track cooldown (`VLM_COOLDOWN=30s`) prevents redundant inference on the same deer.
- Crop resolution guard ensures tiny or corrupted bounding boxes are not passed to BLIP.

### 3.7 Camera Abstraction (`camera/`)
- `CameraSource` abstract base class.
- `VideoSource`: Supports local video files (`videos/deer_test.mp4`), automatic looping for testing.
- `WebcamSource`: USB or integrated camera support via DirectShow / Video4Linux.
- `RTSPSource`: Real IP camera streams with URL credential masking and timeout guards.
- `CameraManager`: Unified source switching, automatic reconnection with configurable retry limits and delays.

### 3.8 Unified Runtime State & Backend (`backend/app/`)
- `RuntimeState`: Thread-safe shared state with mutex synchronization.
- Single background processing loop feeding MJPEG stream and telemetry.
- REST API and MJPEG stream serving both root and `/api/*` routes.
- Production serving of built React static distribution (`frontend/dist`).

---

## 4. Hardware Acceleration

- **GPU**: NVIDIA CUDA (e.g. GeForce RTX 3050 Laptop GPU, PyTorch 2.6.0+cu124).
- **CPU Fallback**: Fully supported on non-GPU workstations without crashes.
