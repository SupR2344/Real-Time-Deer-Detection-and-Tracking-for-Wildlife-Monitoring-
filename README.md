# AI-Based Deer Detection, Tracking & Collision Warning System

> **A multimodal AI and Computer Vision system for detecting deer near roads, tracking their movement, predicting future trajectories, understanding their behavior, and identifying potential collision risks in real time.**

---

## About the Project

Wildlife crossing roads can create serious safety risks, especially in areas where highways pass through forest regions. Simply detecting a deer is not enough — the system needs to understand **where the deer is moving and whether that movement could lead toward the road**.

This project combines **object detection, multi-object tracking, trajectory prediction, geometric risk analysis, and Vision-Language Model reasoning** into a single pipeline.

The core idea is:

```text
Detect → Track → Predict → Analyze → Understand → Warn
```

The system is designed as a modular architecture so that individual components can be improved or replaced independently.

---

# Key Features

* **Custom Deer Detection** using YOLOv11
* **Multi-Object Tracking (MOT)** using ByteTrack
* **Individual Deer Trajectory Tracking**
* **Future Motion Prediction** using Kalman Filter
* **Road-Crossing & Collision Risk Estimation**
* **Behavior Understanding** using Florence-2
* **React + TypeScript** monitoring interface
* **FastAPI** backend for AI pipeline orchestration
* **Grounding DINO-assisted annotation** with manual verification/correction
* Modular and extensible architecture

---

# System Architecture

```text
                    VIDEO INPUT
                         │
                         ▼
                      OpenCV
                         │
                         ▼
                     YOLOv11
                  Deer Detection
                         │
                         ▼
                    Supervision
                         │
                         ▼
                     ByteTrack
                  Multi-Object MOT
                         │
                         ▼
                  Persistent Deer IDs
                         │
                         ▼
                  Trajectory History
                         │
                         ▼
                   Kalman Filter
                  Motion Prediction
                         │
                         ▼
              Collision Risk Engine
                         │
                  ┌──────┴──────┐
                  │             │
                SAFE        POTENTIAL RISK
                                │
                                ▼
                           Deer Crop
                                │
                                ▼
                           Florence-2
                       Behavior Understanding
                                │
                                ▼
                         Final Warning
```

---

# Project Pipeline

## 1. Dataset Creation

The training dataset is created using a **Grounding DINO-assisted annotation workflow**.

```text
Wildlife Videos
      ↓
Frame Extraction
      ↓
Grounding DINO
      ↓
Initial Deer Detections
      ↓
Manual Annotation / Verification
      ↓
YOLO Dataset
      ↓
YOLOv11 Fine-Tuning
```

Grounding DINO is used to generate initial deer detections, while manual annotation and verification are used to correct or refine the labels before training.

This creates a **semi-automated annotation pipeline** rather than relying entirely on manual bounding-box creation.

---

## 2. Real-Time Detection

The trained YOLOv11 model processes incoming video frames and detects deer using:

* Bounding boxes
* Confidence scores
* Class information

Supervision is used for detection processing and visualization.

---

## 3. Multi-Object Tracking

YOLO detects deer independently in each frame, while **ByteTrack** maintains their identity across frames.

Example:

```text
Frame 1 → Deer ID 3
Frame 2 → Deer ID 3
Frame 3 → Deer ID 3
Frame 4 → Deer ID 3
```

This allows the system to maintain a separate movement history for every tracked deer.

---

## 4. Trajectory & Motion Prediction

For every tracked deer, the system stores its previous positions and builds a trajectory.

```text
(100,220)
    ↓
(112,225)
    ↓
(128,231)
    ↓
(141,240)
```

```text

```

This keeps the collision layer lightweight, deterministic, and interpretable.

---

# Florence-2 Behavior Understanding

**Florence-2** is used as an additional Vision-Language Model layer.

It is **not responsible for tracking or trajectory prediction**.

Instead, when the system identifies a potentially dangerous situation, a deer crop can be passed to Florence-2 for behavioral understanding.

For example:

```text
Potential Risk
      ↓
Crop Deer
      ↓
Florence-2
      ↓
Behavior Understanding
```

Possible behavioral observations include:

* Standing
* Running
* Grazing
* Turning toward the road
* Crossing the road

Using Florence-2 selectively prevents the VLM from becoming part of every frame's critical real-time processing path.

---

# Web Application

The system is exposed through a web-based interface.

### Frontend

**React + TypeScript**

Responsible for:

* Video monitoring
* Detection visualization
* Track IDs
* Trajectory display
* Risk status
* Warning display

### Backend

**FastAPI + Python**

Responsible for:

* API endpoints
* Video processing
* Model inference
* Tracking
* Trajectory generation
* Motion prediction
* Collision analysis
* Florence-2 integration

```text
React + TypeScript
        │
        │ API
        ▼
     FastAPI
        │
        ▼
Computer Vision Pipeline
```

---

# Technology Stack

| Component                  | Technology                   |
| -------------------------- | ---------------------------- |
| Frontend                   | React + TypeScript           |
| Backend                    | FastAPI + Python             |
| Video Processing           | OpenCV                       |
| Annotation Assistance      | Grounding DINO               |
| Annotation / Visualization | Supervision                  |
| Object Detection           | YOLOv11                      |
| Multi-Object Tracking      | ByteTrack                    |
| Trajectory                 | OpenCV + Supervision         |
| Motion Prediction          | Kalman Filter                |
| Collision Analysis         | Rule-based Geometry + Motion |
| Behavior Understanding     | Florence-2                   |

---

# Why Multimodal?

The **overall system** can be described as a multimodal AI system because it combines:

* **Visual information** from video/images
* **Temporal information** from tracking and trajectories
* **Spatial/geometric information** for road-risk analysis
* **Vision-Language reasoning** through Florence-2

The individual components are specialized — for example, YOLOv11 performs visual detection and ByteTrack performs tracking — while the complete system combines these different information sources into a single decision pipeline.

---

# Project Structure

```text
deer-detection-system/
│
├── frontend/          # React + TypeScript
│
├── backend/           # FastAPI + Python
│   ├── detection/
│   ├── tracking/
│   ├── trajectory/
│   ├── prediction/
│   ├── collision/
│   └── vlm/
│
├── models/            # Trained model weights
│
├── data/              # Dataset / videos / annotations
│
├── scripts/           # Data preparation & training scripts
│
└── README.md
```

---

# Getting Started

### Backend

```bash
cd backend

python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# End-to-End Flow

```text
Video
  ↓
YOLOv11
  ↓
Deer Detection
  ↓
ByteTrack
  ↓
Persistent IDs
  ↓
Trajectory
  ↓
Florence-2
  ↓
Behavior Understanding
  ↓
Warning
```

---

# Future Improvements

The modular architecture allows the system to be extended with:

* Advanced trajectory prediction models
* Road/lane segmentation
* Vehicle detection and vehicle-aware collision prediction
* Time-to-collision estimation
* Improved deer behavior recognition
* Multi-camera tracking
* Edge-device deployment
* More sophisticated risk scoring

---

# Limitations

The system provides **AI-based risk estimation**, not a guaranteed collision prediction.

Performance can be affected by:

* Poor lighting
* Fog or rain
* Occlusion
* Small or distant deer
* Camera movement
* Sudden changes in deer direction
* Incorrect road-boundary estimation

The warning should therefore be treated as a **decision-support signal** rather than a certainty.

---

# Project Highlights

```text
✓ Custom YOLOv11 Deer Detector
✓ Grounding DINO-Assisted Annotation
✓ Manual Annotation & Verification
✓ ByteTrack Multi-Object Tracking
✓ Deer Trajectory Generation
✓ Florence-2 Behavior Understanding
✓ Multimodal AI Architecture
✓ React + TypeScript Frontend
✓ FastAPI Backend
✓ Real-Time Computer Vision Pipeline
✓ Modular & Extensible Design
```

---

## Detect. Track. Predict. Understand. Warn.

**An AI-powered wildlife monitoring system that moves beyond simply detecting a deer to understanding its movement and identifying potential road-safety risks.**
