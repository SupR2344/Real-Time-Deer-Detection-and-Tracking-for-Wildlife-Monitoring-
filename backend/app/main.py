# -*- coding: utf-8 -*-
"""FastAPI backend entry point — DeerGuard AI.

The application launches ONE background thread that:
  1. Reads frames from the configured camera / video source via CameraManager.
  2. Runs the full AI pipeline (YOLOv11 -> ByteTrack -> Kalman -> Collision -> BLIP).
  3. Stores the latest annotated JPEG frame + track metadata in RuntimeState.

FastAPI endpoints serve that state to the React frontend.
Endpoints are mounted both at root (/*) and at (/api/*) for seamless frontend dev & prod support.
"""

from __future__ import annotations

import logging
import threading
import time
import json
from pathlib import Path
from typing import Generator, Optional, List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.config import get, CONFIG, parse_road_zone
from pipeline.process_frame import PipelineProcessor
from backend.app.state import RuntimeState
from camera.manager import CameraManager
from camera.rtsp_source import mask_rtsp_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deerguard.backend")

app = FastAPI(title="DeerGuard AI — Deer Detection & Collision Warning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global shared state ────────────────────────────────────────────────────────
state = RuntimeState()
processor = PipelineProcessor()
camera_manager = CameraManager()
_start_time = time.time()

logger.info("DeerGuard AI backend initialising…")
logger.info(f"PyTorch: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Source control ─────────────────────────────────────────────────────────────
class SourceConfig(BaseModel):
    type: str   # "integrated" | "webcam" | "rtsp" | "video"
    source: str

class RoadZoneConfig(BaseModel):
    points: List[Tuple[float, float]]
    frame_width: int
    frame_height: int

_current_source: Optional[SourceConfig] = None
_source_lock = threading.Lock()
_source_changed = threading.Event()
_video_loop_running = True
_client_streaming = False
_client_last_frame_time = 0.0
_client_lock = threading.Lock()

def get_current_source() -> Optional[SourceConfig]:
    with _source_lock:
        return _current_source

def set_current_source(config: SourceConfig) -> None:
    global _current_source
    with _source_lock:
        _current_source = config
    state.set_source(config.type, config.source)
    _source_changed.set()

# Initialise from .env if a source is configured
_init_type = get("CAMERA_TYPE", "").lower()
_init_src = get("CAMERA_SOURCE", "")
if _init_src:
    _src_type = "video"
    if "rtsp" in _init_src.lower() or "http" in _init_src.lower():
        _src_type = "rtsp"
    elif "integrated" in _init_type:
        _src_type = "integrated"
    elif "webcam" in _init_type or "usb" in _init_type:
        _src_type = "webcam"
    set_current_source(SourceConfig(type=_src_type, source=_init_src))
    logger.info(f"Auto-loaded source from .env: type={_src_type} source={mask_rtsp_url(_init_src)}")


# ── Background video-processing loop ──────────────────────────────────────────

def _video_loop() -> None:
    """Single authoritative processing loop — reads frames via CameraManager, runs pipeline, updates state."""
    global _client_streaming, _client_last_frame_time
    reconnect_delay: int = get("CAMERA_RECONNECT_DELAY", 3)

    while _video_loop_running:
        with _client_lock:
            if _client_streaming:
                if time.time() - _client_last_frame_time < 5.0:
                    time.sleep(0.05)
                    continue
                else:
                    _client_streaming = False

        source_config = get_current_source()
        if source_config is None:
            camera_manager.stop()
            state.set_camera_status("DISCONNECTED")
            _source_changed.wait(timeout=1.0)
            _source_changed.clear()
            continue

        _source_changed.clear()

        safe_src = mask_rtsp_url(str(source_config.source))
        logger.info(f"[loop] Connecting via CameraManager: type={source_config.type} src={safe_src}")
        state.set_camera_status("CONNECTING")

        # Switching sources safely stops previous source before starting new one (inside CameraManager)
        connected = camera_manager.set_source(source_config.type, source_config.source)
        cur_status = camera_manager.get_status()
        state.set_camera_status(cur_status.value)

        if not connected:
            logger.warning(f"[loop] CameraManager failed to connect: {safe_src}")
            state.set_camera_status("ERROR")
            state.add_event(f"Connection failed ({source_config.type}): {safe_src}", "error")
            _source_changed.wait(timeout=reconnect_delay)
            continue

        processor.reset_source_state()
        logger.info(f"[loop] CameraManager connected: {safe_src}")
        state.set_camera_status("CONNECTED")
        state.add_event(f"Camera connected ({source_config.type}): {safe_src}", "info")

        frame_id = 0
        first_frame = True
        loop_fps_times: list = []

        while not _source_changed.is_set():
            ret, frame = camera_manager.read_frame()
            if not ret or frame is None:
                cur_status = camera_manager.get_status()
                state.set_camera_status(cur_status.value)
                break

            if first_frame:
                first_frame = False
                h, w = frame.shape[:2]
                state.set_frame_dimensions(w, h)
                state.set_camera_status("STREAMING")
                state.add_event(f"Video frames arriving ({w}x{h}) — pipeline active", "info")
                logger.info(f"[loop] First frame received: {w}x{h}")

            try:
                result = processor.process_frame(frame, frame_id)
                tracks = result["tracks"]
                detection_count = result["detection_count"]
                ann_frm = result["annotated_frame"]

                # Encode the *annotated* frame for MJPEG streaming
                ok, jpeg_buf = cv2.imencode(".jpg", ann_frm, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = jpeg_buf.tobytes() if ok else None

                state.update(jpeg_bytes, tracks, frame_id, detection_count)

                # Compute rolling FPS from actual runtime
                t1 = time.time()
                loop_fps_times.append(t1)
                loop_fps_times = [t for t in loop_fps_times if t1 - t < 5.0]
                elapsed = loop_fps_times[-1] - loop_fps_times[0] if len(loop_fps_times) > 1 else 0.0
                fps = (len(loop_fps_times) - 1) / elapsed if elapsed > 0 else 0.0
                state.set_fps(fps)

            except Exception as exc:
                logger.error(f"[loop] Pipeline error frame {frame_id}: {exc}")
                state.add_event(f"Pipeline error: {exc}", "error")

            frame_id += 1

        camera_manager.stop()
        logger.info("[loop] CameraManager stopped")

        if not _source_changed.is_set():
            state.set_camera_status(camera_manager.get_status().value)
            state.add_event(f"Camera disconnected: {safe_src}", "info")
            _source_changed.wait(timeout=reconnect_delay)


_thread = threading.Thread(target=_video_loop, daemon=True, name="VideoLoopThread")
_thread.start()


# ── MJPEG stream generator ─────────────────────────────────────────────────────

def _mjpeg_generator() -> Generator[bytes, None, None]:
    boundary = b"--frame"
    try:
        while True:
            jpeg = state.get_frame()
            if jpeg is None:
                time.sleep(0.033)
                continue
            header = b"\r\nContent-Type: image/jpeg\r\n\r\n"
            yield boundary + header + jpeg + b"\r\n"
            time.sleep(0.033)   # ~30 fps cap
    except (GeneratorExit, StopIteration):
        pass


# ── Core Handlers (Mounted on both / and /api) ──────────────────────────────────

router = APIRouter()

@router.get("/info")
def api_info() -> JSONResponse:
    return JSONResponse(content={"name": "DeerGuard AI", "version": "1.0.0", "docs": "/docs"})

@router.get("/health")
def api_health() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "uptime": round(time.time() - _start_time, 1)})

@router.get("/video/stream")
def api_video_stream() -> StreamingResponse:
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/tracks")
def api_get_tracks() -> JSONResponse:
    tracks = state.get_tracks()
    return JSONResponse(content={
        "frame_id": state.get_frame_id(),
        "tracks": tracks,
    })

@router.get("/runtime")
def api_get_runtime() -> JSONResponse:
    tracks = state.get_tracks()
    camera_status = state.get_camera_status()
    fps = state.get_fps()
    frame_id = state.get_frame_id()
    uptime = round(time.time() - _start_time, 1)
    src = state.get_source()
    road_pts = parse_road_zone()
    vlm_status = processor.vlm.status if processor else "STANDBY"
    captions = [
        {"track_id": t["track_id"], "caption": t["caption"]}
        for t in tracks if t.get("caption")
    ]

    gpu_info = "CPU"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)

    high_risk_count = sum(1 for t in tracks if t.get("risk") == "HIGH_RISK")
    w, h = state.get_frame_dimensions()

    proc_status = "IDLE"
    if camera_status == "STREAMING":
        proc_status = "ACTIVE"
    elif camera_status == "CONNECTING":
        proc_status = "CONNECTING"
    elif camera_status == "ERROR":
        proc_status = "ERROR"

    return JSONResponse(content={
        "system_online": True,
        "backend_connected": True,
        "uptime": uptime,
        "fps": round(fps, 1),
        "frame_id": frame_id,
        "frame_width": w,
        "frame_height": h,
        "processing_status": proc_status,
        "detection_count": state.get_detection_count(),
        "deer_detected": state.get_detection_count() > 0,
        "gpu": gpu_info,
        "active_tracks": len(tracks),
        "high_risk_count": high_risk_count,
        "road_zone_configured": len(road_pts) >= 3,
        "road_zone_points": road_pts,
        "vlm_status": vlm_status,
        "vlm_captions": captions,
        "source": {
            "type": src["type"],
            "source": mask_rtsp_url(src["source"] or "") or None,
            "status": camera_status,
        },
        "components": {
            "camera": camera_status,
            "detection": "ACTIVE" if frame_id > 0 else "IDLE",
            "tracking": "ACTIVE" if frame_id > 0 else "IDLE",
            "prediction": "ACTIVE" if frame_id > 0 else "IDLE",
            "collision_engine": "ACTIVE" if len(road_pts) >= 3 else "IDLE",
            "vlm": vlm_status,
        },
        "vlm_triggered": len(captions) > 0,
    })

@router.get("/status")
def api_get_status() -> JSONResponse:
    return api_get_runtime()

@router.get("/source")
def api_get_source() -> JSONResponse:
    src = state.get_source()
    camera_status = state.get_camera_status()
    return JSONResponse(content={
        "type": src["type"],
        "source": mask_rtsp_url(src["source"] or "") or None,
        "status": camera_status,
    })

@router.get("/source/available")
def api_available_sources() -> JSONResponse:
    webcams = CameraManager.detect_available_cameras()
    videos = CameraManager.list_available_videos()
    return JSONResponse(content={
        "webcams": webcams,
        "videos": videos,
        "supported_types": [
            {"id": "integrated", "name": "Integrated Camera", "default_val": "0"},
            {"id": "webcam", "name": "USB / Webcam", "default_val": "1"},
            {"id": "rtsp", "name": "IP Camera via RTSP", "default_val": ""},
            {"id": "video", "name": "Local Video", "default_val": "13297686_3840_2160_60fps.mp4"},
        ]
    })

@router.post("/source")
def api_configure_source(config: SourceConfig) -> JSONResponse:
    if config.source is None:
        raise HTTPException(status_code=400, detail="source must not be None")
    # Clean default values
    st = config.type.lower().strip()
    if st == "integrated" and not config.source:
        config.source = "0"
    elif st == "webcam" and not config.source:
        config.source = "1"
    set_current_source(config)
    safe_src = mask_rtsp_url(config.source)
    logger.info(f"Source updated: type={config.type} source={safe_src}")
    return JSONResponse(content={"status": "configuring", "type": config.type, "source": safe_src})

@router.post("/source/start")
def api_start_source(config: SourceConfig) -> JSONResponse:
    return api_configure_source(config)

@router.post("/source/stop")
def api_stop_source() -> JSONResponse:
    global _current_source
    with _source_lock:
        _current_source = None
    camera_manager.stop()
    state.set_source(None, None)
    _source_changed.set()
    state.add_event("Camera source stopped by user", "info")
    return JSONResponse(content={"status": "stopped"})

@router.get("/source/status")
def api_source_status() -> JSONResponse:
    return api_get_source()

@router.get("/road-zone")
def api_get_road_zone() -> JSONResponse:
    pts = parse_road_zone()
    w, h = state.get_frame_dimensions()
    return JSONResponse(content={
        "points": pts,
        "frame_width": w,
        "frame_height": h,
        "configured": len(pts) >= 3,
    })

@router.post("/road-zone")
def api_set_road_zone(config: RoadZoneConfig) -> JSONResponse:
    if len(config.points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points required")
    CONFIG["ROAD_ZONE"] = json.dumps(list(config.points))
    global processor
    processor = PipelineProcessor()
    logger.info(f"Road zone set: {len(config.points)} points")
    state.add_event(f"Road zone configured ({len(config.points)} points)", "success")
    return JSONResponse(content={"status": "configured", "points": list(config.points)})

@router.delete("/road-zone")
def api_delete_road_zone() -> JSONResponse:
    CONFIG["ROAD_ZONE"] = ""
    global processor
    processor = PipelineProcessor()
    logger.info("Road zone cleared")
    state.add_event("Road zone cleared", "info")
    return JSONResponse(content={"status": "cleared"})

@router.post("/upload-video")
async def api_upload_video(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a local video file from user device and activate it in the pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: .mp4, .avi, .mov, .mkv, .webm")

    clean_name = f"{int(time.time())}_{Path(file.filename).name.replace(' ', '_')}"
    root = Path(__file__).resolve().parent.parent.parent
    upload_dir = root / "videos" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / clean_name

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    rel_path = f"videos/uploads/{clean_name}"
    set_current_source(SourceConfig(type="video", source=rel_path))
    state.add_event(f"Uploaded video activated: {file.filename}", "success")
    logger.info(f"Video uploaded and activated: {save_path}")

    return JSONResponse(content={
        "status": "uploaded",
        "name": file.filename,
        "path": rel_path,
        "size_bytes": len(content),
    })

@router.post("/client-frame")
async def api_process_client_frame(file: UploadFile = File(...)) -> JSONResponse:
    """Process a single frame captured from the user's browser webcam (with granted permission)."""
    global _client_streaming, _client_last_frame_time
    with _client_lock:
        _client_streaming = True
        _client_last_frame_time = time.time()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")

    state.set_source("browser_webcam", "Browser Camera (Client Device)")
    state.set_camera_status("STREAMING")

    h, w = frame.shape[:2]
    state.set_frame_dimensions(w, h)

    frame_id = state.get_frame_id() + 1

    try:
        result = processor.process_frame(frame, frame_id)
        tracks = result["tracks"]
        detection_count = result["detection_count"]
        ann_frm = result["annotated_frame"]

        ok, jpeg_buf = cv2.imencode(".jpg", ann_frm, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg_buf.tobytes() if ok else None

        state.update(jpeg_bytes, tracks, frame_id, detection_count)

        return JSONResponse(content={
            "status": "ok",
            "frame_id": frame_id,
            "tracks": tracks,
            "detection_count": detection_count,
            "deer_detected": detection_count > 0,
            "frame_width": w,
            "frame_height": h,
        })
    except Exception as exc:
        logger.error(f"Client frame processing error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/client-frame/stop")
def api_stop_client_frame() -> JSONResponse:
    """Stop browser camera client streaming session."""
    global _client_streaming
    with _client_lock:
        _client_streaming = False
    state.set_camera_status("DISCONNECTED")
    state.set_source(None, None)
    state.add_event("Browser camera session ended", "info")
    return JSONResponse(content={"status": "stopped"})

@router.post("/reset")
def api_reset_system() -> JSONResponse:
    global processor, _start_time
    cur = get_current_source()
    state.reset()
    processor = PipelineProcessor()
    _start_time = time.time()
    if cur is not None:
        state.set_source(cur.type, cur.source)
        _source_changed.set()
    logger.info("System reset")
    state.add_event("System reset by user", "info")
    return JSONResponse(content={"status": "reset"})


# Mount router at /api prefix and at root (for endpoints like /health, /tracks, /runtime)
app.include_router(router, prefix="/api")
app.include_router(router)

# ── Production: serve frontend dist ───────────────────────────────────────────
from fastapi.responses import FileResponse

_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _dist.is_dir():
    _assets = _dist / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = _dist / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        return JSONResponse(content={"name": "DeerGuard AI", "version": "1.0.0", "docs": "/docs"})
else:
    @app.get("/", include_in_schema=False)
    def serve_default_root():
        return JSONResponse(content={"name": "DeerGuard AI", "version": "1.0.0", "docs": "/docs"})


if __name__ == "__main__":
    host = get("HOST", "0.0.0.0")
    port = get("PORT", 8000)
    uvicorn.run(app, host=host, port=int(port))
