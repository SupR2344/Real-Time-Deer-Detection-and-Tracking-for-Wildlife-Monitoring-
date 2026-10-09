# -*- coding: utf-8 -*-
"""Unit and integration tests for DeerGuard AI system."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from detection.detector import DeerDetector
from bytetrack_adapter.adapter import ByteTrackAdapter
from trajectory.manager import TrajectoryManager
from prediction.kalman_filter import KalmanFilter
from prediction.manager import PredictionManager
from collision.engine import CollisionEngine, RiskLevel
from collision.geometry import create_polygon, point_distance_to_polygon, trajectory_intersects_polygon
from vlm.manager import VLMManager
from pipeline.process_frame import PipelineProcessor
from camera.base import CameraStatus
from camera.video_source import VideoSource
from camera.manager import CameraManager
from backend.app.state import RuntimeState
from fastapi.testclient import TestClient
from backend.app.main import app


# ── Helpers & Fixtures ────────────────────────────────────────────────────────

def dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture
def mock_detector():
    """Mock DeerDetector to return deterministic detections for pipeline testing."""
    with patch("pipeline.process_frame.DeerDetector") as MockCls:
        instance = MockCls.return_value
        instance.detect.return_value = [
            {"bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.92, "class_id": 0, "class_name": "deer"}
        ]
        yield instance


# ── 1. TrajectoryManager Tests ────────────────────────────────────────────────

def test_trajectory_manager_lifecycle():
    tm = TrajectoryManager()
    # Add point for track 1
    tm.update(track_id=1, frame_idx=0, bbox=[100, 100, 200, 200])
    hist = tm.get_history(1)
    assert len(hist) == 1
    assert hist[0] == (0, 150.0, 150.0)

    # Add second point
    tm.update(track_id=1, frame_idx=1, bbox=[110, 110, 210, 210])
    hist = tm.get_history(1)
    assert len(hist) == 2
    assert hist[1] == (1, 160.0, 160.0)

    # Prune missing: track 1 is kept, track 2 would be removed
    tm.prune_missing(active_ids=[1])
    assert len(tm.get_history(1)) == 2

    # Prune missing: track 1 not active
    tm.prune_missing(active_ids=[])
    assert len(tm.get_history(1)) == 0


# ── 2. Kalman Filter & Prediction Tests ────────────────────────────────────────

def test_kalman_filter_and_prediction():
    kf = KalmanFilter()
    # Initial observation
    kf.update(100.0, 100.0)
    assert kf.current_position == (100.0, 100.0)

    # Moving right: cx=110, cy=100
    kf.update(110.0, 100.0)
    pred_x, pred_y = kf.predict(steps=5)
    # Velocity should be positive along X
    assert pred_x > 110.0

    pm = PredictionManager()
    pm.update(track_id=42, bbox=[100, 100, 200, 200])
    pos = pm.current_position(42)
    assert pos == (150.0, 150.0)
    future = pm.predict(42)
    assert isinstance(future, tuple) and len(future) == 2


# ── 3. Collision Engine & Geometry Tests ───────────────────────────────────────

def test_collision_geometry():
    poly = create_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert poly is not None

    # Point inside polygon
    assert point_distance_to_polygon((50, 50), poly) == 0.0

    # Point outside polygon
    assert point_distance_to_polygon((150, 50), poly) == 50.0

    # Trajectory intersects polygon
    assert trajectory_intersects_polygon((-10, 50), (50, 50), poly) is True
    # Trajectory misses polygon
    assert trajectory_intersects_polygon((200, 200), (300, 300), poly) is False


def test_collision_engine_levels():
    # 1. No polygon configured -> UNKNOWN
    ce_none = CollisionEngine(road_points=[])
    assert ce_none.evaluate((50, 50), speed=5.0) == RiskLevel.UNKNOWN

    # 2. Polygon configured: [100, 100] to [300, 300]
    ce = CollisionEngine(road_points=[[100, 100], [300, 100], [300, 300], [100, 300]])

    # Inside road -> HIGH_RISK
    assert ce.evaluate((200, 200), speed=1.0) == RiskLevel.HIGH_RISK

    # Intersecting road -> HIGH_RISK
    assert ce.evaluate((200, 200), speed=3.0, current_pos=(50, 200)) == RiskLevel.HIGH_RISK

    # Far away -> SAFE
    assert ce.evaluate((1000, 1000), speed=1.0, current_pos=(990, 1000)) == RiskLevel.SAFE


# ── 4. VLM Manager Tests ──────────────────────────────────────────────────────

def test_vlm_manager_lazy_and_cooldown():
    vlm = VLMManager()
    assert vlm.status == "STANDBY"
    assert vlm._loaded is False

    # Guard against empty / small crops
    assert vlm.caption(np.zeros((5, 5, 3), dtype=np.uint8), track_id=1) is None

    # Verify cooldown logic without downloading heavy model weights
    with patch.object(vlm, "_load", return_value=True):
        vlm._processor = MagicMock()
        vlm._model = MagicMock()
        vlm._model.generate.return_value = [[1, 2, 3]]
        vlm._processor.decode.return_value = "a deer on the road"

        # First call succeeds
        crop = dummy_frame(100, 100)
        c1 = vlm.caption(crop, track_id=99)
        assert c1 == "a deer on the road"

        # Immediate second call should be blocked by cooldown
        c2 = vlm.caption(crop, track_id=99)
        assert c2 is None


def test_pipeline_resets_temporal_state_between_sources(mock_detector):
    processor = PipelineProcessor()
    frame = dummy_frame()

    first = processor.process_frame(frame, frame_id=0)
    processor.reset_source_state()
    second = processor.process_frame(frame, frame_id=0)

    assert first["detection_count"] == 1
    assert second["detection_count"] == 1


# ── 5. Camera Source & Manager Tests ──────────────────────────────────────────

def test_camera_manager_lifecycle():
    cm = CameraManager()
    assert cm.get_status() == CameraStatus.DISCONNECTED

    # Test invalid path
    res = cm.set_source("video", "nonexistent_file_12345.mp4")
    assert res is False
    assert cm.get_status() == CameraStatus.ERROR

    info = cm.get_safe_source_info()
    assert info["type"] == "video"
    assert info["status"] == "ERROR"

    cm.stop()
    assert cm.get_status() == CameraStatus.DISCONNECTED


# ── 6. RuntimeState Tests ─────────────────────────────────────────────────────

def test_runtime_state_thread_safety():
    state = RuntimeState()
    assert state.get_camera_status() == "DISCONNECTED"
    assert state.get_frame_id() == -1

    state.set_camera_status("STREAMING")
    state.set_source("video", "sample.mp4")
    state.set_camera_status("STREAMING")
    state.set_fps(24.5)
    state.set_frame_dimensions(1280, 720)
    state.add_event("Camera connected", "info")

    assert state.get_camera_status() == "STREAMING"
    assert state.get_source() == {"type": "video", "source": "sample.mp4"}
    assert state.get_fps() == 24.5
    assert state.get_frame_dimensions() == (1280, 720)
    assert len(state.get_events()) == 1

    state.update(b"fake_jpeg", [{"track_id": 1, "risk": "SAFE"}], frame_id=10, detection_count=3)
    assert state.get_frame() == b"fake_jpeg"
    assert len(state.get_tracks()) == 1
    assert state.get_detection_count() == 3
    assert state.get_frame_id() == 10

    state.reset()
    assert state.get_frame() is None
    assert state.get_frame_id() == -1
    assert state.get_camera_status() == "DISCONNECTED"
    assert state.get_source() == {"type": None, "source": None}
    assert state.get_detection_count() == 0


def test_runtime_state_clears_stale_frame_on_source_loss():
    state = RuntimeState()
    state.update(b"jpeg", [{"track_id": 1}], frame_id=7)
    state.set_fps(12.0)

    state.set_camera_status("ERROR")

    assert state.get_frame() is None
    assert state.get_tracks() == []
    assert state.get_frame_id() == -1
    assert state.get_fps() == 0.0


# ── 7. FastAPI Endpoint Tests ─────────────────────────────────────────────────

def test_fastapi_endpoints():
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Runtime status
    res = client.get("/runtime")
    assert res.status_code == 200
    data = res.json()
    assert "system_online" in data
    assert "components" in data
    assert "active_tracks" in data

    # Tracks
    res = client.get("/tracks")
    assert res.status_code == 200
    assert "tracks" in res.json()

    # Road-zone GET / POST / DELETE
    res = client.get("/road-zone")
    assert res.status_code == 200

    # Configure road zone
    poly_payload = {
        "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "frame_width": 640,
        "frame_height": 480
    }
    res = client.post("/road-zone", json=poly_payload)
    assert res.status_code == 200
    assert res.json()["status"] == "configured"

    # Verify configured
    res = client.get("/road-zone")
    assert res.json()["configured"] is True

    # Delete road zone
    res = client.delete("/road-zone")
    assert res.status_code == 200

    # Source control GET & POST
    res = client.get("/source")
    assert res.status_code == 200

    res = client.post("/source", json={"type": "video", "source": "videos/deer_test.mp4"})
    assert res.status_code == 200

    # Source available
    res = client.get("/source/available")
    assert res.status_code == 200
    avail = res.json()
    assert "webcams" in avail
    assert "videos" in avail
    assert "supported_types" in avail
    assert len(avail["supported_types"]) == 4

    # Stop source
    res = client.post("/source/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"

    # Reset
    res = client.post("/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "reset"


# ── 8. Detailed Camera Source Tests ──────────────────────────────────────────

def test_camera_sources_and_switching():
    from camera.webcam_source import WebcamSource
    from camera.rtsp_source import RTSPSource, mask_rtsp_url

    cm = CameraManager()

    # 1. Local video source
    video_res = cm.set_source("video", "13297686_3840_2160_60fps.mp4")
    assert video_res is True
    assert cm.get_status() in (CameraStatus.CONNECTED, CameraStatus.STREAMING)
    ret, frame = cm.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.shape[0] == 2160 and frame.shape[1] == 3840

    # 2. Switch to webcam (safe stop occurs automatically)
    # Available webcam on machine (index 0)
    webcam_res = cm.set_source("webcam", "0")
    if webcam_res:
        assert cm.get_status() in (CameraStatus.CONNECTED, CameraStatus.STREAMING)
        ret, wframe = cm.read_frame()
        assert ret is True
        assert wframe is not None
    else:
        assert cm.get_status() == CameraStatus.ERROR

    # 3. Invalid webcam index -> must return ERROR, never fake CONNECTED
    bad_webcam = cm.set_source("webcam", "999")
    assert bad_webcam is False
    assert cm.get_status() == CameraStatus.ERROR

    # 4. RTSP URL masking & connection error handling
    test_rtsp_url = "rtsp://admin:secret123@192.168.1.200:554/live"
    masked = mask_rtsp_url(test_rtsp_url)
    assert "secret123" not in masked
    assert "***:***" in masked

    # Unreachable RTSP connection -> must return ERROR, never fake CONNECTED
    rtsp_res = cm.set_source("rtsp", "rtsp://127.0.0.1:9999/unreachable_stream")
    assert rtsp_res is False
    assert cm.get_status() == CameraStatus.ERROR

    # 5. Stop releases all resources
    cm.stop()
    assert cm.get_status() == CameraStatus.DISCONNECTED

    # 6. Available sources detection
    cams = CameraManager.detect_available_cameras()
    assert isinstance(cams, list)
    vids = CameraManager.list_available_videos()
    assert isinstance(vids, list)
    assert any("13297686" in v["name"] for v in vids)

