# -*- coding: utf-8 -*-
"""Smoke test script for verifying all DeerGuard AI components."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


def run_smoke_tests():
    print("=" * 60)
    print("DEERGUARD AI — SMOKE TEST SUITE")
    print("=" * 60)
    all_passed = True

    # 1. Environment & CUDA
    print("[1/9] Checking PyTorch & CUDA...")
    try:
        print(f"      PyTorch: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"      CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"      GPU Device: {torch.cuda.get_device_name(0)}")
        print("      PASS: PyTorch runtime OK")
    except Exception as e:
        print(f"      FAIL: {e}")
        all_passed = False

    # 2. YOLO Model Load
    print("[2/9] Loading YOLO model from model/best.pt...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(PROJECT_ROOT / "model" / "best.pt"))
        print(f"      Model classes: {model.names}")
        print("      PASS: YOLO model loaded successfully")
    except Exception as e:
        print(f"      FAIL: YOLO model loading failed: {e}")
        all_passed = False

    # 3. Detector Smoke Test
    print("[3/9] Testing DeerDetector...")
    try:
        from detection.detector import DeerDetector
        detector = DeerDetector(
            model_path=str(PROJECT_ROOT / "model" / "best.pt"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy)
        print(f"      Inference on blank frame: {len(dets)} detections")
        print("      PASS: DeerDetector inference OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"      FAIL: DeerDetector failed: {e}")
        all_passed = False

    # 4. ByteTrack Adapter
    print("[4/9] Testing ByteTrack adapter...")
    try:
        from bytetrack_adapter.adapter import ByteTrackAdapter
        tracker = ByteTrackAdapter()
        fake_dets = [{"bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.95, "class_id": 0}]
        tracked = tracker.update(fake_dets, frame_id=0)
        print(f"      Tracked detections count: {len(tracked)}")
        print("      PASS: ByteTrack tracker OK")
    except Exception as e:
        print(f"      FAIL: ByteTrack failed: {e}")
        all_passed = False

    # 5. Trajectory Manager
    print("[5/9] Testing TrajectoryManager...")
    try:
        from trajectory.manager import TrajectoryManager
        traj = TrajectoryManager()
        traj.update(track_id=1, frame_idx=0, bbox=[100, 100, 200, 200])
        history = traj.get_history(1)
        assert len(history) == 1, "Trajectory history length mismatch"
        print(f"      Trajectory history entry: {history[0]}")
        print("      PASS: TrajectoryManager OK")
    except Exception as e:
        print(f"      FAIL: TrajectoryManager failed: {e}")
        all_passed = False

    # 6. Kalman Filter & Prediction
    print("[6/9] Testing KalmanFilter & PredictionManager...")
    try:
        from prediction.manager import PredictionManager
        pm = PredictionManager()
        pm.update(1, [100, 100, 200, 200])
        pred = pm.predict(1)
        speed = pm.speed(1)
        print(f"      Predicted future point: {pred}, Speed: {speed}")
        print("      PASS: Kalman prediction OK")
    except Exception as e:
        print(f"      FAIL: PredictionManager failed: {e}")
        all_passed = False

    # 7. Collision Engine
    print("[7/9] Testing CollisionEngine...")
    try:
        from collision.engine import CollisionEngine, RiskLevel
        # Without polygon
        ce_no_poly = CollisionEngine(road_points=[])
        assert ce_no_poly.evaluate((150, 150), 5.0) == RiskLevel.UNKNOWN, "Expected UNKNOWN when no polygon"
        # With polygon
        ce_poly = CollisionEngine(road_points=[[100, 100], [300, 100], [300, 300], [100, 300]])
        # Inside
        assert ce_poly.evaluate((200, 200), 1.0) == RiskLevel.HIGH_RISK, "Expected HIGH_RISK inside polygon"
        # Far outside
        assert ce_poly.evaluate((800, 800), 1.0) == RiskLevel.SAFE, "Expected SAFE far outside"
        print("      PASS: CollisionEngine evaluated UNKNOWN, HIGH_RISK, and SAFE correctly")
    except Exception as e:
        print(f"      FAIL: CollisionEngine failed: {e}")
        all_passed = False

    # 8. VLM Manager (Lazy Load Check)
    print("[8/9] Testing VLMManager lazy-load...")
    try:
        from vlm.manager import VLMManager
        vlm = VLMManager()
        assert vlm._loaded is False, "VLM should be lazy-loaded, not loaded on init"
        assert vlm.status == "STANDBY", "VLM initial status must be STANDBY"
        print("      PASS: VLMManager lazy-load and STANDBY status OK")
    except Exception as e:
        print(f"      FAIL: VLMManager failed: {e}")
        all_passed = False

    # 9. Camera Sources
    print("[9/9] Testing Camera Sources...")
    try:
        from camera.manager import CameraManager
        from camera.video_source import VideoSource
        from camera.webcam_source import WebcamSource
        from camera.rtsp_source import RTSPSource
        cam_mgr = CameraManager()
        print("      PASS: Camera sources and CameraManager instantiated OK")
    except Exception as e:
        print(f"      FAIL: Camera sources failed: {e}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL SMOKE TESTS PASSED [9/9]")
    else:
        print("SOME SMOKE TESTS FAILED")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
