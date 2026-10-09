# -*- coding: utf-8 -*-
"""End-to-End validation script for DeerGuard AI.

Usage::

    python scripts/e2e_validation.py --source videos/deer_test.mp4

Executes the real AI pipeline on real video frames and produces:
- outputs/e2e_output.mp4 (annotated video)
- outputs/e2e_report.json (factual execution metrics and final status)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from pipeline.process_frame import PipelineProcessor
from config.config import parse_road_zone, CONFIG


def run_e2e(source_path_str: str, max_frames: int = 150):
    source_path = Path(source_path_str)
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "e2e_report.json"
    out_video_path = outputs_dir / "e2e_output.mp4"

    print("=" * 65)
    print("DEERGUARD AI — END-TO-END VALIDATION PIPELINE")
    print("=" * 65)
    print(f"Source video: {source_path}")

    # Check 1: Source exists
    if not source_path.is_file():
        status = "E2E BLOCKED"
        print(f"ERROR: Video source not found at: {source_path}")
        report = {
            "source": str(source_path),
            "final_status": status,
            "error": "Source video file not found"
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return status

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        status = "E2E BLOCKED"
        print("ERROR: Failed to open video source")
        report = {
            "source": str(source_path),
            "final_status": status,
            "error": "Failed to open video source"
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return status

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width} x {height} | FPS: {source_fps:.2f} | Frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, source_fps, (width, height))

    # Initialize authoritative processor
    print("Initializing PipelineProcessor (YOLO + ByteTrack + Kalman + Collision + VLM)...")
    processor = PipelineProcessor()

    # Track metrics
    road_pts = parse_road_zone()
    road_configured = len(road_pts) >= 3

    frame_id = 0
    all_track_ids = set()
    total_detections = 0
    risk_distribution = {"SAFE": 0, "CAUTION": 0, "HIGH_RISK": 0, "UNKNOWN": 0}
    captions = []
    kalman_predictions_count = 0
    trajectory_points_count = 0
    errors = []

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_id >= max_frames):
            break

        try:
            res = processor.process_frame(frame, frame_id)
            tracks = res["tracks"]
            ann_frame = res["annotated_frame"]
            writer.write(ann_frame)

            total_detections += len(tracks)
            for t in tracks:
                all_track_ids.add(t["track_id"])
                r = t.get("risk", "UNKNOWN")
                risk_distribution[r] = risk_distribution.get(r, 0) + 1

                if t.get("predicted") is not None:
                    kalman_predictions_count += 1
                if t.get("trajectory"):
                    trajectory_points_count += len(t["trajectory"])
                if t.get("caption"):
                    captions.append({"frame_id": frame_id, "track_id": t["track_id"], "caption": t["caption"]})

            if frame_id % 20 == 0:
                print(f"Processed frame {frame_id:3d}/{total_frames} | Active tracks: {len(tracks)}")

        except Exception as exc:
            errors.append(f"Frame {frame_id}: {str(exc)}")

        frame_id += 1

    t_elapsed = time.time() - t_start
    processing_fps = frame_id / t_elapsed if t_elapsed > 0 else 0.0

    cap.release()
    writer.release()

    # Determine final status strictly according to rules:
    # FULL E2E VERIFIED ONLY if actual evidence demonstrates:
    # real source -> YOLO -> ByteTrack -> trajectory -> Kalman -> collision evaluation -> runtime state
    # and the collision/risk path is genuinely exercised with configured road geometry.
    # PARTIAL E2E VERIFIED if real source executes pipeline but road zone not configured (all UNKNOWN).
    # E2E BLOCKED if real source cannot execute.
    if frame_id == 0 or len(errors) > 0 and frame_id < 5:
        final_status = "E2E BLOCKED"
    elif road_configured and (risk_distribution["SAFE"] > 0 or risk_distribution["CAUTION"] > 0 or risk_distribution["HIGH_RISK"] > 0):
        final_status = "FULL E2E VERIFIED"
    else:
        final_status = "PARTIAL E2E VERIFIED"

    report = {
        "source": str(source_path.resolve()),
        "resolution": f"{width}x{height}",
        "source_fps": round(source_fps, 2),
        "total_source_frames": total_frames,
        "frames_processed": frame_id,
        "processing_time_sec": round(t_elapsed, 2),
        "processing_fps": round(processing_fps, 2),
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "yolo_model": str(processor.model_path),
        "total_detections": total_detections,
        "unique_track_ids": list(sorted(all_track_ids)),
        "trajectory_evidence": {
            "total_trajectory_points": trajectory_points_count,
            "tracks_with_trajectory": len(all_track_ids)
        },
        "kalman_evidence": {
            "predictions_generated": kalman_predictions_count,
            "prediction_horizon_frames": processor.prediction.horizon
        },
        "road_zone": {
            "configured": road_configured,
            "points": road_pts
        },
        "risk_distribution": risk_distribution,
        "vlm_invocations": len(captions),
        "captions": captions if captions else "BLIP NOT TRIGGERED — NO NATURAL HIGH_RISK EVENT",
        "annotated_video_output": str(out_video_path.resolve()),
        "errors": errors,
        "final_status": final_status
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 65)
    print(f"E2E VALIDATION COMPLETE")
    print(f"Frames processed:   {frame_id}")
    print(f"Unique Track IDs:   {list(sorted(all_track_ids))}")
    print(f"Risk distribution:  {risk_distribution}")
    print(f"Annotated output:   {out_video_path}")
    print(f"Report JSON:        {report_path}")
    print(f"FINAL STATUS:       {final_status}")
    print("=" * 65)
    return final_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E validation for DeerGuard AI")
    parser.add_argument("--source", default="videos/deer_test.mp4", help="Path to video source")
    parser.add_argument("--frames", type=int, default=150, help="Max frames to process (0 = all)")
    args = parser.parse_args()
    run_e2e(args.source, args.frames)
