from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from tkinter import Tk, filedialog
from typing import cast

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.bytetrack import ByteTrackTracker
from src.config import ByteTrackConfig
from src.track_manager import PersistentMOT


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "best.pt"
)

DEER_CLASS_ID = 0

YOLO_CONFIDENCE = 0.05
YOLO_IMAGE_SIZE = 1280

SOURCE_FPS_FALLBACK = 30.0

TRAJECTORY_SECONDS = 10.0

MAX_TRACK_MISSED_FRAMES = 150

DISPLAY_WIDTH = 512
DISPLAY_HEIGHT = 720


# ============================================================
# COLORS
# ============================================================

TRACK_COLORS = [
    (0, 255, 0),       # green
    (0, 165, 255),     # orange
    (255, 0, 0),       # blue
    (255, 0, 255),     # pink
    (0, 255, 255),     # yellow
    (255, 255, 0),     # cyan
    (147, 20, 255),
    (255, 144, 30),
]


def get_track_color(track_id: int):
    return TRACK_COLORS[
        track_id % len(TRACK_COLORS)
    ]


# ============================================================
# FILE PICKER
# ============================================================

def select_video() -> str:

    root = Tk()
    root.withdraw()

    root.attributes(
        "-topmost",
        True,
    )

    video_path = filedialog.askopenfilename(
        title="Select Deer Video",
        filetypes=[
            (
                "Video files",
                "*.mp4 *.avi *.mov *.mkv *.webm",
            ),
            (
                "All files",
                "*.*",
            ),
        ],
    )

    root.destroy()

    return video_path


# ============================================================
# TRAJECTORY MANAGER
# ============================================================

class TrajectoryManager:

    def __init__(
        self,
        max_seconds: float,
    ) -> None:

        self.max_seconds = max_seconds

        self.history = defaultdict(
            deque
        )

    def update(
        self,
        track_id: int,
        x: float,
        y: float,
        timestamp: float,
    ) -> None:

        points = self.history[
            track_id
        ]

        points.append(
            (
                timestamp,
                x,
                y,
            )
        )

        cutoff = (
            timestamp
            - self.max_seconds
        )

        while (
            points
            and points[0][0] < cutoff
        ):
            points.popleft()

    def get_all(
        self,
        current_time: float,
    ):

        cutoff = (
            current_time
            - self.max_seconds
        )

        result = {}

        for track_id, points in list(
            self.history.items()
        ):

            while (
                points
                and points[0][0] < cutoff
            ):
                points.popleft()

            if points:
                result[
                    track_id
                ] = list(points)

            else:
                del self.history[
                    track_id
                ]

        return result

    def reset(self) -> None:
        self.history.clear()


# ============================================================
# DRAW TRAJECTORY
# ============================================================

def draw_trajectory_plane(
    trajectories,
    source_width: int,
    source_height: int,
):

    plane = np.zeros(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        plane,
        "DEER TRAJECTORY",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw each trajectory
    # --------------------------------------------------------

    for track_id, points in trajectories.items():

        if not points:
            continue

        color = get_track_color(
            track_id
        )

        scaled_points = []

        for (
            timestamp,
            x,
            y,
        ) in points:

            px = int(
                x
                / max(
                    1,
                    source_width,
                )
                * DISPLAY_WIDTH
            )

            py = int(
                y
                / max(
                    1,
                    source_height,
                )
                * DISPLAY_HEIGHT
            )

            # Keep inside canvas
            px = max(
                0,
                min(
                    DISPLAY_WIDTH - 1,
                    px,
                ),
            )

            py = max(
                0,
                min(
                    DISPLAY_HEIGHT - 1,
                    py,
                ),
            )

            scaled_points.append(
                (px, py)
            )

        # ----------------------------------------------------
        # Draw path
        # ----------------------------------------------------

        for i in range(
            1,
            len(scaled_points),
        ):

            cv2.line(
                plane,
                scaled_points[i - 1],
                scaled_points[i],
                color,
                3,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Current point
        # ----------------------------------------------------

        last_x, last_y = (
            scaled_points[-1]
        )

        cv2.circle(
            plane,
            (
                last_x,
                last_y,
            ),
            5,
            color,
            -1,
        )

        # ----------------------------------------------------
        # Direction arrow
        # ----------------------------------------------------

        if len(scaled_points) >= 2:

            previous_x, previous_y = (
                scaled_points[-2]
            )

            dx = (
                last_x
                - previous_x
            )

            dy = (
                last_y
                - previous_y
            )

            magnitude = (
                dx * dx
                + dy * dy
            ) ** 0.5

            if magnitude > 2:

                arrow_length = 25

                end_x = int(
                    last_x
                    + dx
                    / magnitude
                    * arrow_length
                )

                end_y = int(
                    last_y
                    + dy
                    / magnitude
                    * arrow_length
                )

                cv2.arrowedLine(
                    plane,
                    (
                        last_x,
                        last_y,
                    ),
                    (
                        end_x,
                        end_y,
                    ),
                    color,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )

        # ----------------------------------------------------
        # ID label
        # ----------------------------------------------------

        cv2.putText(
            plane,
            f"ID {track_id}",
            (
                last_x + 8,
                last_y - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return plane


# ============================================================
# RESIZE VIDEO FOR DISPLAY
# ============================================================

def resize_video_for_display(
    frame: np.ndarray,
):

    h, w = frame.shape[:2]

    scale = min(
        DISPLAY_WIDTH / w,
        DISPLAY_HEIGHT / h,
    )

    new_w = max(
        1,
        int(w * scale),
    )

    new_h = max(
        1,
        int(h * scale),
    )

    resized = cv2.resize(
        frame,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        DISPLAY_WIDTH
        - new_w
    ) // 2

    y_offset = (
        DISPLAY_HEIGHT
        - new_h
    ) // 2

    canvas[
        y_offset:
        y_offset + new_h,
        x_offset:
        x_offset + new_w,
    ] = resized

    return canvas


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )
    print(
        " YOLO11-S + ByteTrack + Persistent MOT"
    )
    print(
        "======================================"
    )
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(
        f"Loading model: {MODEL_PATH}"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "Model classes:",
        model.names,
    )

    # --------------------------------------------------------
    # Select video
    # --------------------------------------------------------

    print()
    print(
        "Opening video selector..."
    )

    video_path = select_video()

    if not video_path:

        print(
            "No video selected."
        )

        return

    print(
        f"Selected video: {video_path}"
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    source_fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        source_fps is None
        or source_fps <= 0
    ):
        source_fps = (
            SOURCE_FPS_FALLBACK
        )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Resolution: "
        f"{source_width}x{source_height}"
    )

    print(
        f"Source FPS: {source_fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------

    tracker_config = (
        ByteTrackConfig(
            track_activation_threshold=0.10,
            lost_track_buffer=90,
            minimum_iou_threshold=0.20,
            frame_rate=max(
                1,
                int(round(source_fps)),
            ),
        )
    )

    byte_tracker = (
        ByteTrackTracker(
            tracker_config
        )
    )

    # --------------------------------------------------------
    # Persistent MOT
    # --------------------------------------------------------

    persistent_mot = (
        PersistentMOT(
            max_missed_frames=MAX_TRACK_MISSED_FRAMES,
            appearance_threshold=0.68,
        )
    )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    trajectory_manager = (
        TrajectoryManager(
            max_seconds=TRAJECTORY_SECONDS
        )
    )

    frame_id = 0

    processing_times = deque(
        maxlen=30
    )

    print()
    print(
        "Press Q to quit."
    )
    print()

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        loop_start = time.perf_counter()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            device="cpu",
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            classes=[DEER_CLASS_ID],
            verbose=False,
        )

        # Explicitly tell Pylance that this is a list of
        # Ultralytics Results objects.
        typed_results = cast(
            list[Results],
            results,
        )

        result = typed_results[0]

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:

            # Model is running on CPU, so converting directly
            # to NumPy avoids the .cpu() typing issue.
            boxes = np.asarray(
                result.boxes.xyxy
            )

            confidences = np.asarray(
                result.boxes.conf
            )

            class_ids = np.asarray(
                result.boxes.cls
            ).astype(int)

            from src.types import Detection

            for bbox, confidence, class_id in zip(
                boxes,
                confidences,
                class_ids,
            ):

                if class_id != DEER_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(
                    float,
                    bbox,
                )

                detections.append(
                    Detection(
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        confidence=float(
                            confidence
                        ),
                        class_id=int(
                            class_id
                        ),
                    )
                )

        # ----------------------------------------------------
        # BYTE TRACK
        # ----------------------------------------------------

        tracked_detections = (
            byte_tracker.update(
                detections,
                frame_id,
            )
        )

        # ----------------------------------------------------
        # PERSISTENT MOT
        # ----------------------------------------------------

        persistent_tracks = (
            persistent_mot.update(
                detections=tracked_detections,
                frame=frame,
                frame_id=frame_id,
            )
        )

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        trajectory_timestamp = (
            frame_id
            / source_fps
        )

        for track in persistent_tracks:

            x1, y1, x2, y2 = (
                track.bbox
            )

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            trajectory_manager.update(
                track_id=(
                    track.persistent_id
                ),
                x=center_x,
                y=center_y,
                timestamp=(
                    trajectory_timestamp
                ),
            )

        trajectories = (
            trajectory_manager.get_all(
                trajectory_timestamp
            )
        )

        # ----------------------------------------------------
        # DRAW BOUNDING BOXES
        # ----------------------------------------------------

        for track in persistent_tracks:

            x1, y1, x2, y2 = (
                map(
                    int,
                    track.bbox,
                )
            )

            color = get_track_color(
                track.persistent_id
            )

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            # Center point
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            cv2.circle(
                frame,
                (
                    center_x,
                    center_y,
                ),
                5,
                color,
                -1,
            )

            # ID label
            label = (
                f"ID "
                f"{track.persistent_id}"
            )

            # Background for label
            (
                text_width,
                text_height,
            ), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )
            )

            label_y = max(
                25,
                y1 - 8,
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y
                    - text_height
                    - baseline,
                ),
                (
                    x1
                    + text_width
                    + 8,
                    label_y + 3,
                ),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 4,
                    label_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        processing_time = (
            time.perf_counter()
            - loop_start
        )

        processing_times.append(
            processing_time
        )

        processing_fps = (
            1.0
            / max(
                1e-6,
                np.mean(
                    processing_times
                ),
            )
        )

        # ----------------------------------------------------
        # INFO ON VIDEO
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Processing FPS: "
            f"{processing_fps:.1f}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Detections: "
            f"{len(detections)}",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Tracks: "
            f"{len(persistent_tracks)}",
            (25, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # TRAJECTORY PANEL
        # ----------------------------------------------------

        trajectory_panel = (
            draw_trajectory_plane(
                trajectories=trajectories,
                source_width=source_width,
                source_height=source_height,
            )
        )

        # ----------------------------------------------------
        # VIDEO PANEL
        # ----------------------------------------------------

        video_panel = (
            resize_video_for_display(
                frame
            )
        )

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        dashboard = np.hstack(
            [
                video_panel,
                trajectory_panel,
            ]
        )

        cv2.imshow(
            "YOLO11-S + ByteTrack + Persistent MOT",
            dashboard,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q"),
        ):
            break

        frame_id += 1

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print(
        "Tracking finished."
    )


if __name__ == "__main__":
    main()