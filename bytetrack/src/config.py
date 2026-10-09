from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTrackConfig:
    track_activation_threshold: float = 0.10
    lost_track_buffer: int = 90
    minimum_iou_threshold: float = 0.20
    frame_rate: int = 30

    def __post_init__(self) -> None:
        if not 0.0 <= self.track_activation_threshold <= 1.0:
            raise ValueError(
                "track_activation_threshold must be between 0.0 and 1.0."
            )

        if self.lost_track_buffer < 0:
            raise ValueError(
                "lost_track_buffer must be >= 0."
            )

        if not 0.0 <= self.minimum_iou_threshold <= 1.0:
            raise ValueError(
                "minimum_iou_threshold must be between 0.0 and 1.0."
            )

        if self.frame_rate <= 0:
            raise ValueError(
                "frame_rate must be greater than 0."
            )