from abc import ABC, abstractmethod
from typing import Sequence

from .types import Detection, TrackedDetection


class BaseTracker(ABC):
    """
    Abstract interface for stateful multi-object trackers.

    Implementations receive detections from successive frames and
    maintain tracking state across calls.
    """

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        frame_id: int,
    ) -> list[TrackedDetection]:
        """
        Update tracker state using detections from one processed frame.

        Parameters
        ----------
        detections:
            Detections produced for the current frame.

            This may be empty. An empty detection set is still a valid
            frame and must be passed to the tracker so that its internal
            temporal state can be updated correctly.

        frame_id:
            Monotonically increasing identifier of the processed frame.

            The identifier represents the source/application frame number,
            not the current processing FPS.

        Returns
        -------
        list[TrackedDetection]:
            Current detections associated with persistent track IDs.

            An empty list is valid when no tracks are currently output.

        Raises
        ------
        ValueError:
            If frame_id violates the implementation's frame-order
            requirements.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal tracking state.

        After reset(), the tracker behaves as a newly initialized tracker.
        """
        raise NotImplementedError