"""Camera capture utilities using OpenCV VideoCapture.

Supports listing cameras, capturing frames, and providing a live preview
with an alignment guide overlay.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from card_centering.platform_utils import get_camera_backend

logger = logging.getLogger(__name__)

# Module-level constant computed once per process
_CAMERA_BACKEND = get_camera_backend()


def list_cameras(max_test: int = 5) -> list[dict]:
    """List available camera devices.

    Args:
        max_test: Maximum camera index to test.

    Returns:
        List of dicts with 'index', 'name', 'resolution'.
    """
    cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, _CAMERA_BACKEND)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "index": i,
                "name": f"Camera {i}",
                "resolution": f"{w}×{h}",
            })
            cap.release()
    return cameras


def capture_frame(camera_index: int = 0) -> np.ndarray | None:
    """Capture a single frame from a camera.

    Args:
        camera_index: Camera device index.

    Returns:
        BGR image as numpy array, or None on failure.
    """
    cap = cv2.VideoCapture(camera_index, _CAMERA_BACKEND)
    if not cap.isOpened():
        logger.error("Failed to open camera %d", camera_index)
        return None

    # Warm up: discard first few frames (auto-exposure settling)
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Failed to capture frame from camera %d", camera_index)
        return None

    return frame


class CameraCapture:
    """Manages a camera for live preview.

    Usage:
        cam = CameraCapture(0)
        cam.start()
        while True:
            frame = cam.read()
            if frame is None:
                break
            # show frame...
        cam.stop()
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = 0
        self._height = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def start(self) -> bool:
        """Open the camera and start streaming."""
        self._cap = cv2.VideoCapture(self.camera_index, _CAMERA_BACKEND)
        if not self._cap.isOpened():
            logger.error("Failed to open camera %d", self.camera_index)
            self._cap = None
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Warm up
        for _ in range(5):
            self._cap.read()

        return True

    def read(self) -> np.ndarray | None:
        """Read the latest frame. Returns None on failure."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    def stop(self):
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


def draw_alignment_guide(
    frame: np.ndarray,
    card_aspect: float = 63.0 / 88.0,
    margin_pct: float = 0.1,
) -> np.ndarray:
    """Draw an alignment guide overlay on a camera preview frame.

    Shows a rectangle indicating where to place the card, helping users
    position the card parallel to the camera.

    Args:
        frame: BGR camera frame.
        card_aspect: Target aspect ratio (width/height).
        margin_pct: Margin around the guide as fraction of frame.

    Returns:
        Annotated frame copy.
    """
    result = frame.copy()
    h, w = frame.shape[:2]

    # Calculate guide rectangle
    guide_w = int(w * (1 - 2 * margin_pct))
    guide_h = int(guide_w / card_aspect)

    # Ensure guide fits vertically
    if guide_h > h * (1 - 2 * margin_pct):
        guide_h = int(h * (1 - 2 * margin_pct))
        guide_w = int(guide_h * card_aspect)

    x1 = (w - guide_w) // 2
    y1 = (h - guide_h) // 2
    x2 = x1 + guide_w
    y2 = y1 + guide_h

    # Semi-transparent overlay
    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Corner brackets
    bracket_len = 30
    # Top-left
    cv2.line(overlay, (x1, y1), (x1 + bracket_len, y1), (0, 255, 0), 3)
    cv2.line(overlay, (x1, y1), (x1, y1 + bracket_len), (0, 255, 0), 3)
    # Top-right
    cv2.line(overlay, (x2, y1), (x2 - bracket_len, y1), (0, 255, 0), 3)
    cv2.line(overlay, (x2, y1), (x2, y1 + bracket_len), (0, 255, 0), 3)
    # Bottom-left
    cv2.line(overlay, (x1, y2), (x1 + bracket_len, y2), (0, 255, 0), 3)
    cv2.line(overlay, (x1, y2), (x1, y2 - bracket_len), (0, 255, 0), 3)
    # Bottom-right
    cv2.line(overlay, (x2, y2), (x2 - bracket_len, y2), (0, 255, 0), 3)
    cv2.line(overlay, (x2, y2), (x2, y2 - bracket_len), (0, 255, 0), 3)

    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    # Hint text
    cv2.putText(result, "Align card within the brackets",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(result, "Hold camera parallel to card",
                (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return result
