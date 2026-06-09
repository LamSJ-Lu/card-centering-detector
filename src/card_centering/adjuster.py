"""Manual adjustment tools for card outline corners and content borders.

Provides interactive adjustment with a magnifier overlay for pixel-precise
positioning of card corners and content boundary lines.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Handle size in pixels for corner/edge drag handles
CORNER_HANDLE_RADIUS = 12
EDGE_HANDLE_RADIUS = 8
HIT_THRESHOLD = 14  # Distance in px for hit testing
INNER_LINE_HIT_THRESHOLD = 12  # Distance in px for inner edge-line hit testing


def _point_to_segment_dist(
    px: float, py: float,
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Minimum distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0,
        ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


@dataclass
class CornerAdjustment:
    """State of outer card corner adjustment."""

    corners: np.ndarray              # Current 4 corner points (4, 2)
    is_editing: bool = False
    selected_index: int | None = None
    original_corners: np.ndarray | None = None  # Auto-detected originals

    def __post_init__(self):
        if self.original_corners is None and self.corners is not None:
            self.original_corners = self.corners.copy()

    def reset(self):
        """Restore to auto-detected corners."""
        if self.original_corners is not None:
            self.corners = self.original_corners.copy()
        self.selected_index = None
        self.is_editing = False

    def start_edit(self):
        """Enter editing mode."""
        self.is_editing = True

    def confirm(self):
        """Exit editing mode, keeping current positions."""
        self.is_editing = False
        self.selected_index = None
        self.original_corners = self.corners.copy()

    def cancel(self):
        """Exit editing mode, restoring original positions."""
        self.reset()

    def select_handle(self, pos: tuple[float, float]) -> int | None:
        """Check if a position hits a corner handle. Returns index or None."""
        for i, corner in enumerate(self.corners):
            dist = np.sqrt((corner[0] - pos[0]) ** 2 + (corner[1] - pos[1]) ** 2)
            if dist <= HIT_THRESHOLD:
                return i
        return None

    def move_handle(self, index: int, new_pos: tuple[float, float]):
        """Move a corner handle to a new position."""
        self.corners[index] = [new_pos[0], new_pos[1]]

    def move_all(self, delta: tuple[float, float]):
        """Move all corners by a delta."""
        self.corners[:, 0] += delta[0]
        self.corners[:, 1] += delta[1]

    def get_handle_positions(self) -> list[tuple[float, float]]:
        """Get all handle positions for rendering."""
        return [(c[0], c[1]) for c in self.corners]


@dataclass
class BorderAdjustment:
    """State of inner content border adjustment (rectangle-constrained)."""

    left: int
    right: int
    top: int
    bottom: int
    card_width: int
    card_height: int
    is_editing: bool = False
    selected_edge: str | None = None  # 'left' | 'right' | 'top' | 'bottom'
    original_values: dict | None = None

    def __post_init__(self):
        if self.original_values is None:
            self.original_values = {
                "left": self.left, "right": self.right,
                "top": self.top, "bottom": self.bottom,
            }

    def reset(self):
        """Restore to auto-detected borders."""
        if self.original_values:
            self.left = self.original_values["left"]
            self.right = self.original_values["right"]
            self.top = self.original_values["top"]
            self.bottom = self.original_values["bottom"]
        self.selected_edge = None
        self.is_editing = False

    def start_edit(self):
        self.is_editing = True

    def confirm(self):
        self.is_editing = False
        self.selected_edge = None
        self.original_values = {
            "left": self.left, "right": self.right,
            "top": self.top, "bottom": self.bottom,
        }

    def cancel(self):
        self.reset()

    def select_edge(self, pos: tuple[float, float]) -> str | None:
        """Check if position hits an edge handle. Returns edge name or None."""
        px, py = pos
        threshold = HIT_THRESHOLD

        # Left edge
        if abs(px - self.left) <= threshold and self.top <= py <= self.bottom:
            return "left"
        # Right edge
        if abs(px - self.right) <= threshold and self.top <= py <= self.bottom:
            return "right"
        # Top edge
        if abs(py - self.top) <= threshold and self.left <= px <= self.right:
            return "top"
        # Bottom edge
        if abs(py - self.bottom) <= threshold and self.left <= px <= self.right:
            return "bottom"
        return None

    def move_edge(self, edge: str, new_pos: float):
        """Move an edge, keeping rectangle constraints."""
        margin = 5  # Minimum pixels from card edge and between edges

        if edge == "left":
            self.left = max(margin, min(int(new_pos), self.right - margin))
        elif edge == "right":
            self.right = max(self.left + margin, min(int(new_pos), self.card_width - margin))
        elif edge == "top":
            self.top = max(margin, min(int(new_pos), self.bottom - margin))
        elif edge == "bottom":
            self.bottom = max(self.top + margin, min(int(new_pos), self.card_height - margin))

    def get_content_rect(self) -> "ContentRect":
        """Convert to a ContentRect for analysis."""
        from card_centering.border_detector import ContentRect
        return ContentRect(
            x=self.left,
            y=self.top,
            w=self.right - self.left,
            h=self.bottom - self.top,
            left_border=self.left,
            right_border=self.card_width - self.right,
            top_border=self.top,
            bottom_border=self.card_height - self.bottom,
            confidence=1.0,  # Manual adjustment = maximum confidence
        )


# ---- Magnifier ----
# The actual MagnifierWidget is implemented as a PySide6 QWidget in gui.py.
# This module provides the pure-computation magnifier rendering logic
# that the GUI widget calls into.


@dataclass
class MagnifierState:
    """Pure-data magnifier state, independent of GUI framework."""

    enabled: bool = False
    source_image: np.ndarray | None = None
    center_pos: tuple[int, int] = (0, 0)
    zoom: float = 5.0
    source_radius: int = 25
    display_size: int = 250
    crosshair_color: tuple[int, int, int] = (255, 0, 0)
    grid_enabled: bool = True
    shape: str = "circle"  # "circle" | "rectangle"


def render_magnifier(state: MagnifierState) -> np.ndarray:
    """Render the magnified view as a BGR image.

    The caller converts this to a QPixmap for display.

    Args:
        state: MagnifierState with source image and position.

    Returns:
        BGR image of the magnified region at display_size × display_size.
    """
    if state.source_image is None:
        # Return a blank image
        blank = np.zeros((state.display_size, state.display_size, 3), dtype=np.uint8)
        blank[:] = (50, 50, 50)
        return blank

    img_h, img_w = state.source_image.shape[:2]
    cx, cy = state.center_pos
    r = state.source_radius

    # Clamp source region to image bounds
    x1 = max(0, cx - r)
    x2 = min(img_w, cx + r)
    y1 = max(0, cy - r)
    y2 = min(img_h, cy + r)

    if x2 <= x1 or y2 <= y1:
        blank = np.zeros((state.display_size, state.display_size, 3), dtype=np.uint8)
        blank[:] = (50, 50, 50)
        return blank

    # Crop source region
    crop = state.source_image[y1:y2, x1:x2].copy()

    # Pad if the crop doesn't cover the full radius (near image edges)
    pad_left = cx - r - x1
    pad_top = cy - r - y1
    if pad_left < 0:
        crop = np.pad(crop, ((0, 0), (-pad_left, 0), (0, 0)), constant_values=0)
    if pad_top < 0:
        crop = np.pad(crop, ((-pad_top, 0), (0, 0), (0, 0)), constant_values=0)
    pad_right = (x2 - (cx + r)) if x2 < cx + r else 0
    pad_bottom = (y2 - (cy + r)) if y2 < cy + r else 0
    if pad_right > 0:
        crop = np.pad(crop, ((0, 0), (0, pad_right), (0, 0)), constant_values=0)
    if pad_bottom > 0:
        crop = np.pad(crop, ((0, pad_bottom), (0, 0), (0, 0)), constant_values=0)

    # Ensure crop is exactly 2r × 2r
    target_h, target_w = r * 2, r * 2
    if crop.shape[0] < target_h or crop.shape[1] < target_w:
        crop = cv2_resize_pad(crop, target_w, target_h)

    # Resize with nearest-neighbor to preserve pixel boundaries
    zoomed = cv2_resize(crop, state.display_size, state.display_size,
                        interpolation="nearest")

    # Draw crosshair lines
    mid = state.display_size // 2
    cv2.line(zoomed, (mid, 0), (mid, state.display_size - 1),
             state.crosshair_color, 1)
    cv2.line(zoomed, (0, mid), (state.display_size - 1, mid),
             state.crosshair_color, 1)
    # Center dot
    cv2.circle(zoomed, (mid, mid), 2, state.crosshair_color, -1)

    # Draw pixel grid
    if state.grid_enabled:
        grid_spacing = int(state.zoom)
        if grid_spacing >= 2:
            for i in range(mid % grid_spacing, state.display_size, grid_spacing):
                cv2.line(zoomed, (i, 0), (i, state.display_size - 1),
                         (80, 80, 80), 1, cv2.LINE_AA)
                cv2.line(zoomed, (0, i), (state.display_size - 1, i),
                         (80, 80, 80), 1, cv2.LINE_AA)

    return zoomed


def cv2_resize(img: np.ndarray, width: int, height: int,
               interpolation: str = "nearest") -> np.ndarray:
    """Resize image with specified interpolation."""
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interp = interp_map.get(interpolation, cv2.INTER_NEAREST)
    return cv2.resize(img, (width, height), interpolation=interp)


def cv2_resize_pad(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize or pad image to exactly target_w × target_h."""
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    result = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    result[:copy_h, :copy_w] = img[:copy_h, :copy_w]
    return result


# ── Inner-content corner helpers ──────────────────────────────────────────────

def corners_from_content_rect(rect: "ContentRect") -> np.ndarray:
    """Convert a ContentRect to 4 corner points [TL, TR, BR, BL].

    Used for interactive corner-based inner-border editing.
    """
    return np.array([
        [rect.x, rect.y],                          # TL
        [rect.x + rect.w, rect.y],                 # TR
        [rect.x + rect.w, rect.y + rect.h],        # BR
        [rect.x, rect.y + rect.h],                 # BL
    ], dtype=np.float32)


def content_rect_from_corners(
    corners: np.ndarray,
    card_w: int,
    card_h: int,
) -> "ContentRect":
    """Convert 4 corner points back to an axis-aligned ContentRect.

    Takes the bounding box of the (possibly free-form) corners so the
    result is always a valid rectangle.  Corners are clamped to card
    boundaries.
    """
    from card_centering.border_detector import ContentRect

    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    x = max(0, int(round(np.min(x_coords))))
    y = max(0, int(round(np.min(y_coords))))
    x2 = min(card_w, int(round(np.max(x_coords))))
    y2 = min(card_h, int(round(np.max(y_coords))))
    w = max(1, x2 - x)
    h = max(1, y2 - y)
    return ContentRect(
        x=x, y=y, w=w, h=h,
        left_border=x, right_border=card_w - x2,
        top_border=y, bottom_border=card_h - y2,
        confidence=1.0,
    )
