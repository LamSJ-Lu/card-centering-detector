"""Card outline detection and perspective correction using OpenCV.

Finds the 4 corners of a TCG/sports card in a photo and performs
perspective warp to produce a rectangular front-facing card image.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# TCG standard card aspect ratio: 63mm × 88mm ≈ 0.7159
TCG_ASPECT_RATIO = 63.0 / 88.0       # Portrait: width/height ≈ 0.716
TCG_ASPECT_RATIO_LANDSCAPE = 88.0 / 63.0  # Landscape: width/height ≈ 1.397

# Tolerance range for card aspect ratio detection
ASPECT_MIN = 0.55
ASPECT_MAX = 1.60


@dataclass
class CardOutline:
    """Result of card detection and perspective correction."""

    corners: np.ndarray          # 4 corner points (4, 2) in original photo coords
    size: tuple[int, int]        # Warped card dimensions (w, h)
    warped_image: np.ndarray     # Perspective-corrected rectangular card image
    confidence: float = 1.0      # Detection confidence 0.0 ~ 1.0

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]


def _load_image_safe(path: str) -> np.ndarray | None:
    """Load image with Unicode path support (cv2.imread fails on CJK paths)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image: %s", path)
        return img
    except Exception as e:
        logger.error("Error loading image %s: %s", path, e)
        return None


def _compute_card_aspect(corners: np.ndarray) -> float:
    """Compute the width/height aspect ratio from 4 detected corner points.

    Uses average edge lengths to estimate the actual card orientation,
    so landscape and portrait cards each keep their native shape.

    Args:
        corners: (4, 2) float32 array in [tl, tr, br, bl] order.

    Returns:
        width/height ratio.  < 1.0 → portrait;  > 1.0 → landscape.
    """
    # Top and bottom edge lengths (horizontal edges)
    top_edge = float(np.linalg.norm(corners[1] - corners[0]))
    bottom_edge = float(np.linalg.norm(corners[2] - corners[3]))
    # Left and right edge lengths (vertical edges)
    left_edge = float(np.linalg.norm(corners[3] - corners[0]))
    right_edge = float(np.linalg.norm(corners[2] - corners[1]))

    avg_width = (top_edge + bottom_edge) / 2.0
    avg_height = (left_edge + right_edge) / 2.0

    if avg_height <= 0:
        return TCG_ASPECT_RATIO

    computed = avg_width / avg_height

    # Snap to the closest TCG-standard ratio if within 15%, otherwise keep
    # the computed ratio (e.g. for non-standard card sizes).
    if computed >= 1.0:
        # Landscape region
        if 0.85 * TCG_ASPECT_RATIO_LANDSCAPE <= computed <= 1.15 * TCG_ASPECT_RATIO_LANDSCAPE:
            return TCG_ASPECT_RATIO_LANDSCAPE
        return computed
    else:
        # Portrait region
        if 0.85 * TCG_ASPECT_RATIO <= computed <= 1.15 * TCG_ASPECT_RATIO:
            return TCG_ASPECT_RATIO
        return computed


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = corners.sum(axis=1)
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]

    # Difference: smallest diff = top-right, largest diff = bottom-left
    diff = np.diff(corners, axis=1)
    tr = corners[np.argmin(diff)]
    bl = corners[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _score_contour(contour: np.ndarray, edge_map: np.ndarray,
                   img_area: int) -> float:
    """Score a contour based on area, aspect ratio, and edge alignment.

    Returns 0.0 ~ 1.0, higher is better.
    """
    area = cv2.contourArea(contour)
    area_ratio = area / img_area

    # Area should be 10% ~ 85% of image
    if area_ratio < 0.05 or area_ratio > 0.90:
        return 0.0

    # Aspect ratio check
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return 0.0
    aspect = w / h if w < h else h / w
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        return 0.0

    # Area score: prefer contours that fill 20-60% of image
    if 0.15 <= area_ratio <= 0.65:
        area_score = 1.0
    elif area_ratio < 0.15:
        area_score = area_ratio / 0.15
    else:
        area_score = max(0.0, 1.0 - (area_ratio - 0.65) / 0.25)

    # Edge alignment score: how many contour points lie near edges
    mask = np.zeros(edge_map.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, 2)
    edge_overlap = cv2.countNonZero(cv2.bitwise_and(mask, edge_map))
    contour_len = max(cv2.arcLength(contour, True), 1)
    edge_score = min(1.0, edge_overlap / contour_len * 2.0)

    return area_score * 0.4 + edge_score * 0.6


def detect_card_corners(image: np.ndarray) -> np.ndarray | None:
    """Detect the 4 corner points of a card in a photo.

    Uses a two-pass strategy:
    1. Otsu thresholding (fast, works on high-contrast backgrounds)
    2. Multi-threshold Canny (fallback for textured backgrounds)

    Args:
        image: BGR image as numpy array.

    Returns:
        (4, 2) float32 array of corner points [tl, tr, br, bl], or None.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Pass 1: Otsu-based ----
    corners = _detect_pass_otsu(gray, img_area)
    if corners is not None:
        logger.info("Card corners detected via Otsu method")
        return corners

    # ---- Pass 2: Multi-threshold Canny fallback ----
    corners = _detect_pass_canny(gray, img_area)
    if corners is not None:
        logger.info("Card corners detected via Canny fallback")
        return corners

    logger.warning("No card corners found in image")
    return None


def _detect_pass_otsu(gray: np.ndarray, img_area: int) -> np.ndarray | None:
    """Pass 1: Otsu thresholding to find card contour."""
    # CLAHE for lighting invariance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Try both polarities (dark on light / light on dark)
    best_corners = None
    best_score = 0.0

    edge_map = cv2.Canny(blurred, 50, 150)

    for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, binary = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            score = _score_contour(cnt, edge_map, img_area)
            if score > best_score:
                corners = _extract_corners(cnt)
                if corners is not None:
                    best_score = score
                    best_corners = corners

    if best_score >= 0.3 and best_corners is not None:
        return best_corners

    return None


def _detect_pass_canny(gray: np.ndarray, img_area: int) -> np.ndarray | None:
    """Pass 2: Multi-threshold Canny for textured backgrounds."""
    # Three Canny thresholds combined
    canny1 = cv2.Canny(gray, 20, 80)
    canny2 = cv2.Canny(gray, 40, 160)
    canny3 = cv2.Canny(gray, 60, 200)
    combined = cv2.bitwise_or(canny1, canny2)
    combined = cv2.bitwise_or(combined, canny3)

    # Dilate to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(combined, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_corners = None
    best_score = 0.0

    for cnt in contours:
        score = _score_contour(cnt, combined, img_area)
        if score > best_score:
            corners = _extract_corners(cnt)
            if corners is not None:
                best_score = score
                best_corners = corners

    if best_score >= 0.2 and best_corners is not None:
        return best_corners

    return None


def _extract_corners(contour: np.ndarray) -> np.ndarray | None:
    """Extract 4 ordered corner points from a contour."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        return _order_corners(corners)

    # If not exactly 4 points, use minAreaRect
    if len(contour) >= 4:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)
        return _order_corners(corners)

    return None


def perspective_correct(
    image: np.ndarray,
    corners: np.ndarray,
    target_aspect: float = TCG_ASPECT_RATIO,
    target_width: int = 1200,
) -> CardOutline:
    """Perspective-warp the card from trapezoid to rectangle.

    Args:
        image: Original BGR photo.
        corners: 4 corner points (4, 2) in [tl, tr, br, bl] order.
        target_aspect: Target width/height ratio. Default TCG = 63/88 ≈ 0.716.
        target_width: Target pixel width of the output rectangle.

    Returns:
        CardOutline with the warped rectangular card image.
    """
    target_height = int(target_width / target_aspect)

    dst_corners = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(image, M, (target_width, target_height),
                                 flags=cv2.INTER_LANCZOS4)

    return CardOutline(
        corners=corners.copy(),
        size=(target_width, target_height),
        warped_image=warped,
        confidence=1.0,
    )


def refine_with_grabcut(
    image: np.ndarray,
    corners: np.ndarray,
    target_width: int = 1200,
    iterations: int = 3,
) -> CardOutline | None:
    """Use GrabCut to refine the card boundary, then re-correct.

    This is an optional second pass: after initial corner detection, GrabCut
    can produce a more precise card mask, especially when the background and
    card edges have similar colours that confuse contour-based detection.

    Args:
        image: Original BGR photo.
        corners: Initial 4-corner detection (from detect_card_corners).
        target_width: Output card width.
        iterations: GrabCut iterations (more = slower but finer).

    Returns:
        CardOutline with refined corners, or None on failure.
    """
    if corners is None or len(corners) != 4:
        return None

    h, w = image.shape[:2]

    # Build a bounding rect from corners, expanded slightly for safety margin
    x_coords = corners[:, 0].astype(int)
    y_coords = corners[:, 1].astype(int)
    x1 = max(0, int(np.min(x_coords)) - 10)
    y1 = max(0, int(np.min(y_coords)) - 10)
    x2 = min(w, int(np.max(x_coords)) + 10)
    y2 = min(h, int(np.max(y_coords)) + 10)

    # GrabCut inits
    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # Mark the rect interior as probable foreground
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        logger.warning("GrabCut failed — falling back to initial corners")
        return None

    # Extract foreground (GC_FGD + GC_PR_FGD)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find largest contour in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    refined_corners = _extract_corners(largest)
    if refined_corners is None:
        return None

    # Re-run perspective correction with refined corners
    card_aspect = _compute_card_aspect(refined_corners)
    return perspective_correct(image, refined_corners, target_aspect=card_aspect,
                               target_width=target_width)


def detect_and_correct(
    image: np.ndarray,
    target_width: int = 1200,
) -> CardOutline | None:
    """Full pipeline: detect card corners and perform perspective correction.

    Args:
        image: BGR photo containing a card.
        target_width: Output width of the rectangular card image.

    Returns:
        CardOutline or None if detection fails.
    """
    corners = detect_card_corners(image)
    if corners is None:
        return None

    # Derive aspect ratio from corners to preserve landscape/portrait orientation
    card_aspect = _compute_card_aspect(corners)
    return perspective_correct(image, corners, target_aspect=card_aspect,
                               target_width=target_width)
