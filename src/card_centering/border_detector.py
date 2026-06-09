"""Inner content border detection on perspective-corrected card images.

Strategy: two-stage "matting" approach.
  Stage 1 (detector.py): extract card from background → rectangular card image.
  Stage 2 (this module): extract border color → the inner boundary of the
      border-colored region IS the content boundary.

The key insight: TCG card borders are typically a uniform single color
(white, silver, yellow, etc.). By segmenting "border color" from the
card image, the inner edge of the resulting mask directly gives us the
content boundary.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Base sampling strip width near the card edge for border-colour
# characterisation.  We start small (5 px) so we don't spill into the
# content area on ultra-thin bordered cards.  If the mask from the
# small sample is too small, we expand adaptively.
SAMPLE_WIDTH_BASE = 5
SAMPLE_WIDTH_MAX = 20

# Morphology kernel size for mask cleanup
MORPH_KERNEL = 5

# Maximum fraction of card dimension to search for borders
MAX_BORDER_FRACTION = 0.35

# Minimum content area fraction
MIN_CONTENT_AREA = 0.30
MAX_CONTENT_AREA = 0.98

# HSV tolerances (for fine-tuning the colour distance)
HUE_TOLERANCE = 25.0       # degrees (circular)
SAT_TOLERANCE = 60.0       # 0-255 scale
VAL_TOLERANCE = 60.0       # 0-255 scale


@dataclass
class ContentRect:
    """Inner content bounding box within a perspective-corrected card image."""

    x: int               # Left edge of content (== left border width)
    y: int               # Top edge of content (== top border width)
    w: int               # Content area width
    h: int               # Content area height
    left_border: int     # Left border width in pixels
    right_border: int    # Right border width in pixels
    top_border: int      # Top border width in pixels
    bottom_border: int   # Bottom border width in pixels
    confidence: float = 1.0

    @property
    def is_valid(self) -> bool:
        return (self.x > 0 and self.y > 0 and
                self.w > 0 and self.h > 0 and
                self.left_border >= 0 and self.right_border >= 0 and
                self.top_border >= 0 and self.bottom_border >= 0)


# ── Public API ──────────────────────────────────────────────────────────────

def detect_content_rect(warped_card: np.ndarray) -> ContentRect | None:
    """Detect the printed content boundary on a rectangular card image.

    Pass 1: HSV colour segmentation of the border region (most reliable).
    Pass 2: Canny edge consensus (sharp transitions).
    Pass 3: HoughLines (last resort).

    Args:
        warped_card: BGR image of the perspective-corrected card.

    Returns:
        ContentRect or None.
    """
    if warped_card is None or warped_card.size == 0:
        return None

    h, w = warped_card.shape[:2]

    # -- Pass 1: border-colour segmentation --
    result = _detect_by_color_segmentation(warped_card, w, h)
    if result is not None and result.is_valid:
        logger.info("Content border: HSV colour segmentation")
        return result

    # -- Pass 2: Canny edge consensus --
    result = _detect_by_edge_consensus(warped_card, w, h)
    if result is not None and result.is_valid:
        logger.info("Content border: edge consensus (fallback)")
        return result

    # -- Pass 3: HoughLines --
    result = _detect_by_hough(warped_card, w, h)
    if result is not None and result.is_valid:
        logger.info("Content border: HoughLines (last resort)")
        return result

    logger.warning("Could not detect content border")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1: border-colour segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_by_color_segmentation(
    card: np.ndarray, w: int, h: int
) -> ContentRect | None:
    """Segment the border colour from the card image.

    The border region mask's inner boundary → content boundary.

    Steps:
      1. Sample the border colour from the outermost SAMPLE_WIDTH pixels
         of *each edge independently* (border colour can differ between
         edges on some cards).
      2. Build a binary mask: 255 where the pixel is "close enough" to the
         edge-specific border colour, 0 elsewhere.
      3. Clean the mask with morphology.
      4. For each edge, scan the mask from the card edge inward — the first
         255→0 transition is the content boundary.
      5. Take the median transition position across all scanlines as the
         final boundary.
    """
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Sample border colour for each edge independently
    samples = {}
    for edge in ("left", "right", "top", "bottom"):
        h_med, s_med, v_med, h_std, s_std, v_std = _sample_edge_hsv(hsv, w, h, edge)
        if h_med is None:
            return None
        samples[edge] = (h_med, s_med, v_med, h_std, s_std, v_std)

    # --- Build per-edge masks and combine ---
    # For each edge we only care about the transition zone, not the whole card.
    # This avoids false-positives when border colour happens to match parts of
    # the artwork.
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for edge in ("left", "right", "top", "bottom"):
        h_med, s_med, v_med, h_std, s_std, v_std = samples[edge]

        # Adaptive tolerances: tight for uniform borders, looser for noisy ones
        h_tol = max(HUE_TOLERANCE, h_std * 3.0)
        s_tol = max(SAT_TOLERANCE / 2, s_std * 3.0)
        v_tol = max(VAL_TOLERANCE / 2, v_std * 3.0)

        # Build mask for how far we care from this edge
        max_dist = int(min(w, h) * MAX_BORDER_FRACTION)
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        if edge == "left":
            zone_mask[:, :max_dist] = 255
        elif edge == "right":
            zone_mask[:, w - max_dist:] = 255
        elif edge == "top":
            zone_mask[:max_dist, :] = 255
        else:  # bottom
            zone_mask[h - max_dist:, :] = 255

        # Per-pixel colour distance for this edge
        edge_mask = _color_distance_mask(
            hsv, h_med, s_med, v_med, h_tol, s_tol, v_tol)

        # Only keep matches within this edge's zone
        edge_mask = cv2.bitwise_and(edge_mask, zone_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edge_mask)

    # --- Morphological cleanup ---
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, k)  # fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k)   # remove noise

    # --- Scan mask to find content boundary for each edge ---
    left = _scan_mask_boundary(combined_mask, w, h, "left")
    right = _scan_mask_boundary(combined_mask, w, h, "right")
    top = _scan_mask_boundary(combined_mask, w, h, "top")
    bottom = _scan_mask_boundary(combined_mask, w, h, "bottom")

    if None in (left, right, top, bottom):
        return None

    content_x = left
    content_y = top
    content_w = w - left - right
    content_h = h - top - bottom

    if content_w <= 0 or content_h <= 0:
        return None

    area_ratio = (content_w * content_h) / (w * h)
    if area_ratio < MIN_CONTENT_AREA or area_ratio > MAX_CONTENT_AREA:
        return None

    # Confidence based on how clean the mask is
    border_fill = np.count_nonzero(combined_mask) / (w * h)
    confidence = 0.95 if 0.05 < border_fill < 0.40 else 0.80

    return ContentRect(
        x=content_x, y=content_y, w=content_w, h=content_h,
        left_border=left, right_border=right,
        top_border=top, bottom_border=bottom,
        confidence=confidence,
    )


# ── Colour helpers ──────────────────────────────────────────────────────────

def _sample_edge_hsv(
    hsv: np.ndarray, w: int, h: int, edge: str
) -> tuple[float, ...] | None:
    """Sample the border colour from the outermost SAMPLE_WIDTH pixels of an edge.

    Returns (h_median, s_median, v_median, h_std, s_std, v_std) or None.
    """
    sw = min(SAMPLE_WIDTH_BASE, int(min(w, h) * 0.05))
    if sw < 2:
        return None

    if edge == "left":
        if sw >= w:
            return None
        strip = hsv[:, :sw, :]
    elif edge == "right":
        if sw >= w:
            return None
        strip = hsv[:, w - sw:, :]
    elif edge == "top":
        if sw >= h:
            return None
        strip = hsv[:sw, :, :]
    else:  # bottom
        if sw >= h:
            return None
        strip = hsv[h - sw:, :, :]

    strip = strip.reshape(-1, 3)

    # Median is more robust to outliers (e.g. bright specks on the edge)
    h_med = float(np.median(strip[:, 0]))
    s_med = float(np.median(strip[:, 1]))
    v_med = float(np.median(strip[:, 2]))

    # Use MAD (median absolute deviation) for robust std estimate
    h_mad = float(np.median(np.abs(strip[:, 0] - h_med)))
    s_mad = float(np.median(np.abs(strip[:, 1] - s_med)))
    v_mad = float(np.median(np.abs(strip[:, 2] - v_med)))

    # Convert MAD to a std-like measure (MAD * 1.4826 ≈ std for normal dist.)
    h_std = h_mad * 1.4826 + 1.0
    s_std = s_mad * 1.4826 + 1.0
    v_std = v_mad * 1.4826 + 1.0

    return h_med, s_med, v_med, h_std, s_std, v_std


def _color_distance_mask(
    hsv: np.ndarray,
    h_ref: float, s_ref: float, v_ref: float,
    h_tol: float, s_tol: float, v_tol: float,
) -> np.ndarray:
    """Build a binary mask of pixels "close enough" to the reference HSV colour.

    Hue is treated as circular.  S and V are linear.
    """
    # Hue: circular distance
    h_diff = np.abs(hsv[:, :, 0] - h_ref)
    h_diff = np.minimum(h_diff, 180.0 - h_diff)

    # Saturation and Value: absolute difference
    s_diff = np.abs(hsv[:, :, 1] - s_ref)
    v_diff = np.abs(hsv[:, :, 2] - v_ref)

    # Each channel must be within its tolerance
    mask_h = (h_diff <= h_tol)
    mask_s = (s_diff <= s_tol)
    mask_v = (v_diff <= v_tol)

    # A pixel matches if it is within tolerance in ALL three channels
    mask = mask_h & mask_s & mask_v

    return (mask.astype(np.uint8)) * 255


# ── Mask boundary scanner ───────────────────────────────────────────────────

def _scan_mask_boundary(
    mask: np.ndarray, w: int, h: int, edge: str,
    n_samples: int = 80,
) -> int | None:
    """Scan the mask from the card edge inward to find where border colour ends.

    For each scanline, finds the first 255→0 transition (border → content).
    Returns the median transition distance across all scanlines.

    Returns the border width in pixels (distance from card edge to content).
    """
    max_dist = int(min(w, h) * MAX_BORDER_FRACTION)

    if edge == "left":
        n_lines = h
        search_end = min(max_dist, w)
    elif edge == "right":
        n_lines = h
        search_end = min(max_dist, w)
    elif edge == "top":
        n_lines = w
        search_end = min(max_dist, h)
    else:  # bottom
        n_lines = w
        search_end = min(max_dist, h)

    step = max(1, n_lines // n_samples)
    positions = []

    for line_idx in range(0, n_lines, step):
        if edge == "left":
            line = mask[line_idx, :search_end]
        elif edge == "right":
            line = mask[line_idx, w - search_end:][::-1]
        elif edge == "top":
            line = mask[:search_end, line_idx]
        else:  # bottom
            line = mask[h - search_end:, line_idx][::-1]

        # Find first 255→0 transition: scan until we see 255 (border) then
        # wait for it to drop to 0 (content).  The drop position is the boundary.
        saw_border = False
        for i, val in enumerate(line):
            if val == 255:
                saw_border = True
            elif saw_border and val == 0:
                # Confirm this isn't just a single-pixel gap
                # Check that the next few pixels are also 0
                gap_confirmed = True
                for j in range(1, min(5, len(line) - i)):
                    if line[i + j] == 255:
                        gap_confirmed = False
                        break
                if gap_confirmed:
                    positions.append(i)
                    break

    if len(positions) < n_samples * 0.25:
        return None

    return int(round(np.median(positions)))


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2: Canny edge consensus
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_by_edge_consensus(
    card: np.ndarray, w: int, h: int
) -> ContentRect | None:
    """Multi-threshold Canny edges → per-scanline median consensus."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    e1 = cv2.Canny(gray, 20, 60)
    e2 = cv2.Canny(gray, 35, 100)
    e3 = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_or(e1, e2)
    edges = cv2.bitwise_or(edges, e3)

    n_s = 60
    left = _edge_scan(edges, gray, w, h, "left", n_s)
    right = _edge_scan(edges, gray, w, h, "right", n_s)
    top = _edge_scan(edges, gray, w, h, "top", n_s)
    bottom = _edge_scan(edges, gray, w, h, "bottom", n_s)

    if None in (left, right, top, bottom):
        return None
    if left >= right or top >= bottom:
        return None

    cw, ch = right - left, bottom - top
    if cw <= 0 or ch <= 0:
        return None
    ar = (cw * ch) / (w * h)
    if ar < MIN_CONTENT_AREA or ar > MAX_CONTENT_AREA:
        return None

    return ContentRect(
        x=left, y=top, w=cw, h=ch,
        left_border=left, right_border=w - right,
        top_border=top, bottom_border=h - bottom,
        confidence=0.80,
    )


def _edge_scan(
    edges: np.ndarray, gray: np.ndarray,
    w: int, h: int, edge: str, n_samples: int,
) -> int | None:
    """Scan N evenly-spaced lines; find outermost edge; return median."""
    if edge == "left":
        s0, s1 = 5, int(w * MAX_BORDER_FRACTION)
        n_lines = h
    elif edge == "right":
        s0, s1 = int(w * (1 - MAX_BORDER_FRACTION)), w - 5
        n_lines = h
    elif edge == "top":
        s0, s1 = 5, int(h * MAX_BORDER_FRACTION)
        n_lines = w
    else:
        s0, s1 = int(h * (1 - MAX_BORDER_FRACTION)), h - 5
        n_lines = w

    step = max(1, n_lines // n_samples)
    positions = []

    for li in range(0, n_lines, step):
        pos = None
        if edge == "left":
            row = edges[li, s0:s1]
            ep = np.where(row > 0)[0]
            if len(ep) > 0:
                pos = s0 + ep[0]
                if not _confirm_change(gray, li, pos, edge, w, h):
                    pos = None
        elif edge == "right":
            row = edges[li, s0:s1]
            ep = np.where(row > 0)[0]
            if len(ep) > 0:
                pos = s0 + ep[-1]
                if not _confirm_change(gray, li, pos, edge, w, h):
                    pos = None
        elif edge == "top":
            col = edges[s0:s1, li]
            ep = np.where(col > 0)[0]
            if len(ep) > 0:
                pos = s0 + ep[0]
                if not _confirm_change(gray, pos, li, edge, w, h):
                    pos = None
        else:
            col = edges[s0:s1, li]
            ep = np.where(col > 0)[0]
            if len(ep) > 0:
                pos = s0 + ep[-1]
                if not _confirm_change(gray, pos, li, edge, w, h):
                    pos = None

        if pos is not None:
            positions.append(pos)

    if len(positions) < n_samples * 0.25:
        return None
    return int(round(np.median(positions)))


def _confirm_change(
    gray: np.ndarray, a: int, b: int, edge: str, w: int, h: int,
) -> bool:
    """Verify a real intensity change across boundary (a,b)."""
    d = 5
    if edge == "left":
        before = gray[a, max(0, b - d):b]
        after = gray[a, b + 1:min(w, b + d + 1)]
    elif edge == "right":
        before = gray[a, b + 1:min(w, b + d + 1)]
        after = gray[a, max(0, b - d):b]
    elif edge == "top":
        before = gray[max(0, a - d):a, b]
        after = gray[a + 1:min(h, a + d + 1), b]
    else:
        before = gray[a + 1:min(h, a + d + 1), b]
        after = gray[max(0, a - d):a, b]
    if before.size < 2 or after.size < 2:
        return True
    return abs(float(np.mean(before)) - float(np.mean(after))) > 8


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3: HoughLines (last resort)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_by_hough(
    card: np.ndarray, w: int, h: int
) -> ContentRect | None:
    """Canny + HoughLinesP fallback for very low-contrast borders."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 40, 120)

    ms, me = int(w * 0.02), int(w * MAX_BORDER_FRACTION)
    vs, ve = int(h * 0.02), int(h * MAX_BORDER_FRACTION)

    mask = np.zeros_like(edges)
    mask[:, ms:me] = 255
    mask[:, w - me:w - ms] = 255
    mask[vs:ve, :] = 255
    mask[h - ve:h - vs, :] = 255
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=50,
                            minLineLength=w // 4, maxLineGap=20)
    if lines is None:
        return None

    h_lines, v_lines = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:
            h_lines.append(ln[0])
        elif 80 < angle < 100:
            v_lines.append(ln[0])

    if not v_lines or not h_lines:
        return None

    lb = max(v_lines, key=lambda l: (l[0] + l[2]) // 2) if v_lines else None
    rb = min(v_lines, key=lambda l: (l[0] + l[2]) // 2) if v_lines else None
    tb = max(h_lines, key=lambda l: (l[1] + l[3]) // 2) if h_lines else None
    bb = min(h_lines, key=lambda l: (l[1] + l[3]) // 2) if h_lines else None

    if lb is None or rb is None or tb is None or bb is None:
        return None

    lx = (lb[0] + lb[2]) // 2
    rx = (rb[0] + rb[2]) // 2
    ty = (tb[1] + tb[3]) // 2
    by = (bb[1] + bb[3]) // 2

    if lx >= rx or ty >= by:
        return None

    return ContentRect(
        x=lx, y=ty, w=rx - lx, h=by - ty,
        left_border=lx, right_border=w - rx,
        top_border=ty, bottom_border=h - by,
        confidence=0.55,
    )
