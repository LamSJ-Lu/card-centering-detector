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
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.

    Uses centroid-based polar-angle sorting which is robust for **any**
    in-plane rotation angle (0–360°), unlike the simpler sum/diff heuristic.
    """
    # Centroid
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))

    # Polar angle from centroid (0 = right, increasing counter-clockwise)
    angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)

    # Sort by angle
    order = np.argsort(angles)
    sorted_corners = corners[order]

    # Now sorted_corners[0] has the smallest angle.
    # For a typical card the "top-left" is the corner closest to upper-left
    # of the image.  We identify it as the corner whose sum of coordinates
    # is *smallest* among the four sorted corners, then rotate the array.
    sums = sorted_corners.sum(axis=1)
    tl_idx = int(np.argmin(sums))

    # Rotate so tl_idx becomes index 0
    ordered = np.roll(sorted_corners, -tl_idx, axis=0)

    # Verify clockwise order: tl, tr, br, bl
    # After roll, index 1 should be "right side" — check via x coordinate
    if ordered[1][0] < ordered[0][0] and abs(ordered[1][0] - ordered[0][0]) > abs(ordered[3][0] - ordered[0][0]):
        # index 1 is more left than tl — likely wrong orientation, flip
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return ordered.astype(np.float32)


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

    # Aspect ratio check — use minAreaRect so the true (unrotated) shape
    # is tested instead of the axis-aligned bounding box, which distorts
    # badly when the card is photographed at an angle.
    if len(contour) < 4:
        return 0.0
    rotated_rect = cv2.minAreaRect(contour)
    rw, rh = rotated_rect[1]
    if rw < 1 or rh < 1:
        return 0.0
    aspect = rw / rh if rw < rh else rh / rw
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

    Multi-pass strategy (ordered from fast+reliable to exhaustive):
    1. Otsu thresholding (clean backgrounds)
    2. Multi-threshold Canny (textured backgrounds)
    3. Rotation-aware HoughLines + minAreaRect (tilted / angled cards)
    4. Auto deskew → re-run passes 1-3 (heavily skewed cards, >15°)

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

    # ---- Pass 3: Rotation-aware detection ----
    corners = _detect_pass_rotated(gray, img_area)
    if corners is not None:
        logger.info("Card corners detected via rotation-aware pass")
        return corners

    # ---- Pass 4: Auto-deskew then re-try passes 1-3 ----
    deskewed = _auto_deskew(image, gray)
    if deskewed is not None:
        d_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        d_h, d_w = deskewed.shape[:2]
        d_area = d_h * d_w

        for name, detector in [
            ("Otsu (deskewed)", lambda: _detect_pass_otsu(d_gray, d_area)),
            ("Canny (deskewed)", lambda: _detect_pass_canny(d_gray, d_area)),
            ("rotated (deskewed)", lambda: _detect_pass_rotated(d_gray, d_area)),
        ]:
            d_corners = detector()
            if d_corners is not None:
                # Transform corners back from deskewed image to original
                corners = _transform_corners_back(d_corners, gray, deskewed)
                if corners is not None:
                    logger.info("Card corners detected via %s", name)
                    return corners

    logger.warning("No card corners found in image")
    return None


def _auto_deskew(image: np.ndarray, gray: np.ndarray) -> np.ndarray | None:
    """Rotate the image so the card's edges are aligned with the image axes.

    Detects the dominant edge direction via Canny + HoughLines.  If the
    dominant angle deviates from horizontal by more than 12°, the image is
    rotated to compensate.  Returns the deskewed BGR image, or None when
    the image is already level enough.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(h, w) // 3)
    if lines is None:
        return None

    # Collect line angles, weighted by line "strength" (accumulator value)
    angles = []
    for rho_theta in lines[:80]:  # top 80 strongest lines
        rho, theta = rho_theta[0]
        deg = theta * 180.0 / np.pi
        # Normalise to -45° ~ +45° (we only care about how far from level)
        deg = (deg + 45.0) % 90.0 - 45.0
        angles.append(deg)

    if not angles:
        return None

    # Use median as robust dominant angle estimator
    dominant_deg = float(np.median(angles))

    # Only deskew if the tilt is significant
    if abs(dominant_deg) < 12.0:
        return None

    # Rotate to level
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, dominant_deg, 1.0)
    # Expand canvas to avoid clipping rotated content
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    rot_mat[0, 2] += new_w / 2.0 - center[0]
    rot_mat[1, 2] += new_h / 2.0 - center[1]

    deskewed = cv2.warpAffine(image, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    logger.info("Auto-deskew applied: %.1f° rotation", dominant_deg)
    return deskewed


def _transform_corners_back(
    d_corners: np.ndarray, original_gray: np.ndarray, deskewed: np.ndarray
) -> np.ndarray | None:
    """Map corners detected on a deskewed image back to original coordinates.

    Recalculates the affine transformation that was used for deskew by
    matching ORB features between the two images, then applies the inverse
    matrix to the corner points.
    """
    oh, ow = original_gray.shape[:2]
    dh, dw = deskewed.shape[:2]

    # Use ORB feature matching to recover the forward transform
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(deskewed, None)
    kp2, des2 = orb.detectAndCompute(original_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Deskewed → original homography (RANSAC for robustness)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    # Transform corners
    ones = np.ones((4, 1), dtype=np.float32)
    src = np.hstack([d_corners.reshape(4, 2), ones])
    dst = H @ src.T
    dst = dst[:2, :] / dst[2, :]
    result = dst.T.astype(np.float32)

    # Clip to image bounds
    result[:, 0] = np.clip(result[:, 0], 0, ow - 1)
    result[:, 1] = np.clip(result[:, 1], 0, oh - 1)

    return result


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


def _detect_pass_rotated(gray: np.ndarray, img_area: int) -> np.ndarray | None:
    """Pass 3: rotation-aware detection for tilted / angled cards.

    Strategy A — minAreaRect on every largish contour, pick the one whose
    rotated-aspect best matches a TCG card.
    Strategy B — HoughLinesP to find 4 border lines, intersect them for
    exact corner positions (works even on heavily skewed cards).
    """
    h, w = gray.shape[:2]

    # ---- Strategy A: best-rotated-rectangle search ----
    # Multi-Canny for good edge coverage
    e1 = cv2.Canny(gray, 20, 80)
    e2 = cv2.Canny(gray, 40, 160)
    edges = cv2.bitwise_or(e1, e2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_aspect_err = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.92:
            continue
        if len(cnt) < 4:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue

        # Largest dimension / smallest → "portrait" aspect
        ra = rw / rh if rw < rh else rh / rw
        err = abs(ra - TCG_ASPECT_RATIO)
        if err < abs(best_aspect_err) and ra >= ASPECT_MIN and ra <= ASPECT_MAX:
            best_aspect_err = err
            best_rect = rect

    if best_rect is not None and best_aspect_err < 0.25:
        corners = cv2.boxPoints(best_rect).astype(np.float32)
        return _order_corners(corners)

    # ---- Strategy B: HoughLinesP → 4 lines → 4 corners ----
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=60,
                            minLineLength=min(w, h) // 5, maxLineGap=30)
    if lines is None or len(lines) < 4:
        return None

    # Separate into horizontal-ish and vertical-ish lines
    horiz, vert = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        if angle < 30 or angle > 150:
            horiz.append(ln[0])
        elif 60 < angle < 120:
            vert.append(ln[0])

    if len(horiz) < 2 or len(vert) < 2:
        return None

    # Sort horizontals by y, verticals by x
    horiz_sorted = sorted(horiz, key=lambda l: (l[1] + l[3]) / 2.0)
    vert_sorted = sorted(vert, key=lambda l: (l[0] + l[2]) / 2.0)

    # Take the two outermost lines in each direction
    h_top = horiz_sorted[0]
    h_bot = horiz_sorted[-1]
    v_left = vert_sorted[0]
    v_right = vert_sorted[-1]

    # Intersect the 4 lines to get 4 corners
    corners = np.zeros((4, 2), dtype=np.float32)
    corners[0] = _line_intersection(v_left, h_top)    # tl
    corners[1] = _line_intersection(v_right, h_top)   # tr
    corners[2] = _line_intersection(v_right, h_bot)   # br
    corners[3] = _line_intersection(v_left, h_bot)    # bl

    # Sanity check: all corners inside or near the image boundary
    for pt in corners:
        if pt[0] < -w * 0.15 or pt[0] > w * 1.15 or pt[1] < -h * 0.15 or pt[1] > h * 1.15:
            return None

    return corners.astype(np.float32)


def _line_intersection(ln1: np.ndarray, ln2: np.ndarray) -> np.ndarray:
    """Intersection point of two line segments (each [x1, y1, x2, y2]).

    Uses the parametric / determinant method.
    """
    x1, y1, x2, y2 = ln1
    x3, y3, x4, y4 = ln2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        # Parallel — return midpoint of the first segment
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return np.array([px, py], dtype=np.float32)


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
