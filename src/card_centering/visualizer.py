"""Visual annotation drawing for card centering analysis.

Draws border lines, measurement labels, center indicators, and scoring
overlays on the perspective-corrected card image.
"""

import cv2
import numpy as np


# Color scheme (BGR)
COLORS = {
    "outer_border": (255, 255, 255),    # White
    "inner_border": (0, 255, 255),      # Yellow
    "measurement": (0, 255, 0),         # Green
    "label_bg": (0, 0, 0),             # Black (semi-transparent)
    "label_text": (255, 255, 255),      # White
    "corner_handle": (0, 0, 255),       # Red
    "crosshair": (0, 0, 255),          # Red
    "guide_line": (200, 200, 200),      # Light gray
    "grade_a": (0, 180, 0),            # Green
    "grade_b": (180, 120, 0),          # Blue/teal
    "grade_c": (0, 140, 255),          # Orange
    "grade_d": (0, 0, 255),            # Red
}

GRADE_COLORS = {
    "A": COLORS["grade_a"],
    "B": COLORS["grade_b"],
    "C": COLORS["grade_c"],
    "D": COLORS["grade_d"],
}


def _put_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 1,
    bg_alpha: float = 0.6,
) -> None:
    """Draw text with a semi-transparent background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 3, y - th - 6), (x + tw + 3, y + 4),
                  COLORS["label_bg"], -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    # Text
    cv2.putText(img, text, (x, y), font, font_scale, COLORS["label_text"],
                thickness, cv2.LINE_AA)


def _draw_dimension_arrow(
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    color: tuple[int, int, int] = COLORS["measurement"],
) -> None:
    """Draw a dimension line with arrows and label."""
    # Line
    cv2.line(img, start, end, color, 1, cv2.LINE_AA)
    # Arrowheads
    _draw_arrowhead(img, start, end, color)
    _draw_arrowhead(img, end, start, color)
    # Label at midpoint
    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2
    _put_text_with_bg(img, label, (mid_x - 20, mid_y - 8),
                      font_scale=0.5)


def _draw_arrowhead(
    img: np.ndarray,
    tip: tuple[int, int],
    tail: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 8,
) -> None:
    """Draw a small arrowhead at tip pointing toward tail."""
    dx = tail[0] - tip[0]
    dy = tail[1] - tip[1]
    length = np.sqrt(dx * dx + dy * dy)
    if length == 0:
        return
    dx, dy = dx / length, dy / length

    # Perpendicular
    px, py = -dy, dx
    pt1 = (int(tip[0] + dx * size + px * size * 0.5),
           int(tip[1] + dy * size + py * size * 0.5))
    pt2 = (int(tip[0] + dx * size - px * size * 0.5),
           int(tip[1] + dy * size - py * size * 0.5))

    cv2.line(img, tip, pt1, color, 1, cv2.LINE_AA)
    cv2.line(img, tip, pt2, color, 1, cv2.LINE_AA)


def draw_border_analysis(
    warped_card: np.ndarray,
    content_rect: "ContentRect",
    centering_result: "CenteringResult",
    show_details: bool = True,
) -> np.ndarray:
    """Draw centering analysis annotations on the card image.

    The summary score bar sits **above** the card so it never obscures
    the card artwork.

    Args:
        warped_card: BGR image of perspective-corrected card.
        content_rect: Detected content boundary.
        centering_result: Centering analysis data.
        show_details: Whether to draw detailed measurement labels.

    Returns:
        Annotated BGR image (card_h + bar_height tall).
    """
    bar_height = 40
    card_h, card_w = warped_card.shape[:2]
    grade_color = GRADE_COLORS.get(centering_result.grade, COLORS["grade_b"])

    # ── Annotate the card first (all coords in card-local space) ──
    result = warped_card.copy()

    # ---- 1. Outer card border (image edges) ----
    cv2.rectangle(result, (0, 0), (card_w - 1, card_h - 1),
                  COLORS["outer_border"], 2)

    # ---- 2. Inner content border ----
    inner_x1 = content_rect.x
    inner_y1 = content_rect.y
    inner_x2 = content_rect.x + content_rect.w
    inner_y2 = content_rect.y + content_rect.h
    cv2.rectangle(result, (inner_x1, inner_y1), (inner_x2, inner_y2),
                  COLORS["inner_border"], 2)

    if not show_details:
        # Even without details, still extend and add the bar
        extended_simple = np.zeros((card_h + bar_height, card_w, 3), dtype=np.uint8)
        extended_simple[:bar_height, :] = (35, 35, 35)
        extended_simple[bar_height:, :] = result
        summary = (f"H:{centering_result.h_centering_ratio:.3f} "
                   f"V:{centering_result.v_centering_ratio:.3f} "
                   f"Score:{centering_result.total_score} "
                   f"Grade:{centering_result.grade}")
        cv2.putText(extended_simple, summary, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return extended_simple

    # ---- 3. Dimension lines for border widths ----
    # Left border
    _draw_dimension_arrow(
        result,
        (5, card_h // 2),
        (inner_x1, card_h // 2),
        f"L:{centering_result.left_border_px}px ({centering_result.left_border_pct}%)",
    )
    # Right border
    _draw_dimension_arrow(
        result,
        (card_w - 5, card_h // 2),
        (inner_x2, card_h // 2),
        f"R:{centering_result.right_border_px}px ({centering_result.right_border_pct}%)",
    )
    # Top border
    _draw_dimension_arrow(
        result,
        (card_w // 3, 5),
        (card_w // 3, inner_y1),
        f"T:{centering_result.top_border_px}px ({centering_result.top_border_pct}%)",
    )
    # Bottom border
    _draw_dimension_arrow(
        result,
        (card_w // 3, card_h - 5),
        (card_w // 3, inner_y2),
        f"B:{centering_result.bottom_border_px}px ({centering_result.bottom_border_pct}%)",
    )

    # ---- 4. Card center crosshair ----
    cx, cy = card_w // 2, card_h // 2
    cv2.line(result, (cx - 30, cy), (cx + 30, cy), COLORS["crosshair"], 1, cv2.LINE_AA)
    cv2.line(result, (cx, cy - 30), (cx, cy + 30), COLORS["crosshair"], 1, cv2.LINE_AA)
    cv2.circle(result, (cx, cy), 6, COLORS["crosshair"], 1, cv2.LINE_AA)

    # ── Extend canvas above the card for the summary bar ──
    extended_h = card_h + bar_height
    extended = np.zeros((extended_h, card_w, 3), dtype=np.uint8)

    # Bar background
    extended[:bar_height, :] = (35, 35, 35)

    # Card below the bar
    extended[bar_height:, :] = result

    # ---- 5. Summary text on the bar ----
    summary = (f"H:{centering_result.h_centering_ratio:.3f}  "
               f"({centering_result.h_centering_score}/50)  "
               f"V:{centering_result.v_centering_ratio:.3f}  "
               f"({centering_result.v_centering_score}/50)  "
               f"Score: {centering_result.total_score}/100  "
               f"Grade: {centering_result.grade}")
    cv2.putText(extended, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 6. Grade badge (right side of bar) ----
    badge_w = 52
    badge_x = card_w - badge_w - 6
    badge_y = 4
    badge_h = bar_height - 8
    cv2.rectangle(extended, (badge_x, badge_y),
                  (badge_x + badge_w, badge_y + badge_h),
                  grade_color, -1)
    cv2.putText(extended, centering_result.grade,
                (badge_x + 6, badge_y + badge_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return extended


def draw_original_photo_annotation(
    original_image: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw detected card corners and outline on the original photo.

    Args:
        original_image: BGR original photo.
        corners: 4 corner points (4, 2) in [tl, tr, br, bl] order.
        color: Line color (BGR).

    Returns:
        Annotated image copy.
    """
    result = original_image.copy()
    pts = corners.astype(np.int32)

    # Draw outline
    cv2.polylines(result, [pts], True, color, 2, cv2.LINE_AA)

    # Draw corner circles and labels
    labels = ["TL", "TR", "BR", "BL"]
    for i, (pt, label) in enumerate(zip(pts, labels)):
        cv2.circle(result, tuple(pt), 8, (0, 0, 255), -1)
        cv2.circle(result, tuple(pt), 10, (255, 255, 255), 1, cv2.LINE_AA)
        offset = [(10, -10), (-40, -10), (-40, 20), (10, 20)][i]
        _put_text_with_bg(result, label,
                          (int(pt[0]) + offset[0], int(pt[1]) + offset[1]),
                          font_scale=0.5)

    return result
