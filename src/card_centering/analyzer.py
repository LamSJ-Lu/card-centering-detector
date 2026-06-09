"""Centering calculation and grading for TCG/sports cards.

Computes border widths on all 4 sides, symmetry ratios, and centering scores
using PSA/BGS-style grading standards.
"""

import math
from dataclasses import dataclass, field


@dataclass
class CenteringResult:
    """Complete centering analysis result."""

    # Border widths in pixels
    left_border_px: int
    right_border_px: int
    top_border_px: int
    bottom_border_px: int

    # Border widths as percentage of card dimension
    left_border_pct: float
    right_border_pct: float
    top_border_pct: float
    bottom_border_pct: float

    # Horizontal symmetry
    h_centering_ratio: float       # min(L,R)/max(L,R), 1.0 = perfect
    h_centering_score: int         # 0-50

    # Vertical symmetry
    v_centering_ratio: float       # min(T,B)/max(T,B), 1.0 = perfect
    v_centering_score: int         # 0-50

    # Overall
    total_score: int               # 0-100
    grade: str                     # "A" | "B" | "C" | "D"
    grade_label: str               # 优秀 | 良好 | 一般 | 偏差大

    # Detailed analysis
    centering_detail: str = ""
    suggestions: list[str] = field(default_factory=list)

    @property
    def h_border_ratio_str(self) -> str:
        """Horizontal L:R as a display string."""
        return f"{self.left_border_px}:{self.right_border_px}"

    @property
    def v_border_ratio_str(self) -> str:
        """Vertical T:B as a display string."""
        return f"{self.top_border_px}:{self.bottom_border_px}"


def _score_from_ratio(ratio: float) -> float:
    """Convert a symmetry ratio to a 0.0 ~ 1.0 score.

    Uses a smooth scoring curve based on PSA/BGS grading standards:
    - ratio >= 0.909 (≈ 55/45 or better) → 1.0 (gem mint centering)
    - ratio >= 0.833 (≈ 60/40) → 0.75
    - ratio >= 0.667 (≈ 70/30) → 0.5
    - ratio <  0.667 → linear degradation
    """
    if ratio >= 0.909:  # 55/45
        return 1.0
    elif ratio >= 0.833:  # 55/45 ~ 60/40
        # Linear interpolation: 0.75 ~ 1.0
        return 0.75 + (ratio - 0.833) / (0.909 - 0.833) * 0.25
    elif ratio >= 0.667:  # 60/40 ~ 70/30
        # Linear interpolation: 0.5 ~ 0.75
        return 0.5 + (ratio - 0.667) / (0.833 - 0.667) * 0.25
    elif ratio >= 0.5:
        # 70/30 ~ 80/20
        return 0.25 + (ratio - 0.5) / (0.667 - 0.5) * 0.25
    else:
        return max(0.0, ratio * 0.5)  # worse than 80/20


def _grade_from_score(total: int) -> tuple[str, str]:
    """Map a total score (0-100) to grade and Chinese label.

    Returns (grade_letter, grade_label).
    """
    if total >= 90:
        return ("A", "优秀")
    elif total >= 75:
        return ("B", "良好")
    elif total >= 60:
        return ("C", "一般")
    else:
        return ("D", "偏差大")


def compute_centering(
    card_size: tuple[int, int],
    content_rect: "ContentRect",
) -> CenteringResult:
    """Compute centering metrics from card dimensions and content rectangle.

    Args:
        card_size: (width, height) of the perspective-corrected card image.
        content_rect: ContentRect with inner content bounding box.

    Returns:
        CenteringResult with full analysis.
    """
    card_w, card_h = card_size

    # Border widths (pixels)
    left = content_rect.x
    right = card_w - content_rect.x - content_rect.w
    top = content_rect.y
    bottom = card_h - content_rect.y - content_rect.h

    # Border percentages
    left_pct = left / card_w * 100.0
    right_pct = right / card_w * 100.0
    top_pct = top / card_h * 100.0
    bottom_pct = bottom / card_h * 100.0

    # Symmetry ratios
    if left + right > 0:
        h_ratio = min(left, right) / max(left, right) if max(left, right) > 0 else 1.0
    else:
        h_ratio = 1.0

    if top + bottom > 0:
        v_ratio = min(top, bottom) / max(top, bottom) if max(top, bottom) > 0 else 1.0
    else:
        v_ratio = 1.0

    # Scores (0-50 each)
    h_score = round(_score_from_ratio(h_ratio) * 50)
    v_score = round(_score_from_ratio(v_ratio) * 50)
    total = h_score + v_score

    # Grade
    grade, grade_label = _grade_from_score(total)

    # Detail description
    detail_parts = []
    if h_ratio >= 0.95:
        detail_parts.append("左右基本对称")
    elif left > right:
        detail_parts.append("左边框宽于右边框")
    else:
        detail_parts.append("右边框宽于左边框")

    if v_ratio >= 0.95:
        detail_parts.append("上下基本对称")
    elif top > bottom:
        detail_parts.append("上边框宽于下边框")
    else:
        detail_parts.append("下边框宽于上边框")

    # Suggestions
    suggestions = []
    h_diff = abs(left - right)
    v_diff = abs(top - bottom)

    if h_ratio < 0.999 and h_diff >= 2:
        narrower = "左" if left < right else "右"
        suggestions.append(
            f"水平方向: {narrower}边框偏窄 (差 {h_diff}px / {abs(left_pct - right_pct):.1f}%)"
        )
    else:
        suggestions.append("水平方向: 居中良好 ✓")

    if v_ratio < 0.999 and v_diff >= 2:
        narrower = "上" if top < bottom else "下"
        suggestions.append(
            f"垂直方向: {narrower}边框偏窄 (差 {v_diff}px / {abs(top_pct - bottom_pct):.1f}%)"
        )
    else:
        suggestions.append("垂直方向: 居中良好 ✓")

    # Round percentages for display
    return CenteringResult(
        left_border_px=left,
        right_border_px=right,
        top_border_px=top,
        bottom_border_px=bottom,
        left_border_pct=round(left_pct, 1),
        right_border_pct=round(right_pct, 1),
        top_border_pct=round(top_pct, 1),
        bottom_border_pct=round(bottom_pct, 1),
        h_centering_ratio=round(h_ratio, 4),
        h_centering_score=h_score,
        v_centering_ratio=round(v_ratio, 4),
        v_centering_score=v_score,
        total_score=total,
        grade=grade,
        grade_label=grade_label,
        centering_detail="；".join(detail_parts),
        suggestions=suggestions,
    )
