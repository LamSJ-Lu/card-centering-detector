"""PySide6 GUI for the TCG Card Centering Detector.

Main window with image display, interactive editing (with magnifier),
analysis report panel, and camera capture support.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QPoint, Signal, QSize,
)
from PySide6.QtGui import (
    QAction, QPixmap, QImage, QPainter, QPen, QColor, QBrush,
    QFont, QPainterPath, QCursor, QDragEnterEvent, QDropEvent,
    QMouseEvent, QWheelEvent, QKeyEvent, QResizeEvent, QCloseEvent,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolBar, QStatusBar, QMenuBar, QMenu,
    QFileDialog, QMessageBox, QScrollArea, QProgressBar,
    QListWidget, QListWidgetItem, QFrame, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem,
    QApplication, QStyle, QSizePolicy, QGroupBox, QDialog,
    QDialogButtonBox, QComboBox,
)

from card_centering.detector import (
    detect_and_correct, CardOutline, detect_card_corners,
    perspective_correct, _load_image_safe, _compute_card_aspect,
)
from card_centering.border_detector import detect_content_rect, ContentRect
from card_centering.analyzer import compute_centering, CenteringResult
from card_centering.visualizer import (
    draw_border_analysis, draw_original_photo_annotation,
    GRADE_COLORS,
)
from card_centering.adjuster import (
    CornerAdjustment, BorderAdjustment,
    MagnifierState, render_magnifier,
    CORNER_HANDLE_RADIUS, EDGE_HANDLE_RADIUS,
    corners_from_content_rect, content_rect_from_corners,
    _point_to_segment_dist, INNER_LINE_HIT_THRESHOLD,
)
from card_centering.camera import (
    CameraCapture, draw_alignment_guide, list_cameras,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR image to QPixmap."""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """Convert OpenCV BGR image to QImage (for painting)."""
    if cv_img is None or cv_img.size == 0:
        return QImage()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


# ── Magnifier Widget ──────────────────────────────────────────────────────────

class MagnifierWidget(QWidget):
    """Floating magnifier window for precise corner/edge adjustment.

    Stays on top of the main window, follows the mouse cursor during
    drag operations, and shows a pixel-level zoomed view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._state = MagnifierState()
        self._rendered = np.zeros((250, 250, 3), dtype=np.uint8)
        self._pixmap: QPixmap | None = None
        self.setFixedSize(250, 250)
        self.hide()

    def show_at(self, screen_pos: QPoint, state: MagnifierState):
        """Show magnifier at screen position with updated state."""
        self._state = state
        self._rendered = render_magnifier(state)
        self._pixmap = _cv2_to_qpixmap(self._rendered)
        self.move(screen_pos - QPoint(125, 125))
        if not self.isVisible():
            self.show()
        self.update()

    def hide_magnifier(self):
        """Hide the magnifier."""
        self.hide()
        self._pixmap = None

    def paintEvent(self, event):
        """Paint the magnifier with circular mask."""
        if self._pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        size = self.width()
        radius = size // 2

        if self._state.shape == "circle":
            # Circular clip path
            path = QPainterPath()
            path.addEllipse(0, 0, size, size)
            painter.setClipPath(path)

        # Draw the magnified image
        if self._state.shape == "rectangle":
            painter.drawPixmap(0, 0, self._pixmap)
        else:
            # Center the pixmap in the circle
            painter.drawPixmap(0, 0, self._pixmap)

        if self._state.shape == "circle":
            # Draw outer ring
            painter.setClipping(False)
            pen = QPen(QColor(80, 80, 80), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(1, 1, size - 2, size - 2)

            # Outer highlight
            pen2 = QPen(QColor(180, 180, 180, 100), 1)
            painter.setPen(pen2)
            painter.drawEllipse(2, 2, size - 4, size - 4)

        elif self._state.shape == "rectangle":
            # Draw border
            pen = QPen(QColor(80, 80, 80), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(0, 0, size - 1, size - 1)

        painter.end()


# ── Interactive Graphics View ─────────────────────────────────────────────────

class CardGraphicsView(QGraphicsView):
    """QGraphicsView with support for pan, zoom, and interactive editing.

    Emits signals when the user interacts with edit handles.
    """

    corner_moved = Signal(int, float, float)       # index, x, y
    edge_moved = Signal(str, float, float)          # edge_name, x/y, value
    magnifier_requested = Signal(object)            # MagnifierState or None
    edit_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._handle_items: list[QGraphicsEllipseItem] = []
        self._line_items: list[QGraphicsLineItem] = []
        self._overlay_rect: QGraphicsRectItem | None = None

        # State
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._panning = False
        self._last_pan_pos = QPoint()

        # Edit mode state
        self.edit_mode: str | None = None  # "outer_corners" | "inner_borders" | None
        self._corner_adjustment: CornerAdjustment | None = None
        self._dragging_handle = False
        self._selected_handle_idx: int | None = None
        self._selected_edge: str | None = None    # 'left'|'right'|'top'|'bottom' for inner edge drag
        self._clamp_rect: tuple[int, int] | None = None  # (w, h) for inner-mode clamping
        self._source_image: np.ndarray | None = None  # For magnifier
        self._image_offset = QPointF(0, 0)  # Offset of image in scene
        self._bar_height = 0  # Pixels of summary bar above card (visualizer.py)

        # View settings
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

        # Mouse tracking for magnifier
        self.setMouseTracking(True)

    # ── Public API ──────────────────────────────────────────────────────────

    def set_image(self, cv_img: np.ndarray, bar_height: int = 0):
        """Display a BGR image in the view, replacing any existing content.

        Args:
            cv_img: BGR image.
            bar_height: Height of summary bar drawn above the card
                        (0 = no bar, for original photos).
        """
        self._scene.clear()
        self._pixmap_item = None
        self._handle_items.clear()
        self._line_items.clear()
        self._overlay_rect = None

        self._source_image = cv_img
        self._bar_height = bar_height
        pixmap = _cv2_to_qpixmap(cv_img)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._image_offset = QPointF(0, 0)
        self._zoom = 1.0
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def get_pixmap_rect(self) -> QRectF:
        """Get the pixmap rect in scene coordinates."""
        if self._pixmap_item:
            return self._pixmap_item.sceneBoundingRect()
        return QRectF()

    def scene_to_image(self, scene_pos: QPointF) -> tuple[float, float]:
        """Convert scene coordinates to image pixel coordinates.

        When a summary bar sits above the card, the returned y is card-local
        (bar height already subtracted) so edit-mode comparisons work in
        card space.
        """
        rect = self.get_pixmap_rect()
        x = scene_pos.x() - rect.x()
        y = scene_pos.y() - rect.y() - self._bar_height
        return x, y

    def image_to_scene(self, img_x: float, img_y: float) -> QPointF:
        """Convert image pixel coordinates to scene coordinates.

        When a summary bar sits above the card, the card-local img_y is
        shifted down by bar_height so handles render on the card, not the bar.
        """
        rect = self.get_pixmap_rect()
        return QPointF(rect.x() + img_x, rect.y() + img_y + self._bar_height)

    # ── Zoom ────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        """Mouse wheel zoom."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom * factor

        if self._min_zoom <= new_zoom <= self._max_zoom:
            self._zoom = new_zoom
            self.scale(factor, factor)
        event.accept()

    # ── Mouse Events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press: start pan, corner drag, or edge drag.

        In inner edit mode: corners take priority, then edge-lines, then pan.
        """
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            # In edit mode, first try to grab a corner handle
            if self.edit_mode and self._corner_adjustment:
                scene_pos = self.mapToScene(event.pos())
                img_pos = self.scene_to_image(scene_pos)
                idx = self._corner_adjustment.select_handle(img_pos)
                if idx is not None:
                    self._dragging_handle = True
                    self._selected_handle_idx = idx
                    self._selected_edge = None
                    self._start_magnifier()
                    event.accept()
                    return

                # Inner-border mode: next try edge-line hit
                if self.edit_mode == "inner_borders":
                    edge = self._select_inner_edge(img_pos)
                    if edge is not None:
                        self._dragging_handle = True
                        self._selected_edge = edge
                        self._selected_handle_idx = None
                        self._start_magnifier()
                        event.accept()
                        return

            # Left-click on empty space in edit mode → pan
            if self.edit_mode:
                self._panning = True
                self._last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move: pan or drag handle/edge."""
        if self._panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._dragging_handle:
            scene_pos = self.mapToScene(event.pos())
            img_pos = self.scene_to_image(scene_pos)
            self._update_drag(img_pos)
            self._update_magnifier()
            event.accept()
            return

        # Update cursor for hover
        if self.edit_mode and self._corner_adjustment:
            scene_pos = self.mapToScene(event.pos())
            img_pos = self.scene_to_image(scene_pos)
            idx = self._corner_adjustment.select_handle(img_pos)
            if idx is not None:
                self.setCursor(Qt.CrossCursor)
                return

            if self.edit_mode == "inner_borders":
                edge = self._select_inner_edge(img_pos)
                if edge is not None:
                    if edge in ("left", "right"):
                        self.setCursor(Qt.SizeHorCursor)
                    else:
                        self.setCursor(Qt.SizeVerCursor)
                    return

            self.setCursor(Qt.OpenHandCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release: end pan or end handle/edge drag."""
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.OpenHandCursor if self.edit_mode else Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            if self._dragging_handle:
                self._dragging_handle = False
                self._selected_handle_idx = None
                self._selected_edge = None
                self._stop_magnifier()
                self.edit_finished.emit()
                event.accept()
                return

            if self._panning:
                self._panning = False
                self.setCursor(Qt.OpenHandCursor if self.edit_mode else Qt.ArrowCursor)
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Keyboard shortcuts for edit modes — corner + edge control."""
        if not self.edit_mode or not self._corner_adjustment:
            super().keyPressEvent(event)
            return

        delta = 10 if event.modifiers() & Qt.ShiftModifier else 1
        adj = self._corner_adjustment
        cw, ch = self._clamp_rect or (99999, 99999)

        # ── Shared control keys ──
        if event.key() == Qt.Key_Escape:
            adj.cancel()
            self._selected_edge = None
            self._selected_handle_idx = None
            self._stop_magnifier()
            self.edit_finished.emit()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            adj.confirm()
            self._selected_edge = None
            self._selected_handle_idx = None
            self._stop_magnifier()
            self.edit_finished.emit()
            return

        # ── Corner drag ──
        if self._selected_handle_idx is not None:
            idx = self._selected_handle_idx
            nx, ny = float(adj.corners[idx][0]), float(adj.corners[idx][1])
            if event.key() == Qt.Key_Left:   nx -= delta
            elif event.key() == Qt.Key_Right: nx += delta
            elif event.key() == Qt.Key_Up:    ny -= delta
            elif event.key() == Qt.Key_Down:  ny += delta
            else: return super().keyPressEvent(event)
            if self.edit_mode == "inner_borders":
                nx = max(0, min(cw, nx))
                ny = max(0, min(ch, ny))
            adj.move_handle(idx, (nx, ny))
            self._update_edit_overlay()
            self._update_magnifier()
            self.edit_finished.emit()
            return

        # ── Edge drag (inner mode only) ──
        if self._selected_edge is not None:
            edge = self._selected_edge
            corners = adj.corners
            if edge == "left":
                nx = corners[0][0] - delta
                if event.key() == Qt.Key_Right: nx = corners[0][0] + delta
                elif event.key() != Qt.Key_Left: return super().keyPressEvent(event)
                nx = max(0, min(corners[1][0] - 5, nx))
                corners[0][0] = nx; corners[3][0] = nx
            elif edge == "right":
                nx = corners[1][0] + delta
                if event.key() == Qt.Key_Left: nx = corners[1][0] - delta
                elif event.key() != Qt.Key_Right: return super().keyPressEvent(event)
                nx = max(corners[0][0] + 5, min(cw, nx))
                corners[1][0] = nx; corners[2][0] = nx
            elif edge == "top":
                ny = corners[0][1] - delta
                if event.key() == Qt.Key_Down: ny = corners[0][1] + delta
                elif event.key() != Qt.Key_Up: return super().keyPressEvent(event)
                ny = max(0, min(corners[3][1] - 5, ny))
                corners[0][1] = ny; corners[1][1] = ny
            elif edge == "bottom":
                ny = corners[3][1] + delta
                if event.key() == Qt.Key_Up: ny = corners[3][1] - delta
                elif event.key() != Qt.Key_Down: return super().keyPressEvent(event)
                ny = max(corners[0][1] + 5, min(ch, ny))
                corners[2][1] = ny; corners[3][1] = ny
            self._update_edit_overlay()
            self._update_magnifier()
            self.edit_finished.emit()
            return

        super().keyPressEvent(event)

    # ── Edit Mode Management ─────────────────────────────────────────────────

    def enter_outer_corner_edit(self, corners: np.ndarray):
        """Enter edit mode for outer card corners."""
        self.edit_mode = "outer_corners"
        self._corner_adjustment = CornerAdjustment(corners=corners.copy())
        self._clamp_rect = None
        self._update_edit_overlay()
        self.setCursor(Qt.CrossCursor)

    def enter_inner_border_edit(self, content_rect: ContentRect,
                                card_w: int, card_h: int):
        """Enter edit mode for inner content borders (4-corner)."""
        self.edit_mode = "inner_borders"
        inner_corners = corners_from_content_rect(content_rect)
        self._corner_adjustment = CornerAdjustment(corners=inner_corners)
        self._clamp_rect = (card_w, card_h)
        self._update_edit_overlay()
        self.setCursor(Qt.OpenHandCursor)

    def exit_edit_mode(self):
        """Exit any edit mode."""
        self.edit_mode = None
        self._corner_adjustment = None
        self._clamp_rect = None
        self._selected_edge = None
        self._selected_handle_idx = None
        self._clear_edit_overlay()
        self._stop_magnifier()
        self.setCursor(Qt.ArrowCursor)

    def get_corner_adjustment(self) -> CornerAdjustment | None:
        return self._corner_adjustment

    # ── Internal ─────────────────────────────────────────────────────────────

    def _select_inner_edge(self, img_pos: tuple[float, float]) -> str | None:
        """Return edge name if click is near an inner border line segment.

        Checks distance to each of the 4 line segments connecting the corners.
        Returns None if no edge is close enough.
        """
        adj = self._corner_adjustment
        if adj is None or adj.corners is None:
            return None
        corners = adj.corners
        px, py = img_pos
        edges = [
            ("left",   0, 3),
            ("top",    0, 1),
            ("right",  1, 2),
            ("bottom", 3, 2),
        ]
        best: str | None = None
        best_dist = INNER_LINE_HIT_THRESHOLD
        for name, i, j in edges:
            d = _point_to_segment_dist(
                px, py,
                float(corners[i][0]), float(corners[i][1]),
                float(corners[j][0]), float(corners[j][1]),
            )
            if d < best_dist:
                best_dist = d
                best = name
        return best

    def _update_drag(self, img_pos: tuple[float, float]):
        """Update handle or edge position during drag."""
        adj = self._corner_adjustment
        if adj is None:
            return

        if self._selected_handle_idx is not None:
            x, y = img_pos
            if self.edit_mode == "inner_borders" and self._clamp_rect:
                cw, ch = self._clamp_rect
                x = max(0, min(cw, x))
                y = max(0, min(ch, y))
            adj.move_handle(self._selected_handle_idx, (x, y))
            self._update_edit_overlay()
            return

        if self._selected_edge is not None and self._clamp_rect:
            edge = self._selected_edge
            corners = adj.corners
            cw, ch = self._clamp_rect
            if edge == "left":
                nx = max(0, min(corners[1][0] - 5, int(round(img_pos[0]))))
                corners[0][0] = nx
                corners[3][0] = nx
            elif edge == "right":
                nx = max(corners[0][0] + 5, min(cw, int(round(img_pos[0]))))
                corners[1][0] = nx
                corners[2][0] = nx
            elif edge == "top":
                ny = max(0, min(corners[3][1] - 5, int(round(img_pos[1]))))
                corners[0][1] = ny
                corners[1][1] = ny
            elif edge == "bottom":
                ny = max(corners[0][1] + 5, min(ch, int(round(img_pos[1]))))
                corners[2][1] = ny
                corners[3][1] = ny
            self._update_edit_overlay()

    def _update_edit_overlay(self):
        """Redraw edit handles — both modes use 4-corner editing.

        Inner border lines are thicker and deeper in colour for visibility.
        """
        self._clear_edit_overlay()

        if not self.edit_mode or not self._corner_adjustment:
            return

        corners = self._corner_adjustment.corners

        if self.edit_mode == "outer_corners":
            line_color = QColor(255, 255, 0)       # Yellow
            line_width = 2
            line_style = Qt.DashLine
            handle_pen = QPen(QColor(255, 0, 0), 2)  # Red
            handle_brush = Qt.NoBrush  # transparent — don't obscure the photo
        else:  # inner_borders — deeper cyan, thicker, solid
            line_color = QColor(0, 200, 200)       # Deep cyan
            line_width = 3
            line_style = Qt.SolidLine
            handle_pen = QPen(QColor(0, 180, 0), 2)  # Green
            handle_brush = QBrush(QColor(100, 255, 100, 180))

        # Connect corners with lines
        for i in range(4):
            p1 = self.image_to_scene(corners[i][0], corners[i][1])
            p2 = self.image_to_scene(corners[(i + 1) % 4][0],
                                     corners[(i + 1) % 4][1])
            line = self._scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(),
                                       QPen(line_color, line_width, line_style))
            self._line_items.append(line)

        # Corner handles
        for i, c in enumerate(corners):
            pt = self.image_to_scene(c[0], c[1])
            r = CORNER_HANDLE_RADIUS
            handle = self._scene.addEllipse(
                pt.x() - r, pt.y() - r, r * 2, r * 2,
                handle_pen, handle_brush,
            )
            self._handle_items.append(handle)

    def _clear_edit_overlay(self):
        """Remove all edit handle graphics items."""
        for item in self._handle_items:
            self._scene.removeItem(item)
        self._handle_items.clear()
        for item in self._line_items:
            self._scene.removeItem(item)
        self._line_items.clear()
        if self._overlay_rect:
            self._scene.removeItem(self._overlay_rect)
            self._overlay_rect = None

    # ── Magnifier ─────────────────────────────────────────────────────────────

    def _start_magnifier(self):
        """Start showing magnifier."""
        self._send_magnifier_state()

    def _stop_magnifier(self):
        """Hide magnifier."""
        self.magnifier_requested.emit(None)

    def _update_magnifier(self):
        """Update magnifier content and position."""
        self._send_magnifier_state()

    def _send_magnifier_state(self):
        """Build magnifier state and emit signal.

        Corner drag → circular magnifier (5×) centred on the corner.
        Edge drag (inner mode) → rectangular magnifier (4×) along the edge
        so the guideline is visible but doesn't obscure the card.
        """
        source = self._source_image
        if source is None:
            self.magnifier_requested.emit(None)
            return

        center = (0, 0)
        shape = "circle"
        zoom = 5.0
        radius = 25
        adj = self._corner_adjustment
        bar_off = self._bar_height  # summary-bar offset (non-zero only in inner mode)

        if adj is not None and self._selected_handle_idx is not None:
            # ── Corner drag ──
            c = adj.corners[self._selected_handle_idx]
            center = (int(c[0]), int(c[1]) + bar_off)
            shape = "circle"
            zoom = 5.0
            radius = 25

        elif adj is not None and self._selected_edge is not None:
            # ── Edge drag (inner mode only) ──
            edge = self._selected_edge
            corners = adj.corners
            if edge == "left":
                cx = int(corners[0][0])
                cy = int((corners[0][1] + corners[3][1]) / 2) + bar_off
            elif edge == "right":
                cx = int(corners[1][0])
                cy = int((corners[1][1] + corners[2][1]) / 2) + bar_off
            elif edge == "top":
                cx = int((corners[0][0] + corners[1][0]) / 2)
                cy = int(corners[0][1]) + bar_off
            else:  # bottom
                cx = int((corners[2][0] + corners[3][0]) / 2)
                cy = int(corners[2][1]) + bar_off
            center = (cx, cy)
            shape = "rectangle"
            zoom = 4.0
            radius = 30

        state = MagnifierState(
            enabled=True,
            source_image=source,
            center_pos=center,
            zoom=zoom,
            source_radius=radius,
            display_size=250,
            shape=shape,
            grid_enabled=True,
        )
        self.magnifier_requested.emit(state)


# ── Camera Dialog ─────────────────────────────────────────────────────────────

class CameraDialog(QDialog):
    """Dialog for camera selection and capture."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("摄像头拍摄")
        self.setMinimumSize(640, 480)
        self._camera: CameraCapture | None = None
        self._captured_frame: np.ndarray | None = None
        self._timer: QTimer | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Preview label
        self._preview_label = QLabel("Select a camera and click Start")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(640, 400)
        self._preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        layout.addWidget(self._preview_label, 1)

        # Controls
        ctrl_layout = QHBoxLayout()

        cam_label = QLabel("摄像头:")
        self._camera_combo = QComboBox()
        self._refresh_cameras()
        ctrl_layout.addWidget(cam_label)
        ctrl_layout.addWidget(self._camera_combo, 1)

        self._start_btn = QPushButton("▶ 开始预览")
        self._start_btn.clicked.connect(self._start_preview)
        ctrl_layout.addWidget(self._start_btn)

        self._capture_btn = QPushButton("📷 拍摄")
        self._capture_btn.setEnabled(False)
        self._capture_btn.clicked.connect(self._capture)
        ctrl_layout.addWidget(self._capture_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_preview)
        ctrl_layout.addWidget(self._stop_btn)

        layout.addLayout(ctrl_layout)

        # Dialog buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _refresh_cameras(self):
        self._camera_combo.clear()
        cams = list_cameras()
        if not cams:
            self._camera_combo.addItem("No cameras found", -1)
        else:
            for cam in cams:
                self._camera_combo.addItem(
                    f"{cam['name']} ({cam['resolution']})", cam['index']
                )

    def _start_preview(self):
        idx = self._camera_combo.currentData()
        if idx is None or idx < 0:
            QMessageBox.warning(self, "错误", "没有可用的摄像头")
            return

        self._camera = CameraCapture(idx)
        if not self._camera.start():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(33)  # ~30 fps

        self._start_btn.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._camera_combo.setEnabled(False)

    def _stop_preview(self):
        if self._timer:
            self._timer.stop()
            self._timer = None
        if self._camera:
            self._camera.stop()
            self._camera = None

        self._preview_label.setText("Preview stopped")
        self._preview_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888;")
        self._start_btn.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)

    def _update_preview(self):
        if self._camera is None:
            return
        frame = self._camera.read()
        if frame is None:
            return
        # Draw alignment guide
        frame_with_guide = draw_alignment_guide(frame)
        pixmap = _cv2_to_qpixmap(frame_with_guide)
        scaled = pixmap.scaled(
            self._preview_label.width(), self._preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
        self._preview_label.setStyleSheet("background-color: #000;")

    def _capture(self):
        if self._camera is None:
            return
        frame = self._camera.read()
        if frame is not None:
            self._captured_frame = frame.copy()
            # Show a flash effect
            flash = np.ones_like(frame) * 255
            result = cv2.addWeighted(frame, 0.5, flash, 0.5, 0)
            self._preview_label.setPixmap(
                _cv2_to_qpixmap(result).scaled(
                    self._preview_label.width(), self._preview_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )

    def _accept(self):
        self._stop_preview()
        if self._captured_frame is not None:
            self.accept()
        else:
            QMessageBox.information(self, "提示", "请先拍摄一张照片")

    def closeEvent(self, event: QCloseEvent):
        self._stop_preview()
        super().closeEvent(event)

    def get_frame(self) -> np.ndarray | None:
        return self._captured_frame


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window for TCG Card Centering Detector."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📐 TCG 卡片居中度检测")
        self.setMinimumSize(1200, 750)

        # Application state
        self._original_image: np.ndarray | None = None
        self._card_outline: CardOutline | None = None
        self._content_rect: ContentRect | None = None
        self._centering_result: CenteringResult | None = None
        self._current_view: str = "warped"  # "original" | "warped"
        self._current_file: str | None = None

        # Magnifier
        self._magnifier = MagnifierWidget()
        self._magnifier.hide()

        self._setup_ui()
        self._setup_connections()

    # ── UI Setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        """Build the complete UI."""
        # -- Menu Bar --
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件(&F)")
        open_action = QAction("📂 打开图片...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        export_action = QAction("💾 导出报告...", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self._export_report)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        camera_menu = menu_bar.addMenu("相机(&C)")
        capture_action = QAction("📷 拍摄照片...", self)
        capture_action.setShortcut("Ctrl+T")
        capture_action.triggered.connect(self._open_camera)
        camera_menu.addAction(capture_action)

        help_menu = menu_bar.addMenu("帮助(&H)")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # -- Toolbar --
        toolbar = QToolBar("工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        self._open_btn = QPushButton("📂 打开图片")
        self._open_btn.clicked.connect(self._open_file)
        toolbar.addWidget(self._open_btn)

        self._camera_btn = QPushButton("📷 拍摄")
        self._camera_btn.clicked.connect(self._open_camera)
        toolbar.addWidget(self._camera_btn)

        toolbar.addSeparator()

        self._export_btn = QPushButton("💾 导出报告")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_report)
        toolbar.addWidget(self._export_btn)

        # -- Central Widget --
        central = QWidget()
        self.setCentralWidget(central)

        # Splitter: image left | report right
        splitter = QSplitter(Qt.Horizontal)

        # --- Image View ---
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self._graphics_view = CardGraphicsView()
        image_layout.addWidget(self._graphics_view)

        # View / Edit toggle — two mutual-exclusive buttons (switch mode)
        view_toggle_layout = QHBoxLayout()
        self._btn_outer = QPushButton("✏️ 调整外框")
        self._btn_outer.setCheckable(True)
        self._btn_outer.setEnabled(False)
        self._btn_outer.clicked.connect(self._switch_to_outer)
        self._btn_inner = QPushButton("📐 调整内框")
        self._btn_inner.setCheckable(True)
        self._btn_inner.setEnabled(False)
        self._btn_inner.clicked.connect(self._switch_to_inner)
        view_toggle_layout.addStretch()
        view_toggle_layout.addWidget(self._btn_outer)
        view_toggle_layout.addWidget(self._btn_inner)
        view_toggle_layout.addStretch()
        image_layout.addLayout(view_toggle_layout)

        splitter.addWidget(image_container)

        # --- Report Panel ---
        report_panel = QWidget()
        report_panel.setMaximumWidth(320)
        report_panel.setMinimumWidth(280)
        report_layout = QVBoxLayout(report_panel)
        report_layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("📊 居中度分析报告")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        report_layout.addWidget(title)

        # Separator
        report_layout.addWidget(self._make_separator())

        # Border widths group
        border_group = QGroupBox("边框宽度")
        border_layout = QVBoxLayout(border_group)
        self._label_left = QLabel("左边框: —")
        self._label_right = QLabel("右边框: —")
        self._label_top = QLabel("上边框: —")
        self._label_bottom = QLabel("下边框: —")
        for lbl in [self._label_left, self._label_right,
                     self._label_top, self._label_bottom]:
            border_layout.addWidget(lbl)
        report_layout.addWidget(border_group)

        # Symmetry ratios
        sym_group = QGroupBox("对称性")
        sym_layout = QVBoxLayout(sym_group)
        self._label_h_ratio = QLabel("水平对称比: —")
        self._label_v_ratio = QLabel("垂直对称比: —")
        sym_layout.addWidget(self._label_h_ratio)
        sym_layout.addWidget(self._label_v_ratio)
        report_layout.addWidget(sym_group)

        # Scores
        score_group = QGroupBox("评分")
        score_layout = QVBoxLayout(score_group)

        h_score_layout = QHBoxLayout()
        h_score_layout.addWidget(QLabel("水平:"))
        self._h_score_bar = QProgressBar()
        self._h_score_bar.setMaximum(50)
        self._h_score_bar.setFormat("%v/50")
        h_score_layout.addWidget(self._h_score_bar)
        score_layout.addLayout(h_score_layout)

        v_score_layout = QHBoxLayout()
        v_score_layout.addWidget(QLabel("垂直:"))
        self._v_score_bar = QProgressBar()
        self._v_score_bar.setMaximum(50)
        self._v_score_bar.setFormat("%v/50")
        v_score_layout.addWidget(self._v_score_bar)
        score_layout.addLayout(v_score_layout)

        total_layout = QHBoxLayout()
        total_layout.addWidget(QLabel("综合:"))
        self._total_score_bar = QProgressBar()
        self._total_score_bar.setMaximum(100)
        self._total_score_bar.setFormat("%v/100")
        total_layout.addWidget(self._total_score_bar)
        score_layout.addLayout(total_layout)

        self._grade_label = QLabel("评级: —")
        self._grade_label.setAlignment(Qt.AlignCenter)
        self._grade_label.setMinimumHeight(30)
        score_layout.addWidget(self._grade_label)

        report_layout.addWidget(score_group)

        # Detail
        detail_group = QGroupBox("分析详情")
        detail_layout = QVBoxLayout(detail_group)
        self._detail_label = QLabel("—")
        self._detail_label.setWordWrap(True)
        detail_layout.addWidget(self._detail_label)
        report_layout.addWidget(detail_group)

        # Suggestions
        sug_group = QGroupBox("💡 分析建议")
        sug_layout = QVBoxLayout(sug_group)
        self._suggestion_list = QListWidget()
        self._suggestion_list.setMaximumHeight(80)
        sug_layout.addWidget(self._suggestion_list)
        report_layout.addWidget(sug_group)

        # Card info
        info_group = QGroupBox("卡片信息")
        info_layout = QVBoxLayout(info_group)
        self._info_resolution = QLabel("图片尺寸: —")
        self._info_card_size = QLabel("卡片尺寸: —")
        self._info_ratio = QLabel("宽高比: —")
        self._info_confidence = QLabel("置信度: —")
        for lbl in [self._info_resolution, self._info_card_size,
                     self._info_ratio, self._info_confidence]:
            info_layout.addWidget(lbl)
        report_layout.addWidget(info_group)

        # Copy button
        self._copy_btn = QPushButton("📋 复制报告")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._copy_report)
        report_layout.addWidget(self._copy_btn)

        report_layout.addStretch()

        splitter.addWidget(report_panel)
        splitter.setSizes([850, 300])

        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(splitter)

        # -- Status Bar --
        self._status_bar = QStatusBar()
        self._status_label = QLabel("就绪 — 请打开一张卡片图片")
        self._status_bar.addWidget(self._status_label)
        self.setStatusBar(self._status_bar)

        # Enable drag-drop
        self.setAcceptDrops(True)

    def _make_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    def _setup_connections(self):
        """Connect signals/slots."""
        gv = self._graphics_view
        gv.corner_moved.connect(self._on_corner_moved)
        gv.edge_moved.connect(self._on_edge_moved)
        gv.edit_finished.connect(self._on_edit_finished)
        gv.magnifier_requested.connect(self._on_magnifier_requested)

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_image(path)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _open_file(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开卡片图片",
            str(Path.home() / "Pictures"),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if file_path:
            self._load_image(file_path)

    def _open_camera(self):
        """Open camera capture dialog."""
        dialog = CameraDialog(self)
        if dialog.exec() == QDialog.Accepted:
            frame = dialog.get_frame()
            if frame is not None:
                self._original_image = frame
                self._current_file = None
                self._process_image()

    def _load_image(self, path: str):
        """Load an image from file path."""
        img = _load_image_safe(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载图片:\n{path}")
            return
        self._original_image = img
        self._current_file = path
        self._process_image()

    def _process_image(self):
        """Run the full detection pipeline."""
        if self._original_image is None:
            return

        self._graphics_view.exit_edit_mode()
        self._btn_outer.setChecked(False)
        self._btn_inner.setChecked(False)

        self._status_label.setText("正在检测卡片...")
        QApplication.processEvents()

        # Step 1: Detect card corners + perspective correct
        outline = detect_and_correct(self._original_image)
        if outline is None:
            QMessageBox.warning(
                self, "检测失败",
                "未能检测到卡片。\n\n"
                "请确保:\n"
                "• 卡片在图片中清晰可见\n"
                "• 背景与卡片有明显对比\n"
                "• 卡片没有被严重遮挡\n\n"
                "您也可以手动调整外框角点。"
            )
            self._card_outline = None
            self._status_label.setText("检测失败 — 未找到卡片")
            return

        self._card_outline = outline

        # Step 2: Detect content border
        content_rect = detect_content_rect(outline.warped_image)
        if content_rect is None:
            QMessageBox.warning(
                self, "边框检测失败",
                "未能检测到卡片内容边框。\n\n"
                "您可以手动调整内框边界线。"
            )
            self._content_rect = None
            self._status_label.setText("边框检测失败 — 请手动调整")
            return

        self._content_rect = content_rect

        # Step 3: Compute centering
        self._centering_result = compute_centering(outline.size, content_rect)

        # Step 4: Update display
        self._update_display()
        self._update_report()

        # Enable editing and export
        self._btn_outer.setEnabled(True)
        self._btn_inner.setEnabled(True)
        self._btn_inner.setChecked(True)  # default: warped view
        self._export_btn.setEnabled(True)
        self._copy_btn.setEnabled(True)

        file_name = os.path.basename(self._current_file) if self._current_file else "摄像头拍摄"
        self._status_label.setText(
            f"{file_name}  |  "
            f"校正:{outline.width}×{outline.height}  |  "
            f"H:{self._centering_result.h_centering_ratio:.3f}  |  "
            f"V:{self._centering_result.v_centering_ratio:.3f}  |  "
            f"得分:{self._centering_result.total_score}  |  "
            f"{self._centering_result.grade}"
        )

    def _update_display(self):
        """Update the image display based on current view mode."""
        if self._current_view == "original" and self._original_image is not None:
            annotated = draw_original_photo_annotation(
                self._original_image,
                self._card_outline.corners if self._card_outline else None,
            ) if self._card_outline else self._original_image
            self._graphics_view.set_image(annotated)

        elif self._current_view == "warped" and self._card_outline is not None:
            if self._content_rect and self._centering_result:
                annotated = draw_border_analysis(
                    self._card_outline.warped_image,
                    self._content_rect,
                    self._centering_result,
                    show_details=True,
                )
            else:
                annotated = self._card_outline.warped_image
            # bar_height=40: the annotated image has a summary bar above the card
            self._graphics_view.set_image(annotated, bar_height=40)

    def _update_report(self):
        """Update the report panel with current analysis."""
        cr = self._centering_result
        if cr is None:
            return

        # Border widths
        self._label_left.setText(
            f"左边框: {cr.left_border_px}px ({cr.left_border_pct}%)")
        self._label_right.setText(
            f"右边框: {cr.right_border_px}px ({cr.right_border_pct}%)")
        self._label_top.setText(
            f"上边框: {cr.top_border_px}px ({cr.top_border_pct}%)")
        self._label_bottom.setText(
            f"下边框: {cr.bottom_border_px}px ({cr.bottom_border_pct}%)")

        # Symmetry
        self._label_h_ratio.setText(
            f"水平对称比: 左/右 = {cr.h_centering_ratio:.4f}")
        self._label_v_ratio.setText(
            f"垂直对称比: 上/下 = {cr.v_centering_ratio:.4f}")

        # Scores
        self._h_score_bar.setValue(cr.h_centering_score)
        self._v_score_bar.setValue(cr.v_centering_score)
        self._total_score_bar.setValue(cr.total_score)

        # Color the total score bar
        grade_colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "D": "#F44336"}
        color = grade_colors.get(cr.grade, "#888")
        self._total_score_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}")

        # Grade
        self._grade_label.setText(f"评级: {cr.grade} — {cr.grade_label}")
        bg_colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "D": "#F44336"}
        bg = bg_colors.get(cr.grade, "#888")
        self._grade_label.setStyleSheet(
            f"background-color: {bg}; color: white; font-size: 16px; "
            f"font-weight: bold; border-radius: 4px; padding: 4px;"
        )

        # Detail
        self._detail_label.setText(cr.centering_detail)

        # Suggestions
        self._suggestion_list.clear()
        for sug in cr.suggestions:
            self._suggestion_list.addItem(QListWidgetItem(sug))

        # Info
        if self._original_image is not None:
            oh, ow = self._original_image.shape[:2]
            self._info_resolution.setText(f"图片尺寸: {ow} × {oh}")
        if self._card_outline:
            self._info_card_size.setText(
                f"卡片尺寸: {self._card_outline.width} × {self._card_outline.height}")
            actual_ratio = self._card_outline.width / self._card_outline.height
            self._info_ratio.setText(f"宽高比: {actual_ratio:.3f} (标准≈0.716)")
        if self._content_rect:
            self._info_confidence.setText(f"检测置信度: {self._content_rect.confidence:.0%}")

    # ── View / Edit switches (mutually exclusive, pure switch mode) ───────────

    def _switch_to_outer(self):
        """Switch to outer corner edit on the original photo.

        Always activates. Both buttons stay enabled so the user can
        switch freely at any time with a single click.
        """
        self._btn_outer.setChecked(True)
        self._btn_inner.setChecked(False)

        self._current_view = "original"
        if self._card_outline:
            self._graphics_view.enter_outer_corner_edit(
                self._card_outline.corners)
        self._update_display()

    def _switch_to_inner(self):
        """Switch to inner border edit on the corrected card image.

        Always activates. Both buttons stay enabled so the user can
        switch freely at any time with a single click.
        """
        self._btn_inner.setChecked(True)
        self._btn_outer.setChecked(False)

        self._current_view = "warped"
        if self._content_rect and self._card_outline:
            self._graphics_view.enter_inner_border_edit(
                self._content_rect,
                self._card_outline.width,
                self._card_outline.height,
            )
        self._update_display()

    def _on_corner_moved(self, index: int, x: float, y: float):
        """Handle corner drag during outer edit."""
        pass  # State is updated directly in the view

    def _on_edge_moved(self, edge: str, x: float, y: float):
        """Handle edge drag during inner edit."""
        pass  # State is updated directly in the view

    def _on_edit_finished(self):
        """Called when user finishes a drag in edit mode."""
        if self._graphics_view.edit_mode == "outer_corners":
            adj = self._graphics_view.get_corner_adjustment()
            if adj and self._original_image is not None:
                # Re-run perspective correction with adjusted corners,
                # preserving the detected landscape/portrait orientation.
                card_aspect = _compute_card_aspect(adj.corners)
                outline = perspective_correct(
                    self._original_image, adj.corners,
                    target_aspect=card_aspect)
                self._card_outline = outline

                # Re-detect inner border
                content_rect = detect_content_rect(outline.warped_image)
                if content_rect:
                    self._content_rect = content_rect
                    self._centering_result = compute_centering(
                        outline.size, content_rect)
                    self._update_report()

                self._update_display()

        elif self._graphics_view.edit_mode == "inner_borders":
            adj = self._graphics_view.get_corner_adjustment()
            if adj and self._card_outline:
                cw, ch = self._card_outline.size
                self._content_rect = content_rect_from_corners(adj.corners, cw, ch)
                self._centering_result = compute_centering(
                    self._card_outline.size, self._content_rect)
                self._update_report()
                self._update_display()

    # ── Magnifier ─────────────────────────────────────────────────────────────

    def _on_magnifier_requested(self, state: MagnifierState | None):
        """Show or hide the magnifier."""
        if state is None or not state.enabled:
            self._magnifier.hide_magnifier()
            return

        # Position magnifier near cursor but offset
        cursor_pos = QCursor.pos()
        screen_pos = cursor_pos + QPoint(30, -30)
        self._magnifier.show_at(screen_pos, state)

    # ── Export / Copy ─────────────────────────────────────────────────────────

    def _export_report(self):
        """Export annotated image and text report."""
        if self._centering_result is None:
            return

        default_name = "card_centering_report"
        if self._current_file:
            base = os.path.splitext(os.path.basename(self._current_file))[0]
            default_name = f"{base}_centering_report"

        save_dir = str(Path.home() / "Pictures")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", f"{save_dir}/{default_name}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if not file_path:
            return

        # Generate annotated image
        if self._card_outline and self._content_rect and self._centering_result:
            annotated = draw_border_analysis(
                self._card_outline.warped_image,
                self._content_rect,
                self._centering_result,
                show_details=True,
            )
            # Use safe save for unicode paths
            data = cv2.imencode(os.path.splitext(file_path)[1], annotated)[1]
            data.tofile(file_path)

        # Also save text report
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        report_text = self._build_report_text()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        QMessageBox.information(
            self, "导出成功",
            f"报告已保存至:\n{file_path}\n{txt_path}"
        )

    def _copy_report(self):
        """Copy report text to clipboard."""
        text = self._build_report_text()
        QApplication.clipboard().setText(text)
        self._status_label.setText("报告已复制到剪贴板")

    def _build_report_text(self) -> str:
        """Build a text summary of the analysis."""
        cr = self._centering_result
        if cr is None:
            return "无分析数据"

        lines = [
            "══════════════════════════════",
            "  TCG 卡片居中度检测报告",
            "══════════════════════════════",
            "",
            "【边框宽度】",
            f"  左边框: {cr.left_border_px}px ({cr.left_border_pct}%)",
            f"  右边框: {cr.right_border_px}px ({cr.right_border_pct}%)",
            f"  上边框: {cr.top_border_px}px ({cr.top_border_pct}%)",
            f"  下边框: {cr.bottom_border_px}px ({cr.bottom_border_pct}%)",
            "",
            "【对称性】",
            f"  水平对称比: {cr.h_centering_ratio:.4f}",
            f"  垂直对称比: {cr.v_centering_ratio:.4f}",
            "",
            "【评分】",
            f"  水平得分: {cr.h_centering_score}/50",
            f"  垂直得分: {cr.v_centering_score}/50",
            f"  综合得分: {cr.total_score}/100",
            f"  评级: {cr.grade} — {cr.grade_label}",
            "",
            "【分析】",
            f"  {cr.centering_detail}",
            "",
            "【建议】",
        ]
        for s in cr.suggestions:
            lines.append(f"  • {s}")

        return "\n".join(lines)

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "📐 TCG 卡片居中度检测 v0.1.0\n\n"
            "检测 TCG/球星卡印刷内容的居中度，\n"
            "支持透视校正、自动边框检测、\n"
            "手动微调和放大镜辅助精确定位。\n\n"
            "评分标准参考 PSA/BGS 评级体系。"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent):
        self._magnifier.close()
        super().closeEvent(event)


def main():
    """Launch the application."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    app = QApplication(sys.argv)
    app.setApplicationName("TCG Card Centering Detector")
    app.setApplicationVersion("0.1.0")

    # Dark theme styling.
    # On macOS the native Aqua style automatically respects the system
    # dark / light appearance setting.  Forcing Fusion there would override
    # that and make the app look non-native, so we only apply the custom
    # palette on other platforms.
    import sys as _sys
    if not _sys.platform == "darwin":
        app.setStyle("Fusion")
        dark_palette = app.palette()
        from PySide6.QtGui import QPalette
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        app.setPalette(dark_palette)

    # Global stylesheet tweaks
    app.setStyleSheet("""
        QToolTip { color: #ffffff; background-color: #2a2a2a;
                   border: 1px solid #555; padding: 2px; }
        QGroupBox { font-weight: bold; color: #ccc;
                    border: 1px solid #555; border-radius: 4px;
                    margin-top: 8px; padding-top: 14px; }
        QGroupBox::title { subcontrol-origin: margin;
                           left: 10px; padding: 0 4px; }
        QPushButton { padding: 4px 12px; border: 1px solid #555;
                      border-radius: 3px; background-color: #3a3a3a; }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:pressed { background-color: #2a2a2a; }
        QPushButton:checked { background-color: #2a6a9a; border-color: #4a8aca; }
        QPushButton:disabled { color: #666; }
        QProgressBar { border: 1px solid #555; border-radius: 3px;
                       text-align: center; background-color: #2a2a2a; }
        QProgressBar::chunk { border-radius: 2px; }
        QListWidget { background-color: #2a2a2a; border: 1px solid #444; }
        QStatusBar { border-top: 1px solid #444; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
