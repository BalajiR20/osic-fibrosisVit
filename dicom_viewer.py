"""
╔══════════════════════════════════════════╗
║     DICOM Viewer — Pulmonary Fibrosis    ║
║     3-Plane MPR  ·  View Switcher        ║
╚══════════════════════════════════════════╝

Requirements:
    pip install pydicom PyQt5 numpy

Run:
    python dicom_viewer.py
"""

import sys
import os
import numpy as np

import pydicom

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QListWidget,
    QSplitter, QFrame, QGroupBox, QGridLayout,
    QSizePolicy, QProgressBar, QTabWidget, QTextEdit,
    QSpinBox, QCheckBox, QStatusBar,
    QToolBar, QAction, QMessageBox, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QPalette, QWheelEvent, QMouseEvent
)

# ─── Colour palette ───────────────────────────────────────────────────────────
DARK_BG      = "#0A0E17"
PANEL_BG     = "#111827"
CARD_BG      = "#1A2233"
ACCENT       = "#00D4FF"
ACCENT2      = "#7C3AED"
TEXT_PRIMARY = "#E8EFF9"
TEXT_MUTED   = "#6B7B99"
BORDER       = "#1E2D45"
WARNING      = "#F59E0B"
DANGER       = "#EF4444"

C_AXIAL    = "#FFD700"
C_CORONAL  = "#FF6EB4"
C_SAGITTAL = "#00BFFF"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}}
QSplitter::handle {{
    background: {BORDER};
    width: 2px;
    height: 2px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    font-weight: 600;
    color: {ACCENT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QLabel {{
    color: {TEXT_PRIMARY};
}}
QPushButton {{
    background-color: {CARD_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: {DARK_BG};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: #009BBF;
}}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {ACCENT}, stop:1 {ACCENT2});
    color: white;
    border: none;
    font-weight: 700;
    font-size: 13px;
}}
/* View switcher buttons */
QPushButton#vs_axial {{
    background-color: {CARD_BG};
    color: {C_AXIAL};
    border: 1px solid {C_AXIAL};
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton#vs_axial:hover, QPushButton#vs_axial:checked {{
    background-color: {C_AXIAL};
    color: {DARK_BG};
}}
QPushButton#vs_coronal {{
    background-color: {CARD_BG};
    color: {C_CORONAL};
    border: 1px solid {C_CORONAL};
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton#vs_coronal:hover, QPushButton#vs_coronal:checked {{
    background-color: {C_CORONAL};
    color: {DARK_BG};
}}
QPushButton#vs_sagittal {{
    background-color: {CARD_BG};
    color: {C_SAGITTAL};
    border: 1px solid {C_SAGITTAL};
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton#vs_sagittal:hover, QPushButton#vs_sagittal:checked {{
    background-color: {C_SAGITTAL};
    color: {DARK_BG};
}}
QPushButton#vs_all {{
    background-color: {CARD_BG};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 600;
}}
QPushButton#vs_all:hover, QPushButton#vs_all:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {C_AXIAL}, stop:0.5 {C_CORONAL}, stop:1 {C_SAGITTAL});
    color: {DARK_BG};
    border: none;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QListWidget {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
    outline: none;
}}
QListWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {DARK_BG};
    border-radius: 4px;
}}
QListWidget::item:hover {{
    background-color: {CARD_BG};
}}
QScrollBar:vertical {{
    background: {PANEL_BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar:horizontal {{
    background: {PANEL_BG};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}
QTextEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_MUTED};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    padding: 6px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {PANEL_BG};
}}
QTabBar::tab {{
    background: {CARD_BG};
    color: {TEXT_MUTED};
    padding: 8px 16px;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QStatusBar {{
    background: {PANEL_BG};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
QProgressBar {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 8px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}
QSpinBox {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {CARD_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QToolBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 8px;
    color: {TEXT_PRIMARY};
}}
QToolBar QToolButton:hover {{
    background: {CARD_BG};
    border-color: {BORDER};
}}
"""

# ─── DICOM Loader Thread ──────────────────────────────────────────────────────
class DicomLoader(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)   # volume (Z,Y,X), last ds

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        files = []
        for root, _, fnames in os.walk(self.folder):
            for f in fnames:
                files.append(os.path.join(root, f))

        slices = []; last_ds = None
        for i, path in enumerate(files, 1):
            self.progress.emit(i, len(files))
            try:
                ds = pydicom.dcmread(path, force=True)
                if not hasattr(ds, 'pixel_array'):
                    continue
                arr = ds.pixel_array.astype(np.float32)
                slope     = float(getattr(ds, 'RescaleSlope',     1))
                intercept = float(getattr(ds, 'RescaleIntercept', 0))
                arr = arr * slope + intercept
                z   = float(getattr(ds, 'InstanceNumber', i))
                slices.append((z, arr))
                last_ds = ds
            except Exception:
                pass

        if not slices:
            self.finished.emit(None, None)
            return

        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0)
        self.finished.emit(volume, last_ds)


# ─── Single Plane Canvas ──────────────────────────────────────────────────────
class PlaneCanvas(QLabel):
    """One MPR plane with zoom, pan, crosshair and HU readout."""

    clicked = pyqtSignal(float, float)        # normalised x, y
    hovered = pyqtSignal(str, int, int, float) # plane, x, y, HU

    COLOURS = {
        "Axial":    C_AXIAL,
        "Coronal":  C_CORONAL,
        "Sagittal": C_SAGITTAL,
    }
    ORIENT = {
        "Axial":    ("A", "P", "R", "L"),
        "Coronal":  ("S", "I", "R", "L"),
        "Sagittal": ("S", "I", "A", "P"),
    }

    def __init__(self, plane):
        super().__init__()
        self.plane    = plane
        self._arr     = None
        self._zoom    = 1.0
        self._pan     = [0, 0]
        self._drag    = None
        self._cross   = (0.5, 0.5)
        self._ww      = 1500
        self._wl      = -600
        self._show_cross = True
        self._pm_orig = None
        col = self.COLOURS[plane]
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:#000; border: 2px solid {col}; border-radius: 8px;")
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_slice(self, arr2d):
        self._arr = arr2d
        self._render()

    def set_window(self, ww, wl):
        self._ww, self._wl = ww, wl
        self._render()

    def set_cross(self, nx, ny):
        self._cross = (nx, ny)
        self._draw()

    def toggle_crosshair(self, on):
        self._show_cross = on
        self._draw()

    def reset_view(self):
        self._zoom = 1.0
        self._pan  = [0, 0]
        self._draw()

    def zoom_step(self, factor):
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        self._draw()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self):
        if self._arr is None:
            return
        lo  = self._wl - self._ww / 2
        hi  = self._wl + self._ww / 2
        a8  = np.clip((self._arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        h, w = a8.shape
        self._pm_orig = QPixmap.fromImage(
            QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        )
        self._draw()

    def _draw(self):
        if self._pm_orig is None:
            return
        sw, sh = self.width(), self.height()
        pw, ph = self._pm_orig.width(), self._pm_orig.height()
        if sw < 1 or sh < 1 or pw < 1 or ph < 1:
            return

        scale  = min(sw / pw, sh / ph) * self._zoom
        nw, nh = int(pw * scale), int(ph * scale)
        scaled = self._pm_orig.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        result = QPixmap(sw, sh)
        result.fill(QColor("#000"))
        p = QPainter(result)

        ox = (sw - nw) // 2 + self._pan[0]
        oy = (sh - nh) // 2 + self._pan[1]
        p.drawPixmap(ox, oy, scaled)

        col = self.COLOURS[self.plane]

        # crosshair
        if self._show_cross:
            pen = QPen(QColor(col), 1, Qt.SolidLine)
            p.setPen(pen)
            cx = int(ox + self._cross[0] * nw)
            cy = int(oy + self._cross[1] * nh)
            p.drawLine(cx, 0, cx, sh)
            p.drawLine(0, cy, sw, cy)

        # WL/WW overlay
        p.setPen(QPen(QColor(col)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(8, 16, f"WL:{self._wl}  WW:{self._ww}")
        p.drawText(8, sh - 8, f"Zoom:{self._zoom:.2f}x")

        # orientation labels
        top, bot, left_, right_ = self.ORIENT[self.plane]
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(sw // 2 - 5, 18, top)
        p.drawText(sw // 2 - 5, sh - 6, bot)
        p.drawText(5, sh // 2 + 5, left_)
        p.drawText(sw - 14, sh // 2 + 5, right_)

        p.end()
        self.setPixmap(result)
        self._ox, self._oy, self._scale = ox, oy, scale
        self._nw, self._nh = nw, nh

    def resizeEvent(self, e):
        self._draw()

    # ── mouse ─────────────────────────────────────────────────────────────────
    def wheelEvent(self, e: QWheelEvent):
        self._zoom = max(0.1, min(self._zoom * (1.1 if e.angleDelta().y() > 0 else 0.9), 20.0))
        self._draw()

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._drag = (e.x(), e.y())
            self._emit_click(e.x(), e.y())
        elif e.button() == Qt.MiddleButton:
            self._zoom = 1.0; self._pan = [0, 0]; self._draw()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._drag:
            dx = e.x() - self._drag[0]; dy = e.y() - self._drag[1]
            self._pan[0] += dx; self._pan[1] += dy
            self._drag = (e.x(), e.y())
            self._draw()
        # HU readout
        if self._arr is not None and hasattr(self, '_scale') and self._scale:
            ix = int((e.x() - self._ox) / self._scale)
            iy = int((e.y() - self._oy) / self._scale)
            h, w = self._arr.shape
            if 0 <= ix < w and 0 <= iy < h:
                self.hovered.emit(self.plane, ix, iy, float(self._arr[iy, ix]))

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self._zoom = 1.0; self._pan = [0, 0]; self._draw()

    def _emit_click(self, mx, my):
        if not hasattr(self, '_ox'):
            return
        nx = max(0.0, min(1.0, (mx - self._ox) / self._nw))
        ny = max(0.0, min(1.0, (my - self._oy) / self._nh))
        self.clicked.emit(nx, ny)


# ─── Histogram Widget ─────────────────────────────────────────────────────────
class HistogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)
        self._hist = self._edges = None
        self._wl = 0; self._ww = 1

    def set_data(self, arr, wl, ww):
        self._hist, self._edges = np.histogram(arr.flatten(), bins=256)
        self._wl = wl; self._ww = ww
        self.update()

    def paintEvent(self, e):
        if self._hist is None:
            return
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL_BG))
        hist_max = max(self._hist) or 1
        bar_w    = w / len(self._hist)
        lo = self._wl - self._ww / 2
        hi = self._wl + self._ww / 2
        emin, emax = self._edges[0], self._edges[-1]
        span = emax - emin or 1
        for i, v in enumerate(self._hist):
            bh  = int((v / hist_max) * (h - 8))
            x   = int(i * bar_w)
            val = self._edges[i]
            col = QColor(ACCENT if lo <= val <= hi else BORDER)
            col.setAlpha(180)
            p.fillRect(x, h - bh, max(1, int(bar_w)), bh, col)
        pen = QPen(QColor(WARNING), 1)
        p.setPen(pen)
        lx = int((lo - emin) / span * w)
        hx = int((hi - emin) / span * w)
        p.drawLine(lx, 0, lx, h)
        p.drawLine(hx, 0, hx, h)
        p.end()


# ─── Main Window ──────────────────────────────────────────────────────────────
class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer · Pulmonary Fibrosis · MPR")
        self.resize(1500, 900)
        self.setStyleSheet(STYLESHEET)

        self._vol     = None   # (Z, Y, X) HU
        self._zi = self._yi = self._xi = 0
        self._ww = 1500
        self._wl = -600
        self._last_ds = None
        self._current_view = "all"   # "all" | "axial" | "coronal" | "sagittal"

        self._build_ui()
        self._build_toolbar()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(280)
        left.setMinimumWidth(220)
        left.setStyleSheet(f"background:{PANEL_BG}; border-right:1px solid {BORDER};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(10)

        title = QLabel("⬡ DICOM Viewer")
        title.setStyleSheet(f"font-size:18px; font-weight:800; color:{ACCENT}; letter-spacing:1px;")
        lv.addWidget(title)

        sub = QLabel("Pulmonary Fibrosis Analysis")
        sub.setStyleSheet(f"font-size:11px; color:{TEXT_MUTED};")
        lv.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        lv.addWidget(sep)

        btn_folder = QPushButton("📂  Open DICOM Folder")
        btn_folder.setObjectName("primary")
        btn_folder.clicked.connect(self._open_folder)
        lv.addWidget(btn_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lv.addWidget(self.progress_bar)

        self.load_label = QLabel("")
        self.load_label.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED};")
        self.load_label.setWordWrap(True)
        lv.addWidget(self.load_label)

        # Slice list (axial only — acts as series navigator)
        series_grp = QGroupBox("Slices")
        sg = QVBoxLayout(series_grp)
        self.series_list = QListWidget()
        self.series_list.currentRowChanged.connect(self._on_list_select)
        sg.addWidget(self.series_list)
        lv.addWidget(series_grp, stretch=1)

        splitter.addWidget(left)

        # ── CENTRE: view switcher + 3 canvases + histogram ────────────────────
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(8, 8, 8, 8)
        cv.setSpacing(6)

        # Info bar
        info_row = QHBoxLayout()
        self.lbl_file  = QLabel("No file loaded")
        self.lbl_file.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        self.lbl_pixel = QLabel("")
        self.lbl_pixel.setStyleSheet(f"color:{ACCENT}; font-size:11px; font-weight:600;")
        info_row.addWidget(self.lbl_file)
        info_row.addStretch()
        info_row.addWidget(self.lbl_pixel)
        cv.addLayout(info_row)

        # ── View Switcher ─────────────────────────────────────────────────────
        vs_row = QHBoxLayout()
        vs_row.setSpacing(6)

        vs_lbl = QLabel("View:")
        vs_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
        vs_row.addWidget(vs_lbl)

        # Each button is checkable; we manage exclusivity ourselves
        self._vs_buttons = {}
        view_defs = [
            ("axial",    f"● Axial",    "vs_axial"),
            ("coronal",  "● Coronal",  "vs_coronal"),
            ("sagittal", "● Sagittal", "vs_sagittal"),
            ("all",      "⊞ All 3",    "vs_all"),
        ]
        for view_id, label, obj_name in view_defs:
            btn = QPushButton(label)
            btn.setObjectName(obj_name)
            btn.setCheckable(True)
            btn.setChecked(view_id == "all")
            btn.clicked.connect(lambda checked, v=view_id: self._set_view(v))
            vs_row.addWidget(btn)
            self._vs_buttons[view_id] = btn

        vs_row.addStretch()
        cv.addLayout(vs_row)

        # ── Three canvases (QSplitter keeps equal widths on restore) ─────────
        self._canvas_splitter = QSplitter(Qt.Horizontal)
        self._canvas_splitter.setHandleWidth(4)
        self._canvas_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {BORDER}; border-radius: 2px; }}"
        )

        self.axial    = PlaneCanvas("Axial")
        self.coronal  = PlaneCanvas("Coronal")
        self.sagittal = PlaneCanvas("Sagittal")

        self._canvas_wraps = {}
        for canvas, name, col in [
            (self.axial,    "Axial",    C_AXIAL),
            (self.coronal,  "Coronal",  C_CORONAL),
            (self.sagittal, "Sagittal", C_SAGITTAL),
        ]:
            wrap = QWidget()
            wl2  = QVBoxLayout(wrap)
            wl2.setContentsMargins(0, 0, 0, 0)
            wl2.setSpacing(3)
            lbl = QLabel(f"● {name}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"font-size:13px; font-weight:700; color:{col};"
                f"background:{CARD_BG}; border:1px solid {BORDER};"
                f"border-radius:5px; padding:4px;"
            )
            wl2.addWidget(lbl)
            wl2.addWidget(canvas)
            self._canvas_splitter.addWidget(wrap)
            self._canvas_wraps[name.lower()] = wrap

        # All panels equal stretch by default
        self._canvas_splitter.setStretchFactor(0, 1)
        self._canvas_splitter.setStretchFactor(1, 1)
        self._canvas_splitter.setStretchFactor(2, 1)

        cv.addWidget(self._canvas_splitter, stretch=1)

        # Axial slice slider (below canvases)
        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("🗂  Axial Slice:"))
        self.sld_slice = QSlider(Qt.Horizontal)
        self.sld_slice.setMinimum(0); self.sld_slice.setValue(0)
        self.sld_slice.valueChanged.connect(self._on_z_slider)
        self.lbl_slice = QLabel("0")
        self.lbl_slice_total = QLabel("/ 0")
        self.lbl_slice_total.setStyleSheet(f"color:{TEXT_MUTED};")
        slice_row.addWidget(self.sld_slice, stretch=1)
        slice_row.addWidget(self.lbl_slice)
        slice_row.addWidget(self.lbl_slice_total)
        cv.addLayout(slice_row)

        # Histogram (of current axial slice)
        hist_grp = QGroupBox("Histogram  (Axial slice)")
        hg = QVBoxLayout(hist_grp)
        self.histogram = HistogramWidget()
        hg.addWidget(self.histogram)
        cv.addWidget(hist_grp)

        splitter.addWidget(centre)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        right = QWidget()
        right.setMaximumWidth(280)
        right.setMinimumWidth(220)
        right.setStyleSheet(f"background:{PANEL_BG}; border-left:1px solid {BORDER};")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 10)
        rv.setSpacing(10)

        tabs = QTabWidget()
        rv.addWidget(tabs)

        # — Window tab —
        wt = QWidget()
        wv = QVBoxLayout(wt)
        wv.setSpacing(8)

        ww_grp = QGroupBox("Window Width / Level")
        wg = QGridLayout(ww_grp)

        wg.addWidget(QLabel("Width (WW):"), 0, 0)
        self.sld_ww = QSlider(Qt.Horizontal)
        self.sld_ww.setRange(1, 4000); self.sld_ww.setValue(1500)
        self.sld_ww.valueChanged.connect(self._on_window_change)
        self.spin_ww = QSpinBox(); self.spin_ww.setRange(1, 8000); self.spin_ww.setValue(1500)
        self.spin_ww.valueChanged.connect(lambda v: (self.sld_ww.setValue(v), self._on_window_change()))
        wg.addWidget(self.sld_ww, 0, 1)
        wg.addWidget(self.spin_ww, 0, 2)

        wg.addWidget(QLabel("Level (WL):"), 1, 0)
        self.sld_wl = QSlider(Qt.Horizontal)
        self.sld_wl.setRange(-2000, 2000); self.sld_wl.setValue(-600)
        self.sld_wl.valueChanged.connect(self._on_window_change)
        self.spin_wl = QSpinBox(); self.spin_wl.setRange(-4000, 4000); self.spin_wl.setValue(-600)
        self.spin_wl.valueChanged.connect(lambda v: (self.sld_wl.setValue(v), self._on_window_change()))
        wg.addWidget(self.sld_wl, 1, 1)
        wg.addWidget(self.spin_wl, 1, 2)

        wv.addWidget(ww_grp)

        # Window presets
        presets_grp = QGroupBox("Presets")
        pg = QVBoxLayout(presets_grp)
        preset_data = [
            ("🫁  Lung          WL -600 / WW 1500",  -600, 1500),
            ("🫀  Mediastinum   WL   40 / WW  400",    40,  400),
            ("🦴  Bone          WL  400 / WW 1800",   400, 1800),
            ("🧠  Brain         WL   40 / WW   80",    40,   80),
        ]
        for label, wl, ww in preset_data:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=wl, w=ww: self._apply_preset(l, w))
            pg.addWidget(btn)
        wv.addWidget(presets_grp)

        chk_cross = QCheckBox("Show Crosshair")
        chk_cross.setChecked(True)
        chk_cross.toggled.connect(self._toggle_crosshair)
        wv.addWidget(chk_cross)

        btn_invert = QPushButton("🔄  Invert LUT")
        btn_invert.clicked.connect(self._invert_lut)
        wv.addWidget(btn_invert)

        wv.addStretch()
        tabs.addTab(wt, "Window")

        # — Metadata tab —
        mt = QWidget()
        mv = QVBoxLayout(mt)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        mv.addWidget(self.meta_text)
        tabs.addTab(mt, "Metadata")

        # — Help tab —
        it = QWidget()
        iv = QVBoxLayout(it)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText(
            "DICOM Viewer Controls\n"
            "─────────────────────\n"
            "🖱  Scroll wheel     → Zoom\n"
            "🖱  Left drag        → Pan\n"
            "🖱  Click            → Move crosshair\n"
            "🖱  Double-click     → Reset view\n"
            "🖱  Middle-click     → Reset view\n\n"
            "Keyboard\n"
            "─────────────────────\n"
            "← / →   Previous / Next axial slice\n"
            "+ / -   Window width ±50\n"
            "W / S   Window level ±50\n"
            "F       Fit / reset all views\n"
            "I       Invert LUT\n"
            "1       Show Axial only\n"
            "2       Show Coronal only\n"
            "3       Show Sagittal only\n"
            "4       Show All 3 planes\n\n"
            "View Switcher\n"
            "─────────────────────\n"
            "Use the coloured buttons above the\n"
            "canvases to switch between single-plane\n"
            "focus and the full 3-plane MPR layout.\n\n"
            "Views\n"
            "─────────────────────\n"
            f"🟡 Axial    — transverse plane\n"
            f"🩷 Coronal  — front view\n"
            f"🔵 Sagittal — side view\n\n"
            "Click any view to reposition\n"
            "the crosshair across all 3 planes."
        )
        iv.addWidget(self.info_text)
        tabs.addTab(it, "Help")

        splitter.addWidget(right)
        splitter.setSizes([240, 980, 260])

        # canvas signals
        self.axial.clicked.connect(self._axial_clicked)
        self.coronal.clicked.connect(self._coronal_clicked)
        self.sagittal.clicked.connect(self._sagittal_clicked)
        for c in [self.axial, self.coronal, self.sagittal]:
            c.hovered.connect(self._on_hover)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — Open a DICOM folder to begin.")

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        acts = [
            ("📂 Open Folder", self._open_folder),
            (None, None),
            ("◀ Previous",    self._prev_slice),
            ("▶ Next",        self._next_slice),
            (None, None),
            ("🔍 Zoom In",    lambda: [c.zoom_step(1.2) for c in [self.axial, self.coronal, self.sagittal]]),
            ("🔍 Zoom Out",   lambda: [c.zoom_step(0.8) for c in [self.axial, self.coronal, self.sagittal]]),
            ("⊡ Reset Views", self._reset_all),
            (None, None),
            ("1 Axial",       lambda: self._set_view("axial")),
            ("2 Coronal",     lambda: self._set_view("coronal")),
            ("3 Sagittal",    lambda: self._set_view("sagittal")),
            ("4 All 3",       lambda: self._set_view("all")),
            (None, None),
            ("💾 Screenshot", self._screenshot),
        ]
        for label, fn in acts:
            if label is None:
                tb.addSeparator()
            else:
                a = QAction(label, self); a.triggered.connect(fn); tb.addAction(a)

    # ── View Switcher ─────────────────────────────────────────────────────────
    def _set_view(self, view: str):
        """
        Show one plane full-width or all three side by side.
        view: "axial" | "coronal" | "sagittal" | "all"
        """
        self._current_view = view

        # Update button checked states (exclusive)
        for vid, btn in self._vs_buttons.items():
            btn.setChecked(vid == view)

        # Show/hide canvas wrappers via the splitter
        order   = ["axial", "coronal", "sagittal"]
        visible = order if view == "all" else [view]

        for plane_name, wrap in self._canvas_wraps.items():
            wrap.setVisible(plane_name in visible)

        # When restoring all 3, force equal sizes so no panel is
        # oversized from the previous single-plane state.
        if view == "all":
            total = self._canvas_splitter.width()
            equal = max(1, total // 3)
            self._canvas_splitter.setSizes([equal, equal, equal])

        # Trigger a redraw on all now-visible canvases so they
        # fill their new geometry immediately.
        for canvas in [self.axial, self.coronal, self.sagittal]:
            canvas._draw()

        names = {
            "axial":    "Axial — transverse plane",
            "coronal":  "Coronal — front view",
            "sagittal": "Sagittal — side view",
            "all":      "All 3 planes — MPR view",
        }
        self.status.showMessage(f"View: {names.get(view, view)}")

    # ── Loading ───────────────────────────────────────────────────────────────
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if not folder:
            return
        self.series_list.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.load_label.setText("Scanning…")
        self.loader = DicomLoader(folder)
        self.loader.progress.connect(lambda c, t: self.progress_bar.setValue(int(c / t * 100)))
        self.loader.finished.connect(self._on_load_done)
        self.loader.start()

    def _on_load_done(self, vol, ds):
        self.progress_bar.setVisible(False)
        if vol is None:
            self.load_label.setText("❌ No valid DICOM files found.")
            return
        self._vol     = vol
        self._last_ds = ds
        Z, Y, X = vol.shape
        self.load_label.setText(f"✅  {Z} slices  ·  {Y}×{X} px")

        # Populate slice list
        self.series_list.blockSignals(True)
        self.series_list.clear()
        for i in range(Z):
            self.series_list.addItem(f"Slice {i+1:04d}")
        self.series_list.blockSignals(False)

        self.sld_slice.setMaximum(Z - 1)
        self.lbl_slice_total.setText(f"/ {Z}")

        self._zi = Z // 2; self._yi = Y // 2; self._xi = X // 2
        self.sld_slice.setValue(self._zi)

        self._update_all()
        self._load_meta(ds)
        self.status.showMessage(f"Loaded volume  {Z}×{Y}×{X}  |  WL={self._wl}  WW={self._ww}")

    def _load_meta(self, ds):
        if ds is None: return
        fields = [
            ('PatientName','Patient'), ('PatientID','Patient ID'),
            ('PatientBirthDate','DOB'), ('StudyDate','Study Date'),
            ('Modality','Modality'), ('StudyDescription','Study Desc'),
            ('SeriesDescription','Series Desc'), ('InstanceNumber','Instance #'),
            ('SliceThickness','Slice Thick'), ('SpacingBetweenSlices','Slice Spacing'),
            ('PixelSpacing','Pixel Spacing'), ('Rows','Rows'), ('Columns','Columns'),
            ('BitsAllocated','Bits'), ('KVP','kVp'),
            ('Manufacturer','Manufacturer'), ('InstitutionName','Institution'),
        ]
        lines = []
        for tag, name in fields:
            try: lines.append(f"{name:<18}: {getattr(ds, tag)}")
            except Exception: pass
        self.meta_text.setPlainText('\n'.join(lines))

    # ── Slice updates ─────────────────────────────────────────────────────────
    def _on_z_slider(self, v):
        self._zi = v
        self.lbl_slice.setText(str(v + 1))
        self.series_list.blockSignals(True)
        self.series_list.setCurrentRow(v)
        self.series_list.blockSignals(False)
        if self._vol is not None:
            self.axial.set_slice(self._vol[v, :, :])
            self.histogram.set_data(self._vol[v, :, :], self._wl, self._ww)
            self._update_cross()

    def _on_list_select(self, row):
        if self._vol is not None and 0 <= row < self._vol.shape[0]:
            self.sld_slice.setValue(row)

    def _update_all(self):
        if self._vol is None: return
        self.axial.set_slice(self._vol[self._zi, :, :])
        self.coronal.set_slice(self._vol[:, self._yi, :])
        self.sagittal.set_slice(self._vol[:, :, self._xi])
        self.histogram.set_data(self._vol[self._zi, :, :], self._wl, self._ww)
        self._update_cross()
        try:
            if self._last_ds:
                patient = str(getattr(self._last_ds, 'PatientName', 'Unknown'))
                desc    = str(getattr(self._last_ds, 'SeriesDescription', ''))
                self.lbl_file.setText(f"👤 {patient}   📋 {desc}")
        except Exception:
            pass

    def _update_cross(self):
        if self._vol is None: return
        Z, Y, X = self._vol.shape
        self.axial.set_cross(self._xi / X, self._yi / Y)
        self.coronal.set_cross(self._xi / X, self._zi / Z)
        self.sagittal.set_cross(self._yi / Y, self._zi / Z)

    # ── Canvas clicks → reposition crosshair ─────────────────────────────────
    def _axial_clicked(self, nx, ny):
        if self._vol is None: return
        Z, Y, X = self._vol.shape
        new_x = int(nx * X); new_y = int(ny * Y)
        self._xi = new_x; self._yi = new_y
        self.sagittal.set_slice(self._vol[:, :, new_x])
        self.coronal.set_slice(self._vol[:, new_y, :])
        self._update_cross()

    def _coronal_clicked(self, nx, ny):
        if self._vol is None: return
        Z, Y, X = self._vol.shape
        new_x = int(nx * X); new_z = int(ny * Z)
        self._xi = new_x; self._zi = new_z
        self.sagittal.set_slice(self._vol[:, :, new_x])
        self.sld_slice.setValue(new_z)
        self._update_cross()

    def _sagittal_clicked(self, nx, ny):
        if self._vol is None: return
        Z, Y, X = self._vol.shape
        new_y = int(nx * Y); new_z = int(ny * Z)
        self._yi = new_y; self._zi = new_z
        self.coronal.set_slice(self._vol[:, new_y, :])
        self.sld_slice.setValue(new_z)
        self._update_cross()

    def _on_hover(self, plane, x, y, hu):
        self.lbl_pixel.setText(f"{plane}  x={x} y={y}  HU={hu:.1f}")

    # ── Window ────────────────────────────────────────────────────────────────
    def _on_window_change(self):
        self._ww = self.sld_ww.value()
        self._wl = self.sld_wl.value()
        self.spin_ww.blockSignals(True); self.spin_ww.setValue(self._ww); self.spin_ww.blockSignals(False)
        self.spin_wl.blockSignals(True); self.spin_wl.setValue(self._wl); self.spin_wl.blockSignals(False)
        for c in [self.axial, self.coronal, self.sagittal]:
            c.set_window(self._ww, self._wl)
        if self._vol is not None:
            self.histogram.set_data(self._vol[self._zi, :, :], self._wl, self._ww)

    def _apply_preset(self, wl, ww):
        self.sld_wl.setValue(wl); self.sld_ww.setValue(ww)
        self.status.showMessage(f"Preset applied  WL={wl}  WW={ww}")

    def _invert_lut(self):
        self.sld_wl.setValue(-self.sld_wl.value())
        self.status.showMessage("LUT inverted")

    def _toggle_crosshair(self, on):
        for c in [self.axial, self.coronal, self.sagittal]:
            c.toggle_crosshair(on)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _prev_slice(self):
        self.sld_slice.setValue(max(0, self.sld_slice.value() - 1))

    def _next_slice(self):
        self.sld_slice.setValue(min(self.sld_slice.maximum(), self.sld_slice.value() + 1))

    def _reset_all(self):
        # 1. Reset zoom & pan on every canvas
        for c in [self.axial, self.coronal, self.sagittal]:
            c.reset_view()

        # 2. Switch back to All-3 layout with equal panel sizes
        self._set_view("all")

        # 3. Re-centre the crosshair to the middle of the volume
        if self._vol is not None:
            Z, Y, X = self._vol.shape
            self._zi = Z // 2
            self._yi = Y // 2
            self._xi = X // 2
            self.sld_slice.setValue(self._zi)
            self._update_all()

        self.status.showMessage("Reset — all views restored to default.")

    # ── Screenshot ────────────────────────────────────────────────────────────
    def _screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "dicom_screenshot.png",
            "PNG Files (*.png);;JPEG (*.jpg)"
        )
        if path:
            screen = self.grab()
            screen.save(path)
            self.status.showMessage(f"Screenshot saved: {path}")

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Left, Qt.Key_Up):
            self._prev_slice()
        elif k in (Qt.Key_Right, Qt.Key_Down):
            self._next_slice()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.sld_ww.setValue(self.sld_ww.value() + 50)
        elif k == Qt.Key_Minus:
            self.sld_ww.setValue(max(1, self.sld_ww.value() - 50))
        elif k == Qt.Key_W:
            self.sld_wl.setValue(self.sld_wl.value() + 50)
        elif k == Qt.Key_S:
            self.sld_wl.setValue(self.sld_wl.value() - 50)
        elif k == Qt.Key_F:
            self._reset_all()
        elif k == Qt.Key_I:
            self._invert_lut()
        elif k == Qt.Key_1:
            self._set_view("axial")
        elif k == Qt.Key_2:
            self._set_view("coronal")
        elif k == Qt.Key_3:
            self._set_view("sagittal")
        elif k == Qt.Key_4:
            self._set_view("all")
        else:
            super().keyPressEvent(e)


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DICOM Viewer")
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(DARK_BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base,            QColor(PANEL_BG))
    palette.setColor(QPalette.AlternateBase,   QColor(CARD_BG))
    palette.setColor(QPalette.Text,            QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button,          QColor(CARD_BG))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(DARK_BG))
    app.setPalette(palette)

    win = DicomViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()