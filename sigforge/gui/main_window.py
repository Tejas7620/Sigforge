"""
SigForge Main Window — PyQt6 desktop GUI with dock-based layout.

Layout:
  Left:   File input, metadata, GNI options, parameter overrides, pipeline stages
  Center: Tabbed visualization (Spectrum, Waterfall, Constellation, Eye, Bitstream, Frames, GNI)
  Right:  Signal metrics gauges, hypothesis ranking, evidence breakdown, warnings
  Bottom: Log viewer, progress, elapsed time

GNI Framework:
  Guided, Normal, and Informed (GNI) analysis modes allow the user to
  specify different levels of prior knowledge, guiding the hypothesis loop.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

from PyQt6.QtCore import Qt, QSize, QTimer, QRect, QRectF
from PyQt6.QtGui import (
    QAction, QFont, QKeySequence,
    QDragEnterEvent, QDropEvent, QColor, QPainter, QBrush,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QToolBar, QStatusBar, QProgressBar,
    QFileDialog, QTextEdit, QComboBox, QGroupBox, QFormLayout, QLineEdit,
    QSizePolicy, QFrame, QScrollArea, QCheckBox, QSplitter,
)

import pyqtgraph as pg

from sigforge.gui.styles import get_stylesheet, COLORS
from sigforge.gui.worker import AnalysisWorker
from sigforge.gui.live_worker import LiveStreamWorker
from sigforge.io.gnuradio_source import GNURadioMockServer
from sigforge.pipeline.data_types import PipelineResult



# ── pyqtgraph global config ────────────────────────────────────────────────────
pg.setConfigOptions(
    background=COLORS["bg_primary"],
    foreground=COLORS["text_primary"],
    antialias=True,
)

# ── GNI mode definitions ───────────────────────────────────────────────────────
GNI_MODES = {
    "Auto (GNI)": {
        "description": "Fully automated blind analysis — no prior knowledge required.",
        "color": COLORS["accent"],
        "overrides": False,
    },
    "Guided": {
        "description": "Provide optional hints (sample rate, modulation) to speed up analysis.",
        "color": COLORS["success"],
        "overrides": True,
    },
    "Expert": {
        "description": "Full manual control — override all parameters for expert analysis.",
        "color": COLORS["warning"],
        "overrides": True,
    },
}

PIPELINE_STAGES = [
    ("ingest",     "1. File Ingestion"),
    ("preprocess", "2. Preprocessing"),
    ("analysis",   "3. Signal Analysis"),
    ("param_est",  "4. Parameter Estimation"),
    ("classify",   "5. Modulation Classification"),
    ("sync",       "6. Synchronization"),
    ("demod",      "7. Demodulation"),
    ("bits",       "8. Bit Mapping"),
    ("interleave", "9. Interleaver Search"),
    ("fec",        "10. FEC Search"),
    ("frame",      "11. Frame Analysis"),
    ("validate",   "12. Validation"),
    ("hypothesis", "13. Hypothesis Fusion"),
]


def _safe(obj, *attrs, default=None):
    """Safely traverse a chain of attributes, returning default on any miss."""
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
            if obj is None:
                return default
        except AttributeError:
            return default
    return obj


def _safe_val(val, fmt="{}", default="—"):
    """Format a value safely; returns default if value is None."""
    if val is None:
        return default
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


# ── Animated spinner label ─────────────────────────────────────────────────────
class SpinnerLabel(QLabel):
    """A rotating UTF-8 spinner for running pipeline stages."""

    FRAMES = ["◐", "◓", "◑", "◒"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setText("○")

    def start(self):
        self._timer.start(120)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self.setText(self.FRAMES[self._frame])


# ── SNR / Confidence bar ───────────────────────────────────────────────────────
class MiniBarWidget(QWidget):
    """A compact horizontal gradient bar for metrics display."""

    def __init__(self, color="#6366f1", parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._color = QColor(color)
        self.setFixedHeight(8)
        self.setMinimumWidth(80)

    def set_value(self, v: float):
        self._value = max(0.0, min(1.0, float(v)))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(COLORS["bg_tertiary"]))
        p.drawRoundedRect(r, 4, 4)
        fill_w = int(r.width() * self._value)
        if fill_w > 0:
            grad = QLinearGradient(0, 0, r.width(), 0)
            grad.setColorAt(0, self._color.darker(130))
            grad.setColorAt(1, self._color)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRect(r.left(), r.top(), fill_w, r.height()), 4, 4)
        p.end()


# ── Drop zone widget ───────────────────────────────────────────────────────────
from PyQt6.QtCore import pyqtSignal


class DropZoneWidget(QWidget):
    """Centered drag-and-drop landing page shown before a file is loaded."""

    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("📡")
        icon.setStyleSheet("font-size: 64px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drop a Signal File Here")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 22px; "
            "font-weight: 700; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel(
            "Supported formats:   .bin  .raw  .cf32  .cs16  .cu8  .cs8  .wav  .sigmf-data  .iq"
        )
        sub.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;"
        )
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addSpacing(12)
        layout.addWidget(title)
        layout.addSpacing(6)
        layout.addWidget(sub)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)
                event.acceptProposedAction()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    """SigForge main application window with GNI framework integration."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SigForge — Automated Blind Signal Analysis & Decoding Suite")
        self.setMinimumSize(1400, 850)
        self.resize(1750, 1020)
        self.setAcceptDrops(True)
        self.setStyleSheet(get_stylesheet())

        self._current_file: Optional[str] = None
        self._current_mode: str = "Auto (GNI)"
        self._last_result: Optional[PipelineResult] = None
        self._worker: Optional[AnalysisWorker] = None
        self._analysis_start: float = 0.0

        self._live_worker: Optional[LiveStreamWorker] = None
        self._mock_server: Optional[GNURadioMockServer] = None
        self._live_waterfall_buffer: Optional[np.ndarray] = None


        self._create_toolbar()
        self._create_center_panel()
        self._create_left_dock()
        self._create_right_dock()
        self._create_bottom_dock()
        self._create_status_bar()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)

    # ── Toolbar ──────────────────────────────────────────────────────────────
    def _create_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setFixedHeight(52)
        self.addToolBar(tb)

        brand = QLabel("  🛰  <b>SigForge</b>")
        brand.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 16px; background: transparent;"
        )
        tb.addWidget(brand)
        tb.addSeparator()

        mode_lbl = QLabel("  GNI Mode: ")
        mode_lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: 600; background: transparent;"
        )
        tb.addWidget(mode_lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(GNI_MODES.keys()))
        self.mode_combo.setFixedWidth(140)
        self.mode_combo.setToolTip(
            "GNI = Guided / Normal / Informed analysis framework\n"
            "• Auto (GNI): fully blind\n• Guided: hints\n• Expert: full overrides"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb.addWidget(self.mode_combo)

        self.mode_badge = QLabel("Fully automated blind analysis")
        self.mode_badge.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
        )
        tb.addWidget(self.mode_badge)
        tb.addSeparator()

        open_act = QAction("📂  Open File", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_file)
        tb.addAction(open_act)
        tb.addSeparator()

        self.btn_run = QPushButton("▶  Analyze")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setFixedWidth(130)
        self.btn_run.setShortcut("F5")
        self.btn_run.clicked.connect(self._on_run_pipeline)
        tb.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_pipeline)
        tb.addWidget(self.btn_stop)
        tb.addSeparator()

        export_act = QAction("📄  Export Report", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._on_export_report)
        tb.addAction(export_act)
        tb.addSeparator()

        btn_live_nav = QPushButton("📡  Live SDR (GNU Radio)")
        btn_live_nav.setObjectName("btn_secondary")
        btn_live_nav.setToolTip("Switch to Live SDR / GNU Radio streaming panel")
        btn_live_nav.clicked.connect(lambda: self.center_tabs.setCurrentWidget(self.live_sdr_tab))
        tb.addWidget(btn_live_nav)


        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.clock_label = QLabel()
        self.clock_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            "font-family: monospace; background: transparent;"
        )
        self._update_clock()
        tb.addWidget(self.clock_label)

        tb.addWidget(QLabel("  v0.1.0  "))

    # ── Center Tabs ───────────────────────────────────────────────────────────
    def _create_center_panel(self) -> None:
        self.center_tabs = QTabWidget()
        self.center_tabs.setDocumentMode(True)
        self.setCentralWidget(self.center_tabs)

        # Tab 0: Spectrum (with drop zone overlay)
        self._spectrum_container = QWidget()
        sc = QVBoxLayout(self._spectrum_container)
        sc.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_dropped.connect(self._load_file)
        sc.addWidget(self.drop_zone)

        self.spectrum_plot = pg.PlotWidget(title="Power Spectral Density")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Power", units="dB")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spectrum_plot.hide()
        sc.addWidget(self.spectrum_plot)
        self.center_tabs.addTab(self._spectrum_container, "📊 Spectrum")

        # Tab 1: Waterfall
        self.waterfall_widget = pg.PlotWidget(title="Waterfall / Spectrogram")
        self.waterfall_widget.setLabel("bottom", "Frequency", units="Hz")
        self.waterfall_widget.setLabel("left", "Time", units="s")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_widget.addItem(self.waterfall_image)
        try:
            cmap = pg.colormap.get("viridis")
            self.waterfall_bar = pg.ColorBarItem(values=(-80, 0), colorMap=cmap)
            self.waterfall_bar.setImageItem(self.waterfall_image)
        except Exception:
            pass
        self.center_tabs.addTab(self.waterfall_widget, "🌊 Waterfall")

        # Tab 2: Constellation
        self.constellation_plot = pg.PlotWidget(title="Constellation Diagram")
        self.constellation_plot.setLabel("bottom", "In-Phase (I)")
        self.constellation_plot.setLabel("left", "Quadrature (Q)")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.showGrid(x=True, y=True, alpha=0.15)
        self.constellation_plot.addLine(x=0, pen=pg.mkPen(COLORS["border"], width=1))
        self.constellation_plot.addLine(y=0, pen=pg.mkPen(COLORS["border"], width=1))
        self.center_tabs.addTab(self.constellation_plot, "✨ Constellation")

        # Tab 3: Eye Diagram
        self.eye_plot = pg.PlotWidget(title="Eye Diagram")
        self.eye_plot.setLabel("bottom", "Sample Offset")
        self.eye_plot.setLabel("left", "Amplitude")
        self.eye_plot.showGrid(x=True, y=True, alpha=0.15)
        self.center_tabs.addTab(self.eye_plot, "👁 Eye Diagram")

        # Tab 4: Bitstream
        self.bitstream_text = QTextEdit()
        self.bitstream_text.setReadOnly(True)
        self.bitstream_text.setFont(QFont("Cascadia Code", 11))
        self.bitstream_text.setPlaceholderText(
            "Decoded bitstream (hex dump) will appear here after analysis."
        )
        self.center_tabs.addTab(self.bitstream_text, "🔢 Bitstream")

        # Tab 5: Frames & Protocol
        self.frames_text = QTextEdit()
        self.frames_text.setReadOnly(True)
        self.frames_text.setFont(QFont("Cascadia Code", 11))
        self.frames_text.setPlaceholderText(
            "Frame structure, FEC results, and signal parameters will appear here."
        )
        self.center_tabs.addTab(self.frames_text, "📋 Frames & Protocol")

        # Tab 6: GNI Report
        self.gni_report_text = QTextEdit()
        self.gni_report_text.setReadOnly(True)
        self.gni_report_text.setFont(QFont("Segoe UI", 11))
        self.gni_report_text.setPlaceholderText(
            "GNI Analysis Report — hypothesis chain, evidence scores, confidence badges."
        )
        self.center_tabs.addTab(self.gni_report_text, "📝 GNI Report")

        # Tab 7: Live SDR (GNU Radio)
        self.live_sdr_tab = self._create_live_sdr_tab()
        self.center_tabs.addTab(self.live_sdr_tab, "📡 Live SDR (GNU Radio)")


    # ── Left Dock ─────────────────────────────────────────────────────────────
    def _create_left_dock(self) -> None:
        dock = QDockWidget("Signal Input & Parameters", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setMinimumWidth(300)
        dock.setMaximumWidth(380)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # File Input
        fg = QGroupBox("File Input")
        fl = QVBoxLayout(fg)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 6px; font-size: 12px;")
        fl.addWidget(self.file_label)
        btn = QPushButton("📂  Browse...")
        btn.clicked.connect(self._on_open_file)
        fl.addWidget(btn)
        layout.addWidget(fg)

        # Metadata
        mg = QGroupBox("Detected Metadata")
        ml = QFormLayout(mg)
        ml.setSpacing(5)
        ml.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.meta_labels: dict = {}
        for key, txt in [("format","Format:"),("sample_rate","Sample Rate:"),
                         ("dtype","Data Type:"),("duration","Duration:"),
                         ("samples","Samples:"),("file_size","File Size:")]:
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            ml.addRow(txt, lbl)
            self.meta_labels[key] = lbl
        layout.addWidget(mg)

        # GNI Framework options
        gg = QGroupBox("GNI Framework")
        gl = QVBoxLayout(gg)
        self.gni_desc_label = QLabel(GNI_MODES["Auto (GNI)"]["description"])
        self.gni_desc_label.setWordWrap(True)
        self.gni_desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        gl.addWidget(self.gni_desc_label)
        self.gni_blind_check = QCheckBox("Force Blind Estimation")
        self.gni_blind_check.setChecked(True)
        self.gni_blind_check.setToolTip("Ignore overrides — estimate all parameters from the signal")
        gl.addWidget(self.gni_blind_check)
        self.gni_multi_hyp = QCheckBox("Multi-Hypothesis Testing")
        self.gni_multi_hyp.setChecked(True)
        self.gni_multi_hyp.setToolTip("Test all modulation candidates (recommended)")
        gl.addWidget(self.gni_multi_hyp)
        layout.addWidget(gg)

        # Overrides
        og = QGroupBox("Parameter Overrides")
        ol = QFormLayout(og)
        ol.setSpacing(6)
        ol.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.override_sample_rate = QLineEdit()
        self.override_sample_rate.setPlaceholderText("e.g., 2048000")
        self.override_sample_rate.setEnabled(False)
        ol.addRow("Sample Rate (Hz):", self.override_sample_rate)
        self.override_symbol_rate = QLineEdit()
        self.override_symbol_rate.setPlaceholderText("e.g., 9600")
        self.override_symbol_rate.setEnabled(False)
        ol.addRow("Symbol Rate (Bd):", self.override_symbol_rate)
        self.override_modulation = QComboBox()
        self.override_modulation.addItems([
            "(auto-detect)", "BPSK", "QPSK", "8-PSK", "OQPSK",
            "2-FSK", "GFSK", "MSK", "16-QAM", "64-QAM",
        ])
        self.override_modulation.setEnabled(False)
        ol.addRow("Modulation:", self.override_modulation)
        layout.addWidget(og)

        # Pipeline stages
        pg_ = QGroupBox("Pipeline Stages")
        pl = QVBoxLayout(pg_)
        pl.setSpacing(3)
        self.stage_widgets: dict = {}
        for sid, sname in PIPELINE_STAGES:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            sp = SpinnerLabel()
            sp.setFixedWidth(20)
            sp.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
            nl = QLabel(sname)
            nl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            tl = QLabel("")
            tl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
            tl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(sp)
            row.addWidget(nl, 1)
            row.addWidget(tl)
            pl.addLayout(row)
            self.stage_widgets[sid] = (sp, nl, tl)
        layout.addWidget(pg_)
        layout.addStretch()

        scroll.setWidget(container)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    # ── Right Dock ────────────────────────────────────────────────────────────
    def _create_right_dock(self) -> None:
        dock = QDockWidget("Analysis Results", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setMinimumWidth(320)
        dock.setMaximumWidth(420)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Metrics gauges
        gauge_g = QGroupBox("Signal Metrics")
        gl = QFormLayout(gauge_g)
        gl.setSpacing(8)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def _gauge(color):
            bar = MiniBarWidget(color)
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            row = QHBoxLayout()
            row.addWidget(bar)
            row.addSpacing(6)
            row.addWidget(lbl)
            w = QWidget()
            w.setLayout(row)
            return w, bar, lbl

        snr_r, self.snr_bar, self.snr_lbl = _gauge(COLORS["success"])
        bw_r,  self.bw_bar,  self.bw_lbl  = _gauge(COLORS["accent"])
        cfo_r, self.cfo_bar, self.cfo_lbl = _gauge(COLORS["warning"])
        evm_r, self.evm_bar, self.evm_lbl = _gauge(COLORS["caution"])
        sr_r,  self.sr_bar,  self.sr_lbl  = _gauge(COLORS["text_secondary"])

        gl.addRow("SNR:", snr_r)
        gl.addRow("Bandwidth:", bw_r)
        gl.addRow("CFO:", cfo_r)
        gl.addRow("EVM:", evm_r)
        gl.addRow("Symbol Rate:", sr_r)
        layout.addWidget(gauge_g)

        # Hypothesis ranking
        hg = QGroupBox("Hypothesis Ranking")
        hv = QVBoxLayout(hg)
        self.hypothesis_container = QVBoxLayout()
        self.hypothesis_container.addWidget(
            self._make_hyp_card("⚪", "Awaiting analysis…", "—", COLORS["neutral"])
        )
        hv.addLayout(self.hypothesis_container)
        layout.addWidget(hg)

        # Evidence
        eg = QGroupBox("Evidence Breakdown")
        ev = QVBoxLayout(eg)
        self.evidence_text = QTextEdit()
        self.evidence_text.setReadOnly(True)
        self.evidence_text.setMaximumHeight(200)
        self.evidence_text.setPlaceholderText("Evidence scores from each analysis stage.")
        ev.addWidget(self.evidence_text)
        layout.addWidget(eg)

        # Warnings
        wg = QGroupBox("Warnings & Limitations")
        wv = QVBoxLayout(wg)
        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        self.warnings_text.setMaximumHeight(130)
        self.warnings_text.setPlaceholderText(
            "UNKNOWN parameters, low-confidence detections, and limitation notices."
        )
        wv.addWidget(self.warnings_text)
        layout.addWidget(wg)
        layout.addStretch()

        scroll.setWidget(container)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _make_hyp_card(self, badge, description, confidence, color) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
        """)
        row = QHBoxLayout(card)
        row.setContentsMargins(8, 6, 8, 6)
        e = QLabel(badge)
        e.setFixedWidth(24)
        e.setStyleSheet("font-size: 15px; background: transparent;")
        row.addWidget(e)
        d = QLabel(description)
        d.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; background: transparent;")
        d.setWordWrap(True)
        row.addWidget(d, 1)
        c = QLabel(confidence)
        c.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 13px; background: transparent;")
        row.addWidget(c)
        return card

    # ── Bottom Dock ───────────────────────────────────────────────────────────
    def _create_bottom_dock(self) -> None:
        dock = QDockWidget("Console & Progress", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 6, 10, 6)

        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setFixedHeight(22)
        prog_row.addWidget(self.progress_bar, 1)
        self.perf_label = QLabel("Time: —")
        self.perf_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; "
            "font-family: monospace; background: transparent;"
        )
        prog_row.addWidget(self.perf_label)
        layout.addLayout(prog_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self._log("SigForge v0.1.0 — Ready. GNI framework active.")
        self._log("Load a signal file via Open or drag-and-drop to begin.")
        layout.addWidget(self.log_text)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    # ── Status Bar ────────────────────────────────────────────────────────────
    def _create_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_mode = QLabel(f"Mode: {self._current_mode}")
        self.status_mode.setStyleSheet(f"color: {COLORS['accent']}; font-weight: 600;")
        sb.addWidget(self.status_mode)
        sep = QLabel("  |  ")
        sep.setStyleSheet(f"color: {COLORS['border']};")
        sb.addWidget(sep)
        self.status_file = QLabel("No file loaded")
        self.status_file.setStyleSheet(f"color: {COLORS['text_muted']};")
        sb.addPermanentWidget(self.status_file)

    # ── Drag & Drop ───────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._load_file(path)
                event.acceptProposedAction()

    # ── File Loading ──────────────────────────────────────────────────────────
    def _load_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            self._log(f"⚠ File not found: {file_path}", level="warn")
            return

        self._current_file = file_path
        fname = os.path.basename(file_path)
        fsize = os.path.getsize(file_path)

        self.file_label.setText(f"📄 {fname}")
        self.file_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; padding: 6px; font-weight: 600; font-size: 12px;"
        )
        self.status_file.setText(f"File: {fname}")
        self.meta_labels["file_size"].setText(self._fmt_size(fsize))

        ext = Path(file_path).suffix.lower()
        self.meta_labels["format"].setText(ext.upper().lstrip(".") if ext else "RAW")

        if ext in (".cf32", ".bin", ".raw", ".iq"):
            n = fsize // 8
            self.meta_labels["dtype"].setText("Complex Float32")
            self.meta_labels["samples"].setText(f"{n:,}")
            self.meta_labels["sample_rate"].setText("1.0 MHz (estimated)")
            self.meta_labels["duration"].setText(f"{n / 1e6 * 1000:.1f} ms")
        elif ext == ".cs16":
            n = fsize // 4
            self.meta_labels["dtype"].setText("Complex Int16")
            self.meta_labels["samples"].setText(f"{n:,}")
            self.meta_labels["sample_rate"].setText("1.0 MHz (estimated)")
            self.meta_labels["duration"].setText(f"{n / 1e6 * 1000:.1f} ms")
        elif ext in (".cu8", ".cs8"):
            n = fsize // 2
            self.meta_labels["dtype"].setText("Complex UInt8")
            self.meta_labels["samples"].setText(f"{n:,}")
            self.meta_labels["sample_rate"].setText("1.0 MHz (estimated)")
            self.meta_labels["duration"].setText(f"{n / 1e6 * 1000:.1f} ms")
        elif ext == ".wav":
            try:
                import scipy.io.wavfile as wf
                sr, data = wf.read(file_path)
                self.meta_labels["sample_rate"].setText(f"{sr/1000:.1f} kHz")
                self.meta_labels["samples"].setText(f"{len(data):,}")
                self.meta_labels["duration"].setText(f"{len(data)/sr:.3f} s")
                self.meta_labels["dtype"].setText(str(data.dtype))
            except Exception as ex:
                self._log(f"⚠ WAV read error: {ex}", level="warn")
        else:
            self.meta_labels["dtype"].setText("Binary / Raw IQ")
            self.meta_labels["sample_rate"].setText("Auto-detect")

        self.drop_zone.hide()
        self.spectrum_plot.show()
        self._log(f"✔ Loaded: {fname}  ({self._fmt_size(fsize)})", level="success")

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Signal File", "",
            "Signal Files (*.bin *.raw *.cf32 *.cs16 *.cu8 *.cs8 *.wav *.sigmf-data *.iq)"
            ";;All Files (*)",
        )
        if path:
            self._load_file(path)

    # ── Pipeline Control ──────────────────────────────────────────────────────
    def _on_run_pipeline(self) -> None:
        if not self._current_file:
            self._log("⚠ No file loaded. Open or drop a signal file first.", level="warn")
            return

        self._log(f"▶ Starting {self._current_mode} pipeline…")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing…  0%")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._analysis_start = time.time()
        self._reset_stage_indicators()

        sr_val = sym_val = mod_val = None

        if self.override_sample_rate.isEnabled():
            t = self.override_sample_rate.text().strip()
            if t:
                try:
                    sr_val = float(t)
                except ValueError:
                    self._log(f"⚠ Bad sample rate: {t!r}", level="warn")

        if self.override_symbol_rate.isEnabled():
            t = self.override_symbol_rate.text().strip()
            if t:
                try:
                    sym_val = float(t)
                except ValueError:
                    self._log(f"⚠ Bad symbol rate: {t!r}", level="warn")

        if self.override_modulation.isEnabled():
            txt = self.override_modulation.currentText()
            if txt != "(auto-detect)":
                try:
                    from sigforge.classify.modulation_types import ModulationType
                    for m in ModulationType:
                        if m.value == txt:
                            mod_val = m
                            break
                except Exception:
                    pass

        self._worker = AnalysisWorker(
            file_path=self._current_file,
            sample_rate_override=sr_val,
            symbol_rate_override=sym_val,
            mod_type_override=mod_val,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.stage_update.connect(self._update_stage)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._log("■ Pipeline stopped by user.", level="warn")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat("Stopped")
        self._stop_all_spinners()

    # ── Worker Callbacks ──────────────────────────────────────────────────────
    def _on_progress(self, value: int, msg: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{msg}  {value}%")
        elapsed = time.time() - self._analysis_start
        self.perf_label.setText(f"Time: {elapsed:.1f}s")

    def _on_finished(self, result: PipelineResult) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._last_result = result
        self._stop_all_spinners()

        elapsed = result.processing_time_sec
        self._log(f"✔ Analysis complete in {elapsed:.2f}s", level="success")
        self.perf_label.setText(f"Time: {elapsed:.2f}s")
        self.progress_bar.setFormat("Complete  100%")

        for e in (result.errors or []):
            self._log(f"  ⚠ {e}", level="warn")

        self._update_gauges(result)
        self._update_hypothesis_panel(result)
        self._update_evidence_panel(result)
        self._update_warnings_panel(result)
        self._update_spectrum_plot(result)
        self._update_waterfall_plot(result)
        self._update_constellation_plot(result)
        self._update_eye_diagram(result)
        self._update_bitstream_tab(result)
        self._update_frames_tab(result)
        self._update_gni_report(result)

        self.center_tabs.setCurrentIndex(0)

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._stop_all_spinners()
        self._log(f"✘ Pipeline error: {msg}", level="error")
        self.progress_bar.setFormat("Error")

    # ── Stage Indicators ──────────────────────────────────────────────────────
    def _reset_stage_indicators(self) -> None:
        for sid, (sp, nl, tl) in self.stage_widgets.items():
            sp.stop()
            sp.setText("○")
            sp.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
            nl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            tl.setText("")

    def _stop_all_spinners(self) -> None:
        for _, (sp, _, _) in self.stage_widgets.items():
            sp.stop()

    def _update_stage(self, stage_id: str, status: str) -> None:
        w = self.stage_widgets.get(stage_id)
        if not w:
            return
        sp, nl, tl = w
        if status == "running":
            sp.start()
            sp.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px;")
            nl.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 11px; font-weight: 600;")
        elif status == "done":
            sp.stop()
            sp.setText("●")
            sp.setStyleSheet(f"color: {COLORS['success']}; font-size: 13px;")
            nl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        elif status == "error":
            sp.stop()
            sp.setText("✘")
            sp.setStyleSheet(f"color: {COLORS['error']}; font-size: 13px;")
            nl.setStyleSheet(f"color: {COLORS['error']}; font-size: 11px;")
        elif status == "skipped":
            sp.stop()
            sp.setText("—")
            sp.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")

    # ── Result Display ────────────────────────────────────────────────────────
    def _update_gauges(self, result: PipelineResult) -> None:
        sp = result.signal_profile
        pe = result.parameter_estimates

        snr = _safe(sp, "snr_db")
        if snr is not None:
            self.snr_bar.set_value(float(np.clip(snr / 40.0, 0, 1)))
            self.snr_lbl.setText(f"{snr:.1f} dB")

        bw = _safe(sp, "occupied_bandwidth")
        if bw is not None:
            self.bw_bar.set_value(float(np.clip(bw / 2e6, 0, 1)))
            self.bw_lbl.setText(self._fmt_hz(bw))

        cfo = _safe(pe, "cfo_hz")
        if cfo is not None:
            self.cfo_bar.set_value(float(np.clip(abs(cfo) / 50000, 0, 1)))
            self.cfo_lbl.setText(f"{cfo:+.1f} Hz")

        evm = None
        if result.demod_results:
            evm = _safe(result.demod_results[0], "evm_db")
        if evm is not None:
            self.evm_bar.set_value(float(np.clip((-evm) / 30.0, 0, 1)))
            self.evm_lbl.setText(f"{evm:.1f} dB")

        sr = _safe(pe, "symbol_rate")
        if sr is not None:
            self.sr_bar.set_value(float(np.clip(sr / 1e6, 0, 1)))
            self.sr_lbl.setText(self._fmt_hz(sr, unit="Bd"))

    def _update_hypothesis_panel(self, result: PipelineResult) -> None:
        while self.hypothesis_container.count():
            item = self.hypothesis_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        graph = result.hypothesis_graph
        if graph is None or not graph.hypotheses:
            self.hypothesis_container.addWidget(
                self._make_hyp_card("⚪", "No hypotheses generated", "—", COLORS["neutral"])
            )
            return

        for hyp in graph.hypotheses:
            badge = _safe(hyp, "badge", "emoji") or "⚪"
            desc  = _safe(hyp, "description") or "Unknown"
            conf  = _safe(hyp, "confidence")
            color = _safe(hyp, "badge", "color_hex") or COLORS["neutral"]
            self.hypothesis_container.addWidget(
                self._make_hyp_card(badge, desc,
                                    f"{conf:.1%}" if conf is not None else "—", color)
            )

    def _update_evidence_panel(self, result: PipelineResult) -> None:
        graph = result.hypothesis_graph
        if graph is None or graph.best is None:
            return
        best = graph.best
        conf = _safe(best, "confidence", default=0.0)
        lines = [f"Best: {_safe(best,'description') or 'Unknown'}  ({conf:.1%})", ""]
        for ev in (best.evidence or []):
            bar = "█" * int(ev.score * 10) + "░" * (10 - int(ev.score * 10))
            lines += [f"[{ev.source}]", f"  {bar}  {ev.score:.3f}  wt={ev.weight:.2f}"]
            if ev.detail:
                lines.append(f"  → {ev.detail}")
        self.evidence_text.setPlainText("\n".join(lines))

    def _update_warnings_panel(self, result: PipelineResult) -> None:
        warns = list(result.warnings or [])
        for hyp in (result.hypothesis_graph.hypotheses if result.hypothesis_graph else []):
            warns.extend(hyp.warnings or [])
        self.warnings_text.setPlainText(
            "\n".join(f"⚠ {w}" for w in warns) if warns else "✔ No warnings."
        )

    def _update_spectrum_plot(self, result: PipelineResult) -> None:
        if result.iq_data is None or len(result.iq_data) == 0:
            return
        try:
            import scipy.signal
            sr = result.sample_rate or 1e6
            freqs, psd = scipy.signal.welch(
                result.iq_data, sr,
                nperseg=min(1024, len(result.iq_data)),
                return_onesided=False,
            )
            freqs = np.fft.fftshift(freqs)
            psd   = np.fft.fftshift(psd)
            psd_db = 10 * np.log10(psd + 1e-14)

            self.spectrum_plot.clear()
            self.spectrum_plot.addLine(
                y=float(np.percentile(psd_db, 5)),
                pen=pg.mkPen(COLORS["border"], style=Qt.PenStyle.DashLine)
            )
            self.spectrum_plot.plot(freqs, psd_db, pen=pg.mkPen(COLORS["accent"], width=1.5))
            self.drop_zone.hide()
            self.spectrum_plot.show()
        except Exception as ex:
            self._log(f"⚠ Spectrum: {ex}", level="warn")

    def _update_waterfall_plot(self, result: PipelineResult) -> None:
        if result.iq_data is None or len(result.iq_data) == 0:
            return
        try:
            import scipy.signal
            sr = result.sample_rate or 1e6
            nperseg = min(256, len(result.iq_data))
            f, t, Sxx = scipy.signal.spectrogram(
                result.iq_data, fs=sr,
                nperseg=nperseg, noverlap=nperseg // 2,
                return_onesided=False,
            )
            f   = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx, axes=0)
            Sxx_db = 10 * np.log10(Sxx + 1e-14)
            self.waterfall_image.setImage(Sxx_db.T, autoLevels=True)
            self.waterfall_image.setRect(
                pg.QtCore.QRectF(float(f[0]), float(t[0]),
                                 float(f[-1]-f[0]), float(t[-1]-t[0]))
            )
        except Exception as ex:
            self._log(f"⚠ Waterfall: {ex}", level="warn")

    def _update_constellation_plot(self, result: PipelineResult) -> None:
        if not result.demod_results:
            return
        try:
            self.constellation_plot.clear()
            self.constellation_plot.addLine(x=0, pen=pg.mkPen(COLORS["border"], width=1))
            self.constellation_plot.addLine(y=0, pen=pg.mkPen(COLORS["border"], width=1))

            pts = result.demod_results[0].constellation_scatter
            if pts is None or len(pts) == 0:
                return

            if len(pts) > 4000:
                idx = np.random.choice(len(pts), 4000, replace=False)
                pts = pts[idx]

            scatter = pg.ScatterPlotItem(
                size=4, pen=pg.mkPen(None), brush=pg.mkBrush(99, 102, 241, 160)
            )
            scatter.addPoints(x=pts.real.tolist(), y=pts.imag.tolist())
            self.constellation_plot.addItem(scatter)
        except Exception as ex:
            self._log(f"⚠ Constellation: {ex}", level="warn")

    def _update_eye_diagram(self, result: PipelineResult) -> None:
        if result.iq_data is None or len(result.iq_data) == 0:
            return
        try:
            pe = result.parameter_estimates
            sps = int(round(_safe(pe, "samples_per_symbol") or 4))
            sps = max(2, min(sps, 64))
            eye_span = 2 * sps
            iq = result.iq_data
            n_traces = min(100, len(iq) // eye_span)

            self.eye_plot.clear()
            pen_i = pg.mkPen(color=(99, 102, 241, 100), width=1.2)
            pen_q = pg.mkPen(color=(34, 197, 94, 80), width=1.2)
            t_ax = np.arange(eye_span)

            for k in range(n_traces):
                seg = iq[k * eye_span: k * eye_span + eye_span]
                if len(seg) == eye_span:
                    self.eye_plot.plot(t_ax, seg.real, pen=pen_i)
                    self.eye_plot.plot(t_ax, seg.imag, pen=pen_q)
        except Exception as ex:
            self._log(f"⚠ Eye diagram: {ex}", level="warn")

    def _update_bitstream_tab(self, result: PipelineResult) -> None:
        if not result.demod_results:
            return
        try:
            bits = result.demod_results[0].hard_bits
            if bits is None or len(bits) == 0:
                self.bitstream_text.setPlainText("(no bits decoded)")
                return
            cap = min(len(bits), 4096)
            lines = [
                f"=== Decoded Bitstream  [{cap:,} / {len(bits):,} bits shown] ===",
                f"{'Offset':6}  {'Hex':47}  ASCII",
                "─" * 62,
            ]
            for b_off in range(0, cap, 64):
                chunk = bits[b_off: b_off + 64]
                hex_s = asc_s = ""
                for j in range(0, len(chunk), 8):
                    byte_bits = chunk[j: j + 8]
                    val = 0
                    for bit in byte_bits:
                        val = (val << 1) | int(bit)
                    hex_s += f"{val:02X} "
                    asc_s += chr(val) if 0x20 <= val <= 0x7E else "."
                lines.append(f"{b_off//8:04X}    {hex_s:<48} |{asc_s}|")
            self.bitstream_text.setPlainText("\n".join(lines))
        except Exception as ex:
            self._log(f"⚠ Bitstream: {ex}", level="warn")

    def _update_frames_tab(self, result: PipelineResult) -> None:
        try:
            pe    = result.parameter_estimates
            sp    = result.signal_profile
            graph = result.hypothesis_graph
            best  = graph.best if graph else None

            lines = [
                "╔══════════════════════════════════════════╗",
                "║       GNI Frame & Protocol Analysis      ║",
                "╚══════════════════════════════════════════╝",
                "",
                "── Signal Parameters ──────────────────────",
                f"  Modulation     : {_safe(best,'modulation','value') or 'Unknown'}",
                f"  Symbol Rate    : {_safe_val(_safe(pe,'symbol_rate'), '{:,.1f} Bd')}",
                f"  Samples/Symbol : {_safe_val(_safe(pe,'samples_per_symbol'), '{:.2f}')}",
                f"  Carrier Offset : {_safe_val(_safe(pe,'cfo_hz'), '{:+.1f} Hz')}",
                f"  Bandwidth (Occ): {self._fmt_hz(_safe(sp,'occupied_bandwidth'))}",
                f"  SNR (estimated): {_safe_val(_safe(sp,'snr_db'), '{:.1f} dB')}",
                f"  Noise Floor    : {_safe_val(_safe(sp,'noise_floor_db'), '{:.1f} dB')}",
                "",
                "── Cumulants ──────────────────────────────",
            ]
            if sp and sp.cumulants:
                for k, v in sp.cumulants.items():
                    lines.append(f"  {k:<5}: {v:.4f}")
            else:
                lines.append("  (none)")

            lines += ["", "── FEC Search ─────────────────────────────"]
            if result.fec_results:
                for fec in result.fec_results:
                    s = "✔ syndrome=0" if fec.syndrome_zero else f"match={fec.re_encode_match:.1%}"
                    lines.append(f"  • {fec.codec_type}/{fec.codec_id}  [{s}]  conf={fec.confidence:.3f}")
            else:
                lines.append("  No standard FEC detected (raw bitstream).")

            lines += ["", "── Hypothesis Chain ────────────────────────"]
            if graph and graph.hypotheses:
                for i, h in enumerate(graph.hypotheses):
                    b = _safe(h,"badge","emoji") or "⚪"
                    d = _safe(h,"description") or "Unknown"
                    c = _safe(h,"confidence") or 0.0
                    lines.append(f"  {i+1}. {b} {d:<28} {c:.3f}")
            else:
                lines.append("  (no hypotheses)")

            self.frames_text.setPlainText("\n".join(lines))
        except Exception as ex:
            self._log(f"⚠ Frames tab: {ex}", level="warn")

    def _update_gni_report(self, result: PipelineResult) -> None:
        try:
            pe    = result.parameter_estimates
            sp    = result.signal_profile
            graph = result.hypothesis_graph
            best  = graph.best if graph else None

            lines = [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "  SigForge — GNI Analysis Report",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  Mode     : {self._current_mode}",
                f"  File     : {os.path.basename(self._current_file or '(none)')}",
                f"  Duration : {result.processing_time_sec:.3f}s",
                "",
                "【G】Guided Parameter Priors",
                f"  Sample Rate Override : {self.override_sample_rate.text() or 'N/A (blind)'}",
                f"  Symbol Rate Override : {self.override_symbol_rate.text() or 'N/A (blind)'}",
                f"  Modulation Override  : {self.override_modulation.currentText()}",
                "",
                "【N】Blind-Estimated Parameters",
                f"  Symbol Rate     : {_safe_val(_safe(pe,'symbol_rate'), '{:,.1f} Bd')}",
                f"  Samples/Symbol  : {_safe_val(_safe(pe,'samples_per_symbol'), '{:.3f}')}",
                f"  Carrier Offset  : {_safe_val(_safe(pe,'cfo_hz'), '{:+.2f} Hz')}",
                f"  SNR (est.)      : {_safe_val(_safe(sp,'snr_db'), '{:.1f} dB')}",
                f"  Noise Floor     : {_safe_val(_safe(sp,'noise_floor_db'), '{:.1f} dB')}",
                f"  Occ. Bandwidth  : {self._fmt_hz(_safe(sp,'occupied_bandwidth'))}",
                f"  Inst. Amp. Std  : {_safe_val(_safe(sp,'inst_amplitude_std'), '{:.4f}')}",
                f"  Kurtosis        : {_safe_val(_safe(sp,'kurtosis'), '{:.4f}')}",
                "",
                "【I】Inferred Hypotheses",
            ]

            if best:
                badge = _safe(best,"badge","emoji") or "⚪"
                desc  = _safe(best,"description") or "Unknown"
                conf  = _safe(best,"confidence") or 0.0
                lines += [
                    f"  Best Match  : {badge} {desc}",
                    f"  Confidence  : {conf:.1%}",
                    f"  Hard Valid. : {'YES ✔' if best.has_hard_validation else 'NO'}",
                    "",
                    "  Evidence Contributions:",
                ]
                for ev in (best.evidence or []):
                    bar = "█" * int(ev.score * 10) + "░" * (10 - int(ev.score * 10))
                    lines.append(f"    [{ev.source:<26}] {bar}  {ev.score:.3f}  wt={ev.weight:.2f}")
                    if ev.detail:
                        lines.append(f"      → {ev.detail}")
            else:
                lines.append("  (no hypotheses generated)")

            lines += ["", "  All Candidates:"]
            if graph and graph.hypotheses:
                for i, h in enumerate(graph.hypotheses):
                    b = _safe(h,"badge","emoji") or "⚪"
                    d = _safe(h,"description") or "Unknown"
                    c = _safe(h,"confidence") or 0.0
                    lines.append(f"    {i+1:2}. {b} {d:<28} {c:.3f}")

            if result.fec_results:
                lines += ["", "  FEC Results:"]
                for fec in result.fec_results:
                    s = "✔" if fec.syndrome_zero else "✗"
                    lines.append(f"    {s} {fec.codec_type}/{fec.codec_id}  match={fec.re_encode_match:.1%}  conf={fec.confidence:.3f}")

            if result.errors:
                lines += ["", "  Errors:"]
                for e in result.errors:
                    lines.append(f"    ✘ {e}")

            lines += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
            self.gni_report_text.setPlainText("\n".join(lines))
        except Exception as ex:
            self._log(f"⚠ GNI report: {ex}", level="warn")

    # ── Export ────────────────────────────────────────────────────────────────
    def _on_export_report(self) -> None:
        if self._last_result is None:
            self._log("⚠ Run analysis first before exporting.", level="warn")
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Analysis Report", "sigforge_report.html",
            "HTML Report (*.html);;JSON Data (*.json);;All Files (*)",
        )
        if not path:
            return
        from sigforge.reports.exporter import ReportExporter
        try:
            hint = os.path.basename(self._current_file or "Unknown")
            if path.endswith(".json") or "JSON" in filt:
                ReportExporter.to_json(self._last_result, path)
            else:
                ReportExporter.to_html(self._last_result, path, filename_hint=hint)
            self._log(f"✔ Report saved: {path}", level="success")
        except Exception as ex:
            self._log(f"✘ Export failed: {ex}", level="error")

    # ── Mode Change ───────────────────────────────────────────────────────────
    def _on_mode_changed(self, mode: str) -> None:
        self._current_mode = mode
        self.status_mode.setText(f"Mode: {mode}")
        cfg = GNI_MODES.get(mode, GNI_MODES["Auto (GNI)"])
        self.mode_badge.setText(cfg["description"])
        self.mode_badge.setStyleSheet(
            f"color: {cfg['color']}; font-size: 11px; background: transparent;"
        )
        enabled = cfg["overrides"]
        self.override_sample_rate.setEnabled(enabled)
        self.override_symbol_rate.setEnabled(enabled)
        self.override_modulation.setEnabled(enabled)
        self.gni_desc_label.setText(cfg["description"])
        self.gni_blind_check.setChecked(mode != "Expert")
        self._log(f"GNI mode → {mode}")

    # ── Utility ───────────────────────────────────────────────────────────────
    def _log(self, message: str, level: str = "info") -> None:
        ts = time.strftime("%H:%M:%S")
        colors = {
            "info":    COLORS["text_primary"],
            "warn":    COLORS["warning"],
            "error":   COLORS["error"],
            "success": COLORS["success"],
        }
        c = colors.get(level, COLORS["text_primary"])
        self.log_text.append(
            f'<span style="color:{COLORS["text_muted"]}">[{ts}]</span> '
            f'<span style="color:{c}">{message}</span>'
        )

    def _update_clock(self) -> None:
        self.clock_label.setText(time.strftime("%H:%M:%S"))

    @staticmethod
    def _fmt_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @staticmethod
    def _fmt_hz(val, unit: str = "Hz") -> str:
        if val is None:
            return "—"
        val = float(val)
        if abs(val) >= 1e6:
            return f"{val/1e6:.3f} M{unit}"
        if abs(val) >= 1e3:
            return f"{val/1e3:.2f} k{unit}"
        return f"{val:.1f} {unit}"

    # ── Live SDR (GNU Radio) Panel ───────────────────────────────────────────
    def _create_live_sdr_tab(self) -> QWidget:
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ── Controls Bar ──
        ctrl_card = QGroupBox("GNU Radio / SDR Connection & Controls")
        ctrl_card.setStyleSheet(f"QGroupBox {{ font-weight: 600; color: {COLORS['text_primary']}; }}")
        cl = QVBoxLayout(ctrl_card)
        cl.setSpacing(8)

        # Row 1: Connection parameters
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Host:"))
        self.sdr_host_input = QLineEdit("127.0.0.1")
        self.sdr_host_input.setFixedWidth(110)
        r1.addWidget(self.sdr_host_input)

        r1.addWidget(QLabel("Port:"))
        self.sdr_port_input = QLineEdit("12345")
        self.sdr_port_input.setFixedWidth(65)
        r1.addWidget(self.sdr_port_input)

        r1.addWidget(QLabel("Protocol:"))
        self.sdr_proto_combo = QComboBox()
        self.sdr_proto_combo.addItems(["TCP", "UDP"])
        self.sdr_proto_combo.setFixedWidth(75)
        r1.addWidget(self.sdr_proto_combo)

        r1.addWidget(QLabel("Format:"))
        self.sdr_format_combo = QComboBox()
        self.sdr_format_combo.addItems(["cf32 (Float32)", "cs16 (Int16)", "cu8 (UInt8)"])
        self.sdr_format_combo.setFixedWidth(125)
        r1.addWidget(self.sdr_format_combo)

        r1.addWidget(QLabel("Center Freq:"))
        self.sdr_freq_input = QLineEdit("433.920")
        self.sdr_freq_input.setFixedWidth(75)
        r1.addWidget(self.sdr_freq_input)
        r1.addWidget(QLabel("MHz"))

        r1.addWidget(QLabel("Sample Rate:"))
        self.sdr_rate_input = QLineEdit("1.000")
        self.sdr_rate_input.setFixedWidth(65)
        r1.addWidget(self.sdr_rate_input)
        r1.addWidget(QLabel("MSPS"))

        r1.addStretch()
        cl.addLayout(r1)

        # Row 2: Action buttons & capture selector
        r2 = QHBoxLayout()
        self.btn_sdr_connect = QPushButton("🔗  Connect SDR")
        self.btn_sdr_connect.setObjectName("btn_secondary")
        self.btn_sdr_connect.setFixedWidth(130)
        self.btn_sdr_connect.clicked.connect(self._on_toggle_live_connection)
        r2.addWidget(self.btn_sdr_connect)

        self.btn_sdr_stream = QPushButton("▶  Start Stream")
        self.btn_sdr_stream.setFixedWidth(120)
        self.btn_sdr_stream.setEnabled(False)
        self.btn_sdr_stream.clicked.connect(self._on_toggle_live_stream)
        r2.addWidget(self.btn_sdr_stream)

        r2.addSpacing(10)
        r2.addWidget(QLabel("Snapshot Size:"))
        self.sdr_capture_combo = QComboBox()
        self.sdr_capture_combo.addItem("65,536 samples (~65 ms)", 65536)
        self.sdr_capture_combo.addItem("131,072 samples (~131 ms)", 131072)
        self.sdr_capture_combo.addItem("262,144 samples (~262 ms)", 262144)
        self.sdr_capture_combo.addItem("524,288 samples (~524 ms)", 524288)
        self.sdr_capture_combo.addItem("1,048,576 samples (~1.05 s)", 1048576)
        self.sdr_capture_combo.setCurrentIndex(1)
        r2.addWidget(self.sdr_capture_combo)

        self.btn_sdr_capture = QPushButton("⚡  Capture & Analyze in SigForge")
        self.btn_sdr_capture.setObjectName("btn_primary")
        self.btn_sdr_capture.setFixedWidth(240)
        self.btn_sdr_capture.setEnabled(False)
        self.btn_sdr_capture.clicked.connect(self._on_live_capture_and_analyze)
        r2.addWidget(self.btn_sdr_capture)

        self.btn_sdr_save = QPushButton("💾  Save IQ")
        self.btn_sdr_save.setFixedWidth(95)
        self.btn_sdr_save.setEnabled(False)
        self.btn_sdr_save.clicked.connect(self._on_save_live_iq)
        r2.addWidget(self.btn_sdr_save)

        r2.addStretch()

        self.btn_sdr_sim = QPushButton("🤖  Launch SDR Simulator")
        self.btn_sdr_sim.setObjectName("btn_accent")
        self.btn_sdr_sim.setToolTip("Start built-in high-performance GNU Radio SDR simulator")
        self.btn_sdr_sim.clicked.connect(self._on_toggle_sdr_simulator)
        r2.addWidget(self.btn_sdr_sim)

        cl.addLayout(r2)
        main_layout.addWidget(ctrl_card)

        # ── Visualizations ──
        v_splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. Live Spectrum
        self.live_spectrum_plot = pg.PlotWidget(title="Live Power Spectral Density (dBFS)")
        self.live_spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.live_spectrum_plot.setLabel("left", "Power", units="dBFS")
        self.live_spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.live_spectrum_curve = self.live_spectrum_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.5)
        )
        self.live_spectrum_peak = pg.TextItem(text="", color=COLORS["warning"], anchor=(0.5, 1.2))
        self.live_spectrum_plot.addItem(self.live_spectrum_peak)
        v_splitter.addWidget(self.live_spectrum_plot)

        # 2. Bottom Row: Waterfall + Diagnostics
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Live Waterfall
        self.live_waterfall_plot = pg.PlotWidget(title="Live Waterfall / Spectrogram")
        self.live_waterfall_plot.setLabel("bottom", "Frequency", units="Hz")
        self.live_waterfall_plot.setLabel("left", "Time History (frames)")
        self.live_waterfall_img = pg.ImageItem()
        self.live_waterfall_plot.addItem(self.live_waterfall_img)
        try:
            cmap = pg.colormap.get("viridis")
            self.live_waterfall_bar = pg.ColorBarItem(values=(-90, -10), colorMap=cmap)
            self.live_waterfall_bar.setImageItem(self.live_waterfall_img)
        except Exception:
            pass
        h_splitter.addWidget(self.live_waterfall_plot)

        # Diagnostics & Status Panel
        diag_card = QGroupBox("Stream Telemetry & Diagnostics")
        diag_layout = QVBoxLayout(diag_card)
        diag_form = QFormLayout()
        diag_form.setSpacing(6)

        self.sdr_status_badge = QLabel("● DISCONNECTED")
        self.sdr_status_badge.setStyleSheet(f"color: {COLORS['error']}; font-weight: 700; font-size: 13px;")
        diag_form.addRow("Status:", self.sdr_status_badge)

        self.sdr_rate_lbl = QLabel("0.00 MB/s (0 kbps)")
        self.sdr_rate_lbl.setStyleSheet(f"color: {COLORS['text_primary']}; font-family: monospace; font-size: 12px;")
        diag_form.addRow("Throughput:", self.sdr_rate_lbl)

        # Ring buffer fill
        self.sdr_buffer_bar = MiniBarWidget(COLORS["accent"])
        self.sdr_buffer_lbl = QLabel("0%")
        self.sdr_buffer_lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        buf_row = QHBoxLayout()
        buf_row.addWidget(self.sdr_buffer_bar)
        buf_row.addWidget(self.sdr_buffer_lbl)
        buf_w = QWidget()
        buf_w.setLayout(buf_row)
        diag_form.addRow("Ring Buffer:", buf_w)

        self.sdr_samples_lbl = QLabel("0 samples")
        self.sdr_samples_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-family: monospace; font-size: 12px;")
        diag_form.addRow("Total Rx:", self.sdr_samples_lbl)

        self.sdr_peak_lbl = QLabel("—")
        self.sdr_peak_lbl.setStyleSheet(f"color: {COLORS['warning']}; font-family: monospace; font-size: 12px;")
        diag_form.addRow("Peak RF:", self.sdr_peak_lbl)

        diag_layout.addLayout(diag_form)

        diag_layout.addWidget(QLabel("Stream Log:"))
        self.sdr_log_text = QTextEdit()
        self.sdr_log_text.setReadOnly(True)
        self.sdr_log_text.setMaximumHeight(150)
        self.sdr_log_text.setFont(QFont("Cascadia Code", 9))
        self.sdr_log_text.setPlaceholderText("Live SDR events and GNU Radio connection logs...")
        diag_layout.addWidget(self.sdr_log_text)

        h_splitter.addWidget(diag_card)
        h_splitter.setSizes([600, 350])

        v_splitter.addWidget(h_splitter)
        v_splitter.setSizes([320, 280])

        main_layout.addWidget(v_splitter)
        return container

    def _log_sdr(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.sdr_log_text.append(f"[{ts}] {message}")
        self._log(f"[Live SDR] {message}")

    def _on_toggle_live_connection(self) -> None:
        if self._live_worker and self._live_worker.isRunning():
            self._live_worker.stop()
            self._live_worker.wait(2000)
            self._live_worker = None
            self._on_live_connected(False, "Disconnected by user.")
            return

        host = self.sdr_host_input.text().strip() or "127.0.0.1"
        try:
            port = int(self.sdr_port_input.text().strip() or "12345")
        except ValueError:
            self._log_sdr("⚠ Bad port number")
            return

        proto = self.sdr_proto_combo.currentText().strip()
        raw_fmt = self.sdr_format_combo.currentText().strip()
        fmt_code = "cf32"
        if "cs16" in raw_fmt:
            fmt_code = "cs16"
        elif "cu8" in raw_fmt:
            fmt_code = "cu8"

        try:
            freq_hz = float(self.sdr_freq_input.text().strip() or "433.92") * 1e6
        except ValueError:
            freq_hz = 433.92e6

        try:
            rate_hz = float(self.sdr_rate_input.text().strip() or "1.0") * 1e6
        except ValueError:
            rate_hz = 1.0e6

        self._log_sdr(f"Connecting to {host}:{port} ({proto}) @ {rate_hz/1e6:.2f} MSPS...")
        self._live_waterfall_buffer = np.full((1024, 100), -95.0, dtype=np.float32)

        self._live_worker = LiveStreamWorker(
            host=host,
            port=port,
            protocol=proto,
            data_format=fmt_code,
            sample_rate=rate_hz,
            center_freq=freq_hz,
            fft_size=1024,
        )
        self._live_worker.connected_signal.connect(self._on_live_connected)
        self._live_worker.fft_update.connect(self._on_live_fft_update)
        self._live_worker.waterfall_row.connect(self._on_live_waterfall_row)
        self._live_worker.stats_update.connect(self._on_live_stats_update)
        self._live_worker.capture_complete.connect(self._on_live_capture_complete)
        self._live_worker.error_signal.connect(lambda e: self._log_sdr(f"✘ Error: {e}"))
        self._live_worker.start()

    def _on_live_connected(self, connected: bool, msg: str) -> None:
        if connected:
            self.sdr_status_badge.setText("● STREAMING LIVE")
            self.sdr_status_badge.setStyleSheet(f"color: {COLORS['success']}; font-weight: 700; font-size: 13px;")
            self.btn_sdr_connect.setText("🔌  Disconnect")
            self.btn_sdr_stream.setEnabled(True)
            self.btn_sdr_stream.setText("⏹  Pause Stream")
            self.btn_sdr_capture.setEnabled(True)
            self.btn_sdr_save.setEnabled(True)
            self._log_sdr(f"✔ {msg}")
        else:
            self.sdr_status_badge.setText("● DISCONNECTED")
            self.sdr_status_badge.setStyleSheet(f"color: {COLORS['error']}; font-weight: 700; font-size: 13px;")
            self.btn_sdr_connect.setText("🔗  Connect SDR")
            self.btn_sdr_stream.setEnabled(False)
            self.btn_sdr_stream.setText("▶  Start Stream")
            self.btn_sdr_capture.setEnabled(False)
            self.btn_sdr_save.setEnabled(False)
            self._log_sdr(f"✘ {msg}")

    def _on_toggle_live_stream(self) -> None:
        if not self._live_worker:
            return
        if self._live_worker.source and self._live_worker.source._running:
            self._live_worker.source.stop_stream()
            self.btn_sdr_stream.setText("▶  Resume Stream")
            self.sdr_status_badge.setText("⏸ PAUSED")
            self.sdr_status_badge.setStyleSheet(f"color: {COLORS['warning']}; font-weight: 700; font-size: 13px;")
            self._log_sdr("Stream paused.")
        elif self._live_worker.source:
            self._live_worker.source.start_stream()
            self.btn_sdr_stream.setText("⏹  Pause Stream")
            self.sdr_status_badge.setText("● STREAMING LIVE")
            self.sdr_status_badge.setStyleSheet(f"color: {COLORS['success']}; font-weight: 700; font-size: 13px;")
            self._log_sdr("Stream resumed.")

    def _on_toggle_sdr_simulator(self) -> None:
        if self._mock_server and self._mock_server.is_running:
            self._mock_server.stop()
            self._mock_server = None
            self.btn_sdr_sim.setText("🤖  Launch SDR Simulator")
            self._log_sdr("■ Built-in SDR simulator server stopped.")
            if self._live_worker and self._live_worker.isRunning():
                self._on_toggle_live_connection()
            return

        host = self.sdr_host_input.text().strip() or "127.0.0.1"
        try:
            port = int(self.sdr_port_input.text().strip() or "12345")
        except ValueError:
            port = 12345

        try:
            rate_hz = float(self.sdr_rate_input.text().strip() or "1.0") * 1e6
        except ValueError:
            rate_hz = 1.0e6

        self._mock_server = GNURadioMockServer(
            host=host,
            port=port,
            sample_rate=rate_hz,
            modulation="QPSK",
        )
        self._mock_server.start()
        self.btn_sdr_sim.setText("⏹  Stop SDR Simulator")
        self._log_sdr(f"✔ SDR Simulator broadcasting synthetic QPSK @ {rate_hz/1e6:.2f} MSPS on {host}:{port}")

        # Automatically connect after brief bind delay
        if not self._live_worker or not self._live_worker.isRunning():
            QTimer.singleShot(250, self._on_toggle_live_connection)

    def _on_live_fft_update(self, freqs, psd_db) -> None:
        self.live_spectrum_curve.setData(freqs, psd_db)
        if len(psd_db) > 0:
            max_idx = int(np.argmax(psd_db))
            p_freq = freqs[max_idx]
            p_val = psd_db[max_idx]
            self.live_spectrum_peak.setText(f"Peak: {p_freq/1e6:.3f} MHz ({p_val:.1f} dBFS)")
            self.live_spectrum_peak.setPos(p_freq, p_val)

    def _on_live_waterfall_row(self, psd_row) -> None:
        if self._live_waterfall_buffer is None:
            self._live_waterfall_buffer = np.full((len(psd_row), 100), -95.0, dtype=np.float32)
        self._live_waterfall_buffer = np.roll(self._live_waterfall_buffer, -1, axis=1)
        self._live_waterfall_buffer[:, -1] = psd_row
        self.live_waterfall_img.setImage(self._live_waterfall_buffer, autoLevels=False, levels=(-90, -10))

    def _on_live_stats_update(self, stats: dict) -> None:
        self.sdr_rate_lbl.setText(f"{stats.get('rate_mbps', 0.0):.2f} MB/s")
        self.sdr_samples_lbl.setText(f"{stats.get('total_samples', 0):,} samples")
        fill = stats.get('buffer_fill', 0)
        cap = max(1, stats.get('buffer_capacity', 1))
        pct = fill / cap
        self.sdr_buffer_bar.set_value(pct)
        self.sdr_buffer_lbl.setText(f"{int(pct * 100)}% ({fill:,})")
        self.sdr_peak_lbl.setText(f"{stats.get('peak_freq', 0.0)/1e6:.4f} MHz ({stats.get('peak_power', 0.0):.1f} dBFS)")

    def _on_live_capture_and_analyze(self) -> None:
        if not self._live_worker or not self._live_worker.isRunning():
            self._log_sdr("⚠ Live stream is not running. Connect or start stream first.")
            return
        self.btn_sdr_capture.setEnabled(False)
        num_samples = self.sdr_capture_combo.currentData() or 131072
        self._log_sdr(f"⚡ Capturing {num_samples:,} live IQ samples from ring buffer...")
        self._live_worker.request_capture(num_samples)

    def _on_live_capture_complete(self, iq_data: np.ndarray, sample_rate: float) -> None:
        self.btn_sdr_capture.setEnabled(True)
        if len(iq_data) == 0:
            self._log_sdr("⚠ Capture buffer empty.")
            return
        self._log_sdr(f"✔ Captured {len(iq_data):,} samples @ {sample_rate/1e6:.2f} MSPS. Ingesting into SigForge GNI pipeline...")
        self._log(f"⚡ Live SDR IQ captured ({len(iq_data):,} samples). Starting {self._current_mode} pipeline…", level="success")

        # Update metadata labels in left dock
        self._current_file = f"Live_SDR_Capture_{int(time.time())}.iq"
        self.file_label.setText(f"📡 Live SDR Capture\n({len(iq_data):,} samples)")
        self.status_file.setText(f"Live SDR: {len(iq_data):,} samples")
        self.meta_labels["format"].setText("GNU Radio SDR")
        self.meta_labels["sample_rate"].setText(f"{sample_rate/1e6:.3f} MHz")
        self.meta_labels["dtype"].setText("Complex Float32")
        self.meta_labels["samples"].setText(f"{len(iq_data):,}")
        self.meta_labels["duration"].setText(f"{len(iq_data)/sample_rate * 1000:.1f} ms")
        self.meta_labels["file_size"].setText(f"{len(iq_data)*8 / 1024:.1f} KB (RAM)")

        self.drop_zone.hide()
        self.spectrum_plot.show()

        # Launch pipeline
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ingesting Live IQ… 0%")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._analysis_start = time.time()
        self._reset_stage_indicators()

        sr_val = sample_rate
        if self.override_sample_rate.isEnabled():
            t = self.override_sample_rate.text().strip()
            if t:
                try:
                    sr_val = float(t)
                except ValueError:
                    pass

        sym_val = None
        if self.override_symbol_rate.isEnabled():
            t = self.override_symbol_rate.text().strip()
            if t:
                try:
                    sym_val = float(t)
                except ValueError:
                    pass

        mod_val = None
        if self.override_modulation.isEnabled():
            txt = self.override_modulation.currentText()
            if txt != "(auto-detect)":
                try:
                    from sigforge.classify.modulation_types import ModulationType
                    for m in ModulationType:
                        if m.value == txt:
                            mod_val = m
                            break
                except Exception:
                    pass

        self._worker = AnalysisWorker(
            iq_data=iq_data,
            sample_rate_override=sr_val,
            symbol_rate_override=sym_val,
            mod_type_override=mod_val,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.stage_update.connect(self._update_stage)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_save_live_iq(self) -> None:
        if not self._live_worker or not self._live_worker.source:
            self._log_sdr("⚠ Live SDR is not connected.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Captured IQ Stream", "live_sdr_capture.cf32",
            "Raw Complex Float32 (*.cf32 *.raw *.iq *.bin);;WAV File (*.wav);;All Files (*)",
        )
        if not path:
            return
        num_samples = self.sdr_capture_combo.currentData() or 131072
        iq = self._live_worker.source.capture_snapshot(num_samples)
        if len(iq) == 0:
            self._log_sdr("⚠ No samples in buffer to save.")
            return
        try:
            self._live_worker.source.save_to_file(path, iq)
            self._log_sdr(f"✔ Saved {len(iq):,} samples to {path}")
            self._log(f"✔ Live IQ saved: {path}", level="success")
        except Exception as ex:
            self._log_sdr(f"✘ Save failed: {ex}")

    def closeEvent(self, event) -> None:
        if self._live_worker and self._live_worker.isRunning():
            self._live_worker.stop()
            self._live_worker.wait(1000)
        if self._mock_server and self._mock_server.is_running:
            self._mock_server.stop()
        event.accept()

