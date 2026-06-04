"""
OscilloscopeWidget
==================

Real-time multi-channel oscilloscope panel for the BLDC Simulator monitoring tab.

Uses **pyqtgraph** when available (fast, native Qt, built-in zoom/pan/crosshair).
Falls back to a **matplotlib NavigationToolbar** canvas when pyqtgraph is absent
so the application remains functional without the optional dependency.

Features
--------
* Up to 4 configurable channel strips, each driven by a ``QComboBox`` selector.
* Scrolling time-window selector (1 s / 5 s / 10 s / 30 s / All data).
* Pause / Resume live update without stopping the simulation.
* Reset-Axes button (auto-scale Y to visible data).
* Per-strip grid toggle (synced to a single checkbox for simplicity).
* Vertical crosshair that spans all strips; a readout label shows the values of
  all active channels at the hovered time.
* Ghost-run overlay: keeps the last completed run as a translucent trace so
  before/after comparisons are immediate.
* Fully accessible: every interactive control has ``accessibleName`` and
  ``accessibleDescription``.

Usage
-----
::

    widget = OscilloscopeWidget(channel_keys=list(engine.get_history().keys()))
    widget.push_sample(time=0.001, values={"speed": 120.0, "torque": 1.3, ...})
    widget.set_paused(False)

:author: SPINOTOR Team
:version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    import pyqtgraph as pg  # type: ignore[import]

    _PYQTGRAPH_AVAILABLE = True
    pg.setConfigOption("background", "#1E1E2E")
    pg.setConfigOption("foreground", "#CDD6F4")
    pg.setConfigOption("antialias", True)
except ImportError:  # pragma: no cover
    _PYQTGRAPH_AVAILABLE = False
    logger.info(
        "pyqtgraph not found — OscilloscopeWidget will use matplotlib fallback. "
        "Install pyqtgraph for best performance: pip install pyqtgraph"
    )

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    _PYSIDE6_AVAILABLE = True
except ImportError:  # pragma: no cover — headless/CI environments
    _PYSIDE6_AVAILABLE = False
    # Provide stub base classes so the module-level constants and
    # _ChannelBuffer can be imported without a running Qt environment.
    Qt = None  # type: ignore[assignment,misc]
    QTimer = None  # type: ignore[assignment,misc]
    QWidget = object  # type: ignore[assignment,misc]
    QFrame = object  # type: ignore[assignment,misc]
    QGroupBox = object  # type: ignore[assignment,misc]
    QHBoxLayout = object  # type: ignore[assignment,misc]
    QVBoxLayout = object  # type: ignore[assignment,misc]
    QSplitter = object  # type: ignore[assignment,misc]
    QCheckBox = object  # type: ignore[assignment,misc]
    QComboBox = object  # type: ignore[assignment,misc]
    QLabel = object  # type: ignore[assignment,misc]
    QPushButton = object  # type: ignore[assignment,misc]
    QSizePolicy = object  # type: ignore[assignment,misc]
    logger.warning(
        "PySide6 not found — OscilloscopeWidget UI classes are unavailable. "
        "Pure-Python components (_ChannelBuffer, CHANNEL_REGISTRY) remain accessible."
    )

# ── Constants ─────────────────────────────────────────────────────────────────

#: Default palette — one color per strip.
_STRIP_COLORS = ["#89B4FA", "#A6E3A1", "#FAB387", "#F38BA8"]
#: Ghost (previous run) alpha multiplier.
_GHOST_ALPHA = 0.35
#: Maximum number of samples kept in the rolling buffer per channel.
_MAX_BUFFER = 50_000

#: All available time windows (seconds).  0 means "show all".
_TIME_WINDOWS = [("1 s", 1.0), ("5 s", 5.0), ("10 s", 10.0), ("30 s", 30.0), ("All", 0.0)]

#: Signals exposed as selectable channels, with human-readable labels and units.
CHANNEL_REGISTRY: dict[str, tuple[str, str]] = {
    "speed_rpm": ("Rotor Speed", "RPM"),
    "speed": ("Rotor Speed (rad/s)", "rad/s"),
    "torque": ("Electromagnetic Torque", "N·m"),
    "load_torque": ("Load Torque", "N·m"),
    "currents_a": ("Phase A Current", "A"),
    "currents_b": ("Phase B Current", "A"),
    "currents_c": ("Phase C Current", "A"),
    "voltages_a": ("Phase A Voltage", "V"),
    "voltages_b": ("Phase B Voltage", "V"),
    "voltages_c": ("Phase C Voltage", "V"),
    "emf_a": ("Back-EMF A", "V"),
    "emf_b": ("Back-EMF B", "V"),
    "emf_c": ("Back-EMF C", "V"),
    "theta": ("Rotor Angle", "rad"),
    "theta_electrical": ("Electrical Angle (est)", "rad"),
    "observer_confidence": ("Observer Confidence", ""),
    "efficiency": ("System Efficiency", ""),
    "input_power": ("Input Power", "W"),
    "mechanical_output_power": ("Mechanical Output Power", "W"),
    "effective_dc_voltage": ("DC-Link Voltage", "V"),
    "dc_link_ripple_v": ("DC-Link Ripple", "V"),
}


# ── Strip model — channel buffer ──────────────────────────────────────────────


class _ChannelBuffer:
    """Ring-buffer holding (time, value) pairs for one channel."""

    def __init__(self, maxlen: int = _MAX_BUFFER) -> None:
        self._t: deque[float] = deque(maxlen=maxlen)
        self._v: deque[float] = deque(maxlen=maxlen)
        self._ghost_t: list[float] = []
        self._ghost_v: list[float] = []

    def push(self, t: float, v: float) -> None:
        self._t.append(t)
        self._v.append(v)

    def snapshot_as_ghost(self) -> None:
        """Freeze current data as the ghost overlay for run comparison."""
        self._ghost_t = list(self._t)
        self._ghost_v = list(self._v)

    def clear(self) -> None:
        self._t.clear()
        self._v.clear()

    # ── Windowed views ────────────────────────────────────────────────────────

    def get_window(self, window_s: float) -> tuple[list[float], list[float]]:
        """Return (time, value) lists trimmed to the last *window_s* seconds."""
        if not self._t:
            return [], []
        if window_s <= 0:
            return list(self._t), list(self._v)
        t_now = self._t[-1]
        cutoff = t_now - window_s
        pairs = [(t, v) for t, v in zip(self._t, self._v) if t >= cutoff]
        if not pairs:  # pragma: no cover - unreachable: latest sample always satisfies cutoff
            return [], []
        ts, vs = zip(*pairs)
        return list(ts), list(vs)

    def get_ghost(self) -> tuple[list[float], list[float]]:
        return self._ghost_t, self._ghost_v

    def last_time(self) -> float:
        return self._t[-1] if self._t else 0.0

    def last_value(self) -> float:
        return self._v[-1] if self._v else float("nan")


# ── pyqtgraph-based strip ─────────────────────────────────────────────────────

if _PYQTGRAPH_AVAILABLE:

    class _PgStrip(QFrame):
        """One pyqtgraph PlotWidget strip with crosshair support."""

        def __init__(
            self,
            key: str,
            label: str,
            unit: str,
            color: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.key = key
            self.label = label
            self.unit = unit
            self.color = color

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel("left", label, units=unit)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self.plot_widget)

            # Main curve
            pen = pg.mkPen(color=color, width=2)
            self.curve = self.plot_widget.plot([], [], pen=pen, name=label)

            # Ghost curve (previous run)
            ghost_pen = pg.mkPen(
                color=color + "59",  # ~35 % alpha in hex
                width=1,
                style=Qt.PenStyle.DashLine,
            )
            self.ghost_curve = self.plot_widget.plot([], [], pen=ghost_pen, name=label + " (prev)")

            # Crosshair vertical line
            self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#FFFFFF", width=1))
            self.plot_widget.addItem(self._vline, ignoreBounds=True)

            # Legend
            self.plot_widget.addLegend(offset=(10, 10))

        def update_data(
            self,
            t: list[float],
            v: list[float],
            ghost_t: list[float],
            ghost_v: list[float],
        ) -> None:
            import numpy as np

            self.curve.setData(np.array(t), np.array(v))
            if ghost_t:
                self.ghost_curve.setData(np.array(ghost_t), np.array(ghost_v))

        def set_crosshair(self, x: float) -> None:
            self._vline.setPos(x)

        def set_grid(self, on: bool) -> None:
            self.plot_widget.showGrid(x=on, y=on, alpha=0.25)

        def autoscale(self) -> None:
            self.plot_widget.enableAutoRange()


# ── matplotlib fallback strip ─────────────────────────────────────────────────


class _MplStrip(QFrame):  # pragma: no cover - fallback when pyqtgraph absent
    """One matplotlib-canvas strip (fallback when pyqtgraph absent)."""

    def __init__(
        self,
        key: str,
        label: str,
        unit: str,
        color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self.label = label
        self.unit = unit
        self.color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(5, 2), dpi=80, facecolor="#1E1E2E")
        self.ax = self.fig.add_subplot(111, facecolor="#1E1E2E")
        self.ax.tick_params(colors="#CDD6F4")
        self.ax.yaxis.label.set_color("#CDD6F4")
        self.ax.xaxis.label.set_color("#CDD6F4")
        self.ax.title.set_color("#CDD6F4")
        self.ax.set_ylabel(f"{label} ({unit})" if unit else label, fontsize=8, color="#CDD6F4")
        self.ax.grid(True, alpha=0.25, color="#45475A")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475A")

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setMaximumHeight(28)

        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        self._line = None
        self._ghost_line = None
        self._vline = None

    def update_data(
        self,
        t: list[float],
        v: list[float],
        ghost_t: list[float],
        ghost_v: list[float],
    ) -> None:
        import numpy as np

        self.ax.cla()
        self.ax.set_facecolor("#1E1E2E")
        self.ax.tick_params(colors="#CDD6F4")
        self.ax.set_ylabel(
            f"{self.label} ({self.unit})" if self.unit else self.label,
            fontsize=8,
            color="#CDD6F4",
        )
        self.ax.grid(True, alpha=0.25, color="#45475A")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475A")

        if t:
            self.ax.plot(np.array(t), np.array(v), color=self.color, linewidth=1.8)
        if ghost_t:
            self.ax.plot(
                np.array(ghost_t),
                np.array(ghost_v),
                color=self.color,
                linewidth=1.0,
                linestyle="--",
                alpha=_GHOST_ALPHA,
            )
        self.canvas.draw_idle()

    def set_crosshair(self, x: float) -> None:  # noqa: ARG002
        pass  # Handled by mpl toolbar

    def set_grid(self, on: bool) -> None:
        self.ax.grid(on, alpha=0.25, color="#45475A")
        self.canvas.draw_idle()

    def autoscale(self) -> None:
        self.ax.autoscale()
        self.canvas.draw_idle()


# ── Channel header (strip label + channel selector) ───────────────────────────


class _StripHeader(QWidget):
    """Compact header row above a strip: channel label + dropdown selector."""

    def __init__(
        self,
        strip_index: int,
        available_keys: list[str],
        default_key: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        label = QLabel(f"Ch {strip_index + 1}:")
        label.setFixedWidth(36)
        layout.addWidget(label)

        self.combo = QComboBox()
        self.combo.setAccessibleName(f"Channel {strip_index + 1} signal selector")
        self.combo.setAccessibleDescription(
            "Choose the signal displayed on this oscilloscope strip."
        )
        self.combo.addItem("— Off —", userData=None)
        for key in available_keys:
            reg = CHANNEL_REGISTRY.get(key)
            display = reg[0] if reg else key
            self.combo.addItem(display, userData=key)

        # Pre-select default
        idx = next(
            (i for i in range(self.combo.count()) if self.combo.itemData(i) == default_key),
            0,
        )
        self.combo.setCurrentIndex(idx)
        layout.addWidget(self.combo, 1)

    @property
    def current_key(self) -> str | None:
        data = self.combo.currentData()
        return str(data) if data is not None else None


# ── Main OscilloscopeWidget ───────────────────────────────────────────────────


class OscilloscopeWidget(QWidget):
    """
    Multi-channel live oscilloscope panel.

    Parameters
    ----------
    channel_keys:
        All signal keys exposed to the user (typically ``engine.get_history().keys()``).
    n_strips:
        Number of strips (1–4, default 3).
    default_keys:
        Pre-selected keys for each strip.  Padded with ``None`` if shorter than
        *n_strips*.
    parent:
        Parent Qt widget.
    """

    #: Suggested default channels
    _DEFAULT_KEYS = ["speed_rpm", "torque", "currents_a"]

    def __init__(
        self,
        channel_keys: list[str] | None = None,
        n_strips: int = 3,
        default_keys: list[str | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._channel_keys: list[str] = channel_keys or list(CHANNEL_REGISTRY.keys())
        self._n_strips: int = max(1, min(n_strips, 4))
        _def = list(default_keys or self._DEFAULT_KEYS)
        while len(_def) < self._n_strips:
            _def.append(None)
        self._default_keys = _def

        self._paused: bool = False
        self._window_s: float = 5.0
        self._grid_on: bool = True

        # Per-channel circular buffers (keyed by signal key)
        self._buffers: dict[str, _ChannelBuffer] = {}

        # Per-strip active key (may change when user edits dropdown)
        self._active_keys: list[str | None] = list(self._default_keys[: self._n_strips])

        # Redraw timer (decoupled from data push rate)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(100)  # 10 Hz refresh
        self._redraw_timer.timeout.connect(self._refresh)
        self._redraw_timer.start()

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Pause / Resume
        self._btn_pause = QPushButton("⏸ Pause")
        self._btn_pause.setCheckable(True)
        self._btn_pause.setFixedWidth(90)
        self._btn_pause.setAccessibleName("Pause live update")
        self._btn_pause.setAccessibleDescription(
            "Freeze the oscilloscope display without stopping the simulation."
        )
        self._btn_pause.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self._btn_pause)

        # Reset axes
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setFixedWidth(80)
        btn_reset.setAccessibleName("Reset axes")
        btn_reset.setAccessibleDescription("Auto-scale all strip Y-axes to the visible data.")
        btn_reset.clicked.connect(self._autoscale_all)
        toolbar.addWidget(btn_reset)

        # Grid toggle
        self._chk_grid = QCheckBox("Grid")
        self._chk_grid.setChecked(True)
        self._chk_grid.setAccessibleName("Show grid on oscilloscope")
        self._chk_grid.setAccessibleDescription("Toggle grid lines on all oscilloscope strips.")
        self._chk_grid.toggled.connect(self._on_grid_toggled)
        toolbar.addWidget(self._chk_grid)

        # Ghost run
        btn_ghost = QPushButton("📸 Snapshot")
        btn_ghost.setFixedWidth(100)
        btn_ghost.setAccessibleName("Save ghost snapshot")
        btn_ghost.setAccessibleDescription(
            "Capture current traces as ghost overlays for before/after comparison."
        )
        btn_ghost.clicked.connect(self._snapshot_ghost)
        toolbar.addWidget(btn_ghost)

        toolbar.addStretch()

        # Time window
        lbl_win = QLabel("Window:")
        lbl_win.setFixedWidth(55)
        toolbar.addWidget(lbl_win)
        self._combo_window = QComboBox()
        self._combo_window.setFixedWidth(70)
        self._combo_window.setAccessibleName("Time window selector")
        self._combo_window.setAccessibleDescription(
            "Choose how many seconds of history are shown in the oscilloscope."
        )
        for label, value in _TIME_WINDOWS:
            self._combo_window.addItem(label, userData=value)
        self._combo_window.setCurrentIndex(1)  # default: 5 s
        self._combo_window.currentIndexChanged.connect(self._on_window_changed)
        toolbar.addWidget(self._combo_window)

        root.addLayout(toolbar)

        # ── Crosshair readout ─────────────────────────────────────────────────
        self._readout_label = QLabel("Hover over a strip to read values.")
        self._readout_label.setStyleSheet("color: #CDD6F4; font-size: 9pt; padding: 2px;")
        self._readout_label.setAccessibleName("Oscilloscope crosshair readout")
        self._readout_label.setAccessibleDescription(
            "Shows signal values at the current cursor position."
        )
        root.addWidget(self._readout_label)

        # ── Strips in a vertical splitter ─────────────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setAccessibleName("Oscilloscope strips")
        self._splitter.setAccessibleDescription(
            "Vertically stacked signal strips. Drag separators to resize."
        )

        self._strips: list[_PgStrip | _MplStrip] = []
        self._headers: list[_StripHeader] = []

        for i in range(self._n_strips):
            default_k = self._default_keys[i]
            container = QWidget()
            c_layout = QVBoxLayout(container)
            c_layout.setContentsMargins(0, 0, 0, 0)
            c_layout.setSpacing(0)

            header = _StripHeader(i, self._channel_keys, default_k or "")
            header.combo.currentIndexChanged.connect(
                lambda _val, idx=i: self._on_channel_changed(idx)
            )
            c_layout.addWidget(header)
            self._headers.append(header)

            key = default_k or ""
            reg = CHANNEL_REGISTRY.get(key, (key, ""))
            label, unit = reg[0], reg[1]
            color = _STRIP_COLORS[i % len(_STRIP_COLORS)]

            if _PYQTGRAPH_AVAILABLE:
                strip: _PgStrip | _MplStrip = _PgStrip(key, label, unit, color, container)
                # Connect crosshair mouse event
                assert isinstance(strip, _PgStrip)
                strip.plot_widget.scene().sigMouseMoved.connect(
                    lambda pos, idx=i, s=strip: self._on_mouse_moved(pos, idx, s)
                )
            else:  # pragma: no cover - matplotlib fallback path
                strip = _MplStrip(key, label, unit, color, container)

            self._strips.append(strip)
            c_layout.addWidget(strip, 1)

            self._splitter.addWidget(container)

        root.addWidget(self._splitter, 1)

        # Backend info label (only when fallback)
        if not _PYQTGRAPH_AVAILABLE:  # pragma: no cover - matplotlib fallback path
            note = QLabel(
                "ℹ️ pyqtgraph not installed — using matplotlib fallback. "
                "Run: pip install pyqtgraph  for full oscilloscope features."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #FAB387; font-size: 8pt;")
            root.addWidget(note)

    # ── Public API ────────────────────────────────────────────────────────────

    def push_sample(self, time: float, values: dict[str, float]) -> None:
        """
        Push one telemetry sample to all channel buffers.

        Parameters
        ----------
        time:
            Simulation timestamp [s].
        values:
            Mapping from signal key to current float value.
        """
        if self._paused:
            return
        for key, val in values.items():
            if key not in self._buffers:
                self._buffers[key] = _ChannelBuffer()
            try:
                self._buffers[key].push(time, float(val))
            except (TypeError, ValueError):
                pass

    def set_paused(self, paused: bool) -> None:
        """Programmatically pause or resume live updates."""
        self._paused = paused
        self._btn_pause.setChecked(paused)
        self._btn_pause.setText("▶ Resume" if paused else "⏸ Pause")

    def set_available_keys(self, keys: list[str]) -> None:
        """Replace the list of selectable channels (e.g. after engine reset)."""
        self._channel_keys = keys
        for i, header in enumerate(self._headers):
            current = header.current_key
            header.combo.blockSignals(True)
            header.combo.clear()
            header.combo.addItem("— Off —", userData=None)
            for k in keys:
                reg = CHANNEL_REGISTRY.get(k)
                display = reg[0] if reg else k
                header.combo.addItem(display, userData=k)
            # Restore previous selection if still available
            idx = next(
                (j for j in range(header.combo.count()) if header.combo.itemData(j) == current),
                0,
            )
            header.combo.setCurrentIndex(idx)
            header.combo.blockSignals(False)
            self._active_keys[i] = header.current_key

    def clear_data(self) -> None:
        """Clear all live buffers (call at simulation reset)."""
        for buf in self._buffers.values():
            buf.clear()

    def start_new_run(self) -> None:
        """
        Snapshot current data as ghost and clear for a fresh run.
        Useful when the user clicks Run again without explicitly snapshotting.
        """
        self._snapshot_ghost()
        self.clear_data()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = checked
        self._btn_pause.setText("▶ Resume" if checked else "⏸ Pause")

    def _on_window_changed(self, _index: int) -> None:
        self._window_s = float(self._combo_window.currentData() or 0.0)

    def _on_grid_toggled(self, checked: bool) -> None:
        self._grid_on = checked
        for strip in self._strips:
            strip.set_grid(checked)

    def _on_channel_changed(self, strip_index: int) -> None:
        key = self._headers[strip_index].current_key
        self._active_keys[strip_index] = key
        strip = self._strips[strip_index]
        if key:
            reg = CHANNEL_REGISTRY.get(key, (key, ""))
            strip.label = reg[0]
            strip.unit = reg[1]
            strip.key = key
        # Force immediate redraw with empty data
        strip.update_data([], [], [], [])

    def _on_mouse_moved(
        self,
        pos: Any,
        strip_index: int,
        source_strip: Any,
    ) -> None:
        """Propagate crosshair and update readout for pyqtgraph strips."""
        if not _PYQTGRAPH_AVAILABLE:  # pragma: no cover - guard for matplotlib fallback path
            return
        try:
            vb = source_strip.plot_widget.plotItem.vb
            if source_strip.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                x = mouse_point.x()
                for strip in self._strips:
                    strip.set_crosshair(x)
                # Build readout text
                parts = [f"t = {x:.4f} s"]
                for i, strip in enumerate(self._strips):
                    k = self._active_keys[i]
                    if k and k in self._buffers:
                        buf = self._buffers[k]
                        ts, vs = buf.get_window(self._window_s)
                        if ts:
                            import bisect

                            idx = bisect.bisect_left(ts, x)
                            idx = max(0, min(idx, len(vs) - 1))
                            reg = CHANNEL_REGISTRY.get(k, (k, ""))
                            parts.append(f"Ch{i + 1} {reg[0]}: {vs[idx]:.4g} {reg[1]}")
                self._readout_label.setText("  |  ".join(parts))
        except Exception:
            pass

    def _autoscale_all(self) -> None:
        for strip in self._strips:
            strip.autoscale()

    def _snapshot_ghost(self) -> None:
        """Save current traces as ghost overlay (run comparison)."""
        for key, buf in self._buffers.items():
            buf.snapshot_as_ghost()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Called by the 10-Hz timer — update all strips from their buffers."""
        if self._paused:
            return
        for i, (strip, key) in enumerate(zip(self._strips, self._active_keys)):
            if not key or key not in self._buffers:
                strip.update_data([], [], [], [])
                continue
            buf = self._buffers[key]
            t, v = buf.get_window(self._window_s)
            gt, gv = buf.get_ghost()
            strip.update_data(t, v, gt, gv)
