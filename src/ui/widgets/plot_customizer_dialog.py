"""
PlotCustomizerDialog
====================

A Qt dialog that lets users customize any matplotlib Figure produced by
:class:`~src.visualization.visualization.SimulationPlotter` **after** it has
been generated — without having to re-run the simulation.

Features
--------
* **Figure settings**: overall title, figure size (W × H in inches), DPI.
* **Axes styling**: font sizes for title, axis labels, and tick labels; optional
  tight-layout enforcement.
* **Trace styling**: per-trace color (color-picker), line width, and line style.
* **Legend**: on/off, position, font size.
* **Grid**: major/minor grid toggle, alpha slider.
* **Export presets**: one-click presets for common publication formats.
* **Live preview**: clicking "Apply" re-renders the figure inside the dialog's
  embedded canvas so the user sees the result before saving.
* **Save to file**: PNG / PDF / SVG / EPS with user-chosen DPI.

All controls are accessible (``accessibleName`` + ``accessibleDescription``).

Usage
-----
::

    fig = SimulationPlotter.create_current_plot(history)
    dlg = PlotCustomizerDialog(fig, parent=self)
    dlg.exec()

:author: SPINOTOR Team
:version: 1.0.0
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ── Export presets ────────────────────────────────────────────────────────────


@dataclass
class ExportPreset:
    """Named export preset for publication/report use."""

    name: str
    figsize: tuple[float, float]
    dpi: int
    title_fontsize: int
    axis_fontsize: int
    tick_fontsize: int
    legend_fontsize: int
    linewidth: float
    description: str = ""


EXPORT_PRESETS: list[ExportPreset] = [
    ExportPreset(
        name="Screen (default)",
        figsize=(10.0, 6.0),
        dpi=96,
        title_fontsize=12,
        axis_fontsize=10,
        tick_fontsize=9,
        legend_fontsize=9,
        linewidth=1.5,
        description="Optimised for on-screen display.",
    ),
    ExportPreset(
        name="IEEE 2-column",
        figsize=(3.5, 2.6),
        dpi=300,
        title_fontsize=8,
        axis_fontsize=7,
        tick_fontsize=6,
        legend_fontsize=6,
        linewidth=1.0,
        description="Fits a single column of an IEEE double-column paper (88.9 mm wide).",
    ),
    ExportPreset(
        name="IEEE 1-column (full width)",
        figsize=(7.16, 4.0),
        dpi=300,
        title_fontsize=10,
        axis_fontsize=9,
        tick_fontsize=8,
        legend_fontsize=8,
        linewidth=1.2,
        description="Full-width figure for an IEEE double-column paper (182 mm wide).",
    ),
    ExportPreset(
        name="A4 Report",
        figsize=(6.3, 4.0),
        dpi=150,
        title_fontsize=11,
        axis_fontsize=10,
        tick_fontsize=9,
        legend_fontsize=9,
        linewidth=1.5,
        description="Suitable for embedding in an A4 Word / PDF report.",
    ),
    ExportPreset(
        name="Presentation 16:9",
        figsize=(10.0, 5.6),
        dpi=120,
        title_fontsize=14,
        axis_fontsize=12,
        tick_fontsize=11,
        legend_fontsize=11,
        linewidth=2.0,
        description="Large text and thick lines for slide decks.",
    ),
    ExportPreset(
        name="High-DPI Print (600 dpi)",
        figsize=(6.3, 4.0),
        dpi=600,
        title_fontsize=11,
        axis_fontsize=10,
        tick_fontsize=9,
        legend_fontsize=9,
        linewidth=1.2,
        description="600 dpi print quality.",
    ),
]


# ── PlotStyle dataclass (shared with SimulationPlotter) ───────────────────────


@dataclass
class PlotStyle:
    """
    Styling parameters that can be applied to any matplotlib Figure.

    This dataclass acts as the single source of truth passed between
    :class:`PlotCustomizerDialog`, :class:`PlotStyleApplicator`, and
    :class:`~src.visualization.visualization.SimulationPlotter`.
    """

    # Figure
    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: int = 96
    suptitle: str = ""

    # Typography
    title_fontsize: int = 12
    axis_fontsize: int = 10
    tick_fontsize: int = 9
    legend_fontsize: int = 9

    # Lines
    linewidth: float = 1.5
    #: Override colors per trace label (label -> hex color string)
    trace_colors: dict[str, str] = field(default_factory=dict)
    #: Override line styles per trace label
    trace_linestyles: dict[str, str] = field(default_factory=dict)

    # Grid
    grid_major: bool = True
    grid_minor: bool = False
    grid_alpha: float = 0.3

    # Legend
    legend_visible: bool = True
    legend_loc: str = "best"

    # Layout
    tight_layout: bool = True

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_preset(cls, preset: ExportPreset) -> PlotStyle:
        return cls(
            figsize=preset.figsize,
            dpi=preset.dpi,
            title_fontsize=preset.title_fontsize,
            axis_fontsize=preset.axis_fontsize,
            tick_fontsize=preset.tick_fontsize,
            legend_fontsize=preset.legend_fontsize,
            linewidth=preset.linewidth,
        )


# ── Style applicator (stateless, no Qt dependency) ───────────────────────────


class PlotStyleApplicator:
    """
    Apply a :class:`PlotStyle` to an existing matplotlib Figure in-place.

    This class has **no Qt dependency** and can therefore be tested without a
    running ``QApplication``.
    """

    @staticmethod
    def apply(fig: Figure, style: PlotStyle) -> None:  # noqa: C901
        """
        Apply *style* to *fig* in-place.

        Parameters
        ----------
        fig:
            Target matplotlib Figure (modified in-place).
        style:
            Style settings to apply.
        """
        # Figure-level
        fig.set_size_inches(style.figsize[0], style.figsize[1])
        fig.set_dpi(style.dpi)
        if style.suptitle:
            fig.suptitle(style.suptitle, fontsize=style.title_fontsize + 2, fontweight="bold")

        for ax in fig.get_axes():
            # Axis labels & ticks
            ax.title.set_fontsize(style.title_fontsize)
            ax.xaxis.label.set_fontsize(style.axis_fontsize)
            ax.yaxis.label.set_fontsize(style.axis_fontsize)
            ax.tick_params(axis="both", labelsize=style.tick_fontsize)

            # Grid
            ax.grid(style.grid_major, alpha=style.grid_alpha)
            if style.grid_minor:
                ax.minorticks_on()
                ax.grid(which="minor", alpha=style.grid_alpha * 0.4, linestyle=":")
            else:
                ax.minorticks_off()

            # Legend
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(style.legend_visible)
                if style.legend_visible:
                    # set_fontsize is not universally available; update text items
                    for txt in legend.get_texts():
                        txt.set_fontsize(style.legend_fontsize)

            # Line widths and colors
            for line in ax.get_lines():
                line.set_linewidth(style.linewidth)
                lbl: str = str(line.get_label())
                if lbl and lbl in style.trace_colors:
                    line.set_color(style.trace_colors[lbl])
                if lbl and lbl in style.trace_linestyles:
                    line.set_linestyle(style.trace_linestyles[lbl])

        if style.tight_layout:
            try:
                fig.tight_layout()
            except Exception:  # pragma: no cover - matplotlib rarely raises here
                pass

    @staticmethod
    def save(fig: Figure, path: Path, dpi: int | None = None) -> None:
        """Save *fig* to *path* with optional DPI override."""
        path.parent.mkdir(parents=True, exist_ok=True)
        effective_dpi = dpi or fig.get_dpi()
        fig.savefig(str(path), dpi=effective_dpi, bbox_inches="tight")
        logger.info("Figure saved to %s (dpi=%d)", path, effective_dpi)


# ── Qt dialog ─────────────────────────────────────────────────────────────────

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from PySide6.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover - PySide6 always available in tests
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    class PlotCustomizerDialog(QDialog):
        """
        Dialog window to customise a matplotlib Figure for publication or reports.

        The dialog shows a live preview panel on the right and a settings panel on
        the left.  Changes are applied to a **deep copy** of the figure; clicking
        **Save** overwrites the original.

        Parameters
        ----------
        figure:
            The matplotlib Figure to customise.
        parent:
            Parent Qt widget.
        """

        def __init__(self, figure: Figure, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Customize Plot")
            self.resize(1100, 700)
            self.setAccessibleName("Plot Customizer Dialog")
            self.setAccessibleDescription(
                "Configure typography, colors, grid, and export settings for the plot."
            )

            self._original_fig = figure
            self._work_fig: Figure = copy.deepcopy(figure)
            self._style = PlotStyle()

            self._build_ui()
            self._apply_preview()

        # ── UI ────────────────────────────────────────────────────────────────

        def _build_ui(self) -> None:
            root = QHBoxLayout(self)
            root.setSpacing(6)

            # ── Left: settings panel (tabs) ───────────────────────────────────
            settings_scroll = QScrollArea()
            settings_scroll.setWidgetResizable(True)
            settings_scroll.setFixedWidth(420)
            settings_scroll.setAccessibleName("Plot style settings")

            settings_widget = QWidget()
            settings_layout = QVBoxLayout(settings_widget)
            settings_layout.setSpacing(6)

            tabs = QTabWidget()
            tabs.setAccessibleName("Style tabs")
            tabs.addTab(self._build_figure_tab(), "Figure")
            tabs.addTab(self._build_typography_tab(), "Typography")
            tabs.addTab(self._build_traces_tab(), "Traces")
            tabs.addTab(self._build_grid_tab(), "Grid")
            tabs.addTab(self._build_legend_tab(), "Legend")
            tabs.addTab(self._build_export_tab(), "Export")
            settings_layout.addWidget(tabs)

            # Preset bar
            preset_grp = QGroupBox("Quick Presets")
            preset_layout = QVBoxLayout(preset_grp)
            self._preset_combo = QComboBox()
            self._preset_combo.setAccessibleName("Export preset selector")
            for p in EXPORT_PRESETS:
                self._preset_combo.addItem(p.name)
            btn_apply_preset = QPushButton("Apply Preset")
            btn_apply_preset.setAccessibleName("Apply selected export preset")
            btn_apply_preset.clicked.connect(self._apply_preset)
            preset_layout.addWidget(self._preset_combo)
            preset_layout.addWidget(btn_apply_preset)
            settings_layout.addWidget(preset_grp)

            # Action buttons
            btn_layout = QHBoxLayout()
            btn_preview = QPushButton("👁 Preview")
            btn_preview.setAccessibleName("Preview changes")
            btn_preview.clicked.connect(self._apply_preview)
            btn_save = QPushButton("💾 Save to file…")
            btn_save.setAccessibleName("Save figure to file")
            btn_save.clicked.connect(self._save_to_file)
            btn_apply_original = QPushButton("✔ Apply to original")
            btn_apply_original.setAccessibleName("Apply style to original figure")
            btn_apply_original.setAccessibleDescription(
                "Apply the current style to the original figure object used by the simulation."
            )
            btn_apply_original.clicked.connect(self._apply_to_original)
            btn_layout.addWidget(btn_preview)
            btn_layout.addWidget(btn_save)
            settings_layout.addLayout(btn_layout)
            settings_layout.addWidget(btn_apply_original)

            settings_scroll.setWidget(settings_widget)
            root.addWidget(settings_scroll)

            # ── Right: live preview ───────────────────────────────────────────
            preview_group = QGroupBox("Live Preview")
            preview_layout = QVBoxLayout(preview_group)

            self._canvas = FigureCanvas(self._work_fig)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._canvas.setAccessibleName("Plot preview canvas")
            preview_layout.addWidget(self._canvas)
            root.addWidget(preview_group, 1)

        # ── Tabs ──────────────────────────────────────────────────────────────

        def _build_figure_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)

            self._suptitle_edit = QLineEdit()
            self._suptitle_edit.setPlaceholderText("(none)")
            self._suptitle_edit.setAccessibleName("Figure super-title")
            form.addRow("Super-title:", self._suptitle_edit)

            self._fig_w = QDoubleSpinBox()
            self._fig_w.setRange(1.0, 30.0)
            self._fig_w.setSingleStep(0.5)
            self._fig_w.setValue(self._style.figsize[0])
            self._fig_w.setAccessibleName("Figure width in inches")
            form.addRow("Width (in):", self._fig_w)

            self._fig_h = QDoubleSpinBox()
            self._fig_h.setRange(1.0, 20.0)
            self._fig_h.setSingleStep(0.5)
            self._fig_h.setValue(self._style.figsize[1])
            self._fig_h.setAccessibleName("Figure height in inches")
            form.addRow("Height (in):", self._fig_h)

            self._dpi_spin = QSpinBox()
            self._dpi_spin.setRange(50, 1200)
            self._dpi_spin.setSingleStep(50)
            self._dpi_spin.setValue(self._style.dpi)
            self._dpi_spin.setAccessibleName("Figure DPI (dots per inch)")
            form.addRow("DPI:", self._dpi_spin)

            return w

        def _build_typography_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)

            self._title_fs = QSpinBox()
            self._title_fs.setRange(6, 36)
            self._title_fs.setValue(self._style.title_fontsize)
            self._title_fs.setAccessibleName("Axes title font size")
            form.addRow("Title font size:", self._title_fs)

            self._axis_fs = QSpinBox()
            self._axis_fs.setRange(6, 30)
            self._axis_fs.setValue(self._style.axis_fontsize)
            self._axis_fs.setAccessibleName("Axis label font size")
            form.addRow("Axis label font size:", self._axis_fs)

            self._tick_fs = QSpinBox()
            self._tick_fs.setRange(5, 24)
            self._tick_fs.setValue(self._style.tick_fontsize)
            self._tick_fs.setAccessibleName("Tick label font size")
            form.addRow("Tick label font size:", self._tick_fs)

            self._legend_fs = QSpinBox()
            self._legend_fs.setRange(5, 24)
            self._legend_fs.setValue(self._style.legend_fontsize)
            self._legend_fs.setAccessibleName("Legend font size")
            form.addRow("Legend font size:", self._legend_fs)

            return w

        def _build_traces_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            layout.addWidget(QLabel("Line width (all traces):"))
            self._lw_spin = QDoubleSpinBox()
            self._lw_spin.setRange(0.3, 6.0)
            self._lw_spin.setSingleStep(0.1)
            self._lw_spin.setValue(self._style.linewidth)
            self._lw_spin.setAccessibleName("Global line width for all traces")
            layout.addWidget(self._lw_spin)

            layout.addWidget(QLabel("Per-trace color overrides:"))

            # Build color buttons for all lines in the figure
            self._trace_color_btns: dict[str, QPushButton] = {}
            axes = self._work_fig.get_axes()
            seen: set[str] = set()
            for ax in axes:
                for line in ax.get_lines():
                    lbl: str = str(line.get_label())
                    if not lbl or lbl.startswith("_") or lbl in seen:
                        continue
                    seen.add(lbl)
                    row = QHBoxLayout()
                    row_lbl = QLabel(lbl[:28])
                    row.addWidget(row_lbl, 1)
                    btn = QPushButton()
                    btn.setFixedSize(28, 20)
                    color = matplotlib.colors.to_hex(line.get_color())
                    btn.setStyleSheet(f"background-color: {color};")
                    btn.setAccessibleName(f"Color picker for trace {lbl}")
                    btn.clicked.connect(lambda _c=False, _l=lbl, _b=btn: self._pick_color(_l, _b))
                    row.addWidget(btn)
                    self._trace_color_btns[lbl] = btn
                    container = QWidget()
                    container.setLayout(row)
                    layout.addWidget(container)

            layout.addStretch()
            return w

        def _build_grid_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)

            self._grid_major_chk = QCheckBox()
            self._grid_major_chk.setChecked(self._style.grid_major)
            self._grid_major_chk.setAccessibleName("Show major grid lines")
            form.addRow("Major grid:", self._grid_major_chk)

            self._grid_minor_chk = QCheckBox()
            self._grid_minor_chk.setChecked(self._style.grid_minor)
            self._grid_minor_chk.setAccessibleName("Show minor grid lines")
            form.addRow("Minor grid:", self._grid_minor_chk)

            self._grid_alpha_spin = QDoubleSpinBox()
            self._grid_alpha_spin.setRange(0.05, 1.0)
            self._grid_alpha_spin.setSingleStep(0.05)
            self._grid_alpha_spin.setValue(self._style.grid_alpha)
            self._grid_alpha_spin.setAccessibleName("Grid transparency (alpha)")
            form.addRow("Grid alpha:", self._grid_alpha_spin)

            return w

        def _build_legend_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)

            self._legend_visible_chk = QCheckBox()
            self._legend_visible_chk.setChecked(self._style.legend_visible)
            self._legend_visible_chk.setAccessibleName("Show legend")
            form.addRow("Show legend:", self._legend_visible_chk)

            self._legend_loc_combo = QComboBox()
            locs = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            for loc in locs:
                self._legend_loc_combo.addItem(loc)
            self._legend_loc_combo.setCurrentText(self._style.legend_loc)
            self._legend_loc_combo.setAccessibleName("Legend location")
            form.addRow("Legend position:", self._legend_loc_combo)

            return w

        def _build_export_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)

            self._export_format_combo = QComboBox()
            for fmt in ["PNG", "PDF", "SVG", "EPS", "TIFF"]:
                self._export_format_combo.addItem(fmt)
            self._export_format_combo.setAccessibleName("Export file format")
            form.addRow("Format:", self._export_format_combo)

            self._export_dpi_spin = QSpinBox()
            self._export_dpi_spin.setRange(50, 1200)
            self._export_dpi_spin.setSingleStep(50)
            self._export_dpi_spin.setValue(self._style.dpi)
            self._export_dpi_spin.setAccessibleName("Export DPI override")
            form.addRow("Export DPI:", self._export_dpi_spin)

            note = QLabel(
                "Tip: use PNG/300+ dpi for reports; PDF/SVG for lossless vector graphics."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: gray; font-size: 8pt;")
            form.addRow(note)

            return w

        # ── Slots ─────────────────────────────────────────────────────────────

        def _collect_style(self) -> PlotStyle:
            """Read all form widgets into a :class:`PlotStyle` instance."""
            colors: dict[str, str] = {}
            for lbl, btn in self._trace_color_btns.items():
                bg = btn.styleSheet().replace("background-color: ", "").strip().rstrip(";")
                if bg:
                    colors[lbl] = bg

            return PlotStyle(
                figsize=(self._fig_w.value(), self._fig_h.value()),
                dpi=self._dpi_spin.value(),
                suptitle=self._suptitle_edit.text(),
                title_fontsize=self._title_fs.value(),
                axis_fontsize=self._axis_fs.value(),
                tick_fontsize=self._tick_fs.value(),
                legend_fontsize=self._legend_fs.value(),
                linewidth=self._lw_spin.value(),
                trace_colors=colors,
                grid_major=self._grid_major_chk.isChecked(),
                grid_minor=self._grid_minor_chk.isChecked(),
                grid_alpha=self._grid_alpha_spin.value(),
                legend_visible=self._legend_visible_chk.isChecked(),
                legend_loc=self._legend_loc_combo.currentText(),
                tight_layout=True,
            )

        def _apply_preview(self) -> None:
            """Re-render the work copy with current style and refresh the canvas."""
            self._style = self._collect_style()
            self._work_fig = copy.deepcopy(self._original_fig)
            PlotStyleApplicator.apply(self._work_fig, self._style)
            # Replace canvas figure
            self._canvas.figure = self._work_fig
            self._work_fig.set_canvas(self._canvas)
            self._canvas.draw()

        def _apply_preset(self) -> None:
            idx = self._preset_combo.currentIndex()
            if 0 <= idx < len(EXPORT_PRESETS):
                p = EXPORT_PRESETS[idx]
                self._fig_w.setValue(p.figsize[0])
                self._fig_h.setValue(p.figsize[1])
                self._dpi_spin.setValue(p.dpi)
                self._title_fs.setValue(p.title_fontsize)
                self._axis_fs.setValue(p.axis_fontsize)
                self._tick_fs.setValue(p.tick_fontsize)
                self._legend_fs.setValue(p.legend_fontsize)
                self._lw_spin.setValue(p.linewidth)
                self._export_dpi_spin.setValue(p.dpi)

        def _pick_color(self, label: str, btn: QPushButton) -> None:
            from PySide6.QtGui import QColor

            current_hex = (
                btn.styleSheet().replace("background-color: ", "").strip().rstrip(";") or "#1565C0"
            )
            color = QColorDialog.getColor(QColor(current_hex), self, f"Pick color for '{label}'")
            if color.isValid():
                hex_color = color.name()
                btn.setStyleSheet(f"background-color: {hex_color};")

        def _apply_to_original(self) -> None:
            """Apply current style directly to the original figure."""
            self._style = self._collect_style()
            PlotStyleApplicator.apply(self._original_fig, self._style)

        def _save_to_file(self) -> None:
            fmt = self._export_format_combo.currentText().lower()
            export_dpi = self._export_dpi_spin.value()
            path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save Figure",
                "",
                f"{fmt.upper()} Files (*.{fmt});;All Files (*)",
            )
            if not path_str:
                return
            path = Path(path_str)
            if not path.suffix:
                path = path.with_suffix(f".{fmt}")
            self._style = self._collect_style()
            self._work_fig = copy.deepcopy(self._original_fig)
            PlotStyleApplicator.apply(self._work_fig, self._style)
            PlotStyleApplicator.save(self._work_fig, path, dpi=export_dpi)
            plt.close(self._work_fig)

else:  # pragma: no cover - PySide6 always available in tests
    # Stub when Qt is not available (e.g. in unit-test environments)
    class PlotCustomizerDialog:  # type: ignore[no-redef]
        """Stub (Qt not available)."""

        def __init__(self, figure: Figure, parent: Any = None) -> None:  # noqa: ARG002
            self.figure = figure

        def exec(self) -> None:
            logger.warning("PlotCustomizerDialog: Qt not available.")
