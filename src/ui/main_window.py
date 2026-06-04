"""
Main GUI Window Module
======================

Main application window for BLDC motor control simulator.

Provides comprehensive GUI for:
- Motor parameter configuration
- Load profile definition
- Control and monitoring
- Real-time plotting
- Data export

:author: BLDC Control Team
:version: 0.12.0
"""

import csv
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QProcess, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QTextDocument,
)
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.control import BaseController, FOCController, SVMGenerator, VFController
from src.control.transforms import inverse_clarke
from src.core import (
    BLDCMotor,
    ConstantLoad,
    MotorParameters,
    RampLoad,
    SimulationEngine,
    recommend_efficiency_adjustments,
)
from src.hardware import InverterCurrentSense, MockDAQHardware, ShuntAmplifierChannel
from src.ui.widgets.accessible_widgets import (
    AccessibleButton,
    AccessibleGroupBox,
    AccessibleTableWidget,
    AccessibleTabWidget,
    LabeledComboBox,
    LabeledSpinBox,
)
from src.utils.config import (
    DEFAULT_LOAD_PROFILE,
    DEFAULT_MOTOR_PARAMS,
    FOC_FIELD_WEAKENING_PARAMS,
    FOC_STARTUP_PARAMS,
    MOTOR_PROFILES_DIR,
    SIMULATION_PARAMS,
    VF_CONTROLLER_PARAMS,
)
from src.utils.data_logger import DataLogger
from src.utils.motor_profiles import (
    list_motor_profiles,
    load_motor_profile,
    save_motor_profile,
)
from src.utils.speech import (
    is_audio_assistance_enabled,
    set_audio_assistance_enabled,
    speak,
)
from src.visualization.visualization import SimulationPlotter

# QAccessible was removed from PySide6.QtCore in some builds of PySide6.
# We still want the code to run even if accessibility support is not available.
QAccessible: Any = getattr(QtCore, "QAccessible", None)
if QAccessible is None:  # pragma: no cover

    class _DummyQAccessible:
        class Event:
            NameChange = 0
            ValueChange = 1

        @staticmethod
        def updateAccessibility(*args, **kwargs):
            return

    QAccessible = _DummyQAccessible

logger = logging.getLogger(__name__)


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert mixed config payload values to float for widget initialization."""
    try:
        if isinstance(value, (int, float, str)):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default


class AccessibleTextBlock(QFrame):
    """
    Accessible visual text block component for monitoring values.
    Each block displays a labeled monitoring value with units.
    Optimized for screen reader detection and visual distinction.
    """

    def __init__(
        self,
        parameter_name: str,
        unit: str,
        index: int,
        total_count: int,
        parent=None,
    ):
        super().__init__(parent)

        self.parameter_name = parameter_name
        self.unit = unit
        self.index = index
        self.total_count = total_count
        self.current_value = 0.0

        # Visual styling for text block
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setStyleSheet(
            """
            AccessibleTextBlock {
                background-color: #f0f0f0;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 8px;
                margin: 4px;
            }
            AccessibleTextBlock:focus {
                border: 2px solid #2196F3;
                background-color: #e8f4f8;
            }
        """
        )

        # Make focusable for keyboard navigation and screen readers
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)

        # Parameter name label (bold, prominent)
        self.name_label = QLabel(parameter_name)
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(10)
        self.name_label.setFont(name_font)
        self.name_label.setAccessibleName(f"Parameter {self.index + 1}: {parameter_name}")
        layout.addWidget(self.name_label)

        # Value display (large, readable)
        self.value_label = QLabel("-- " + unit)
        value_font = QFont()
        value_font.setPointSize(12)
        value_font.setBold(True)
        value_font.setFamily("Courier")  # Monospace for alignment
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet("color: #1976D2;")
        self.value_label.setAccessibleName(f"Value: {parameter_name} in {unit}")
        layout.addWidget(self.value_label)

        self.setLayout(layout)

        # Set accessible properties for screen readers
        self.setAccessibleName(
            f"Monitoring Block {self.index + 1} of {self.total_count}: {parameter_name}"
        )
        self.setAccessibleDescription(
            f"{parameter_name} current value (Unit: {unit}). Navigation: Tab to next block, Shift+Tab to previous."  # noqa: E501
        )

    def update_value(self, value: float):
        """Update the displayed value."""
        self.current_value = value
        formatted_value = f"{value:.4g}"
        self.value_label.setText(f"{formatted_value} {self.unit}")
        self.value_label.setToolTip(f"{self.parameter_name}: {formatted_value} {self.unit}")

        # Notify screen readers of value change
        QAccessible.updateAccessibility(self.value_label, 0, QAccessible.Event.NameChange)

    def keyPressEvent(self, event):
        """Handle keyboard navigation for accessibility."""
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            self.focusPreviousChild()
            event.accept()
        elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            self.focusNextChild()
            event.accept()
        else:
            super().keyPressEvent(event)


logger = logging.getLogger(__name__)


class SimulationThread(QThread):
    """Background simulation thread."""

    update_signal = Signal(dict)  # Emit current state
    finished_signal = Signal()

    def __init__(self):
        super().__init__()
        self.running = False
        self.engine: SimulationEngine | None = None
        self.svm: SVMGenerator | None = None
        # controller may be VFController or FOCController or any BaseController
        self.controller: BaseController | None = None
        self.update_interval = 0.1  # Update GUI every 100ms
        self.max_duration = 0.0  # 0 = infinite
        self._latest_state_lock = Lock()
        self._latest_state: dict = {}
        self._warning_once_keys: set[str] = set()
        # start flag will be set when set_simulation is called

    def _warn_once(self, key: str, message: str, exc: Exception | None = None) -> None:
        """Log a runtime warning only once per key to avoid flooding logs."""
        if key in self._warning_once_keys:
            return
        self._warning_once_keys.add(key)
        if exc is None:
            logger.warning(message)
            return
        logger.warning("%s: %s", message, exc)

    def get_latest_state(self) -> dict:
        """Return the latest simulation snapshot without blocking control loop."""
        with self._latest_state_lock:
            return dict(self._latest_state)

    def set_simulation(
        self,
        engine: SimulationEngine,
        svm: SVMGenerator,
        controller: BaseController,
        max_duration: float = 0.0,
        pwm_frequency_hz: float | None = None,
    ):
        """Assign engine, svm and controller before running thread."""
        self.engine = engine
        self.svm = svm
        self.controller = controller
        self.max_duration = max_duration
        self.svm.set_sample_time(engine.dt)
        resolved_pwm_hz = float(pwm_frequency_hz) if pwm_frequency_hz else (1.0 / engine.dt)
        self.engine.set_pwm_frequency(resolved_pwm_hz)

    def start_simulation(self):
        """Flag thread to begin executing the simulation loop."""
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_simulation(self):
        """Stop simulation loop."""
        self.running = False

    def run(self):  # noqa: C901  # TODO: extract step-dispatch logic into helper methods (16)
        """Main simulation loop."""
        if not self.engine or not self.svm or not self.controller:
            return

        dt = self.engine.dt
        control_period_s = self.engine.get_control_timing_state().get("control_period_s", dt)
        control_period_s = max(float(control_period_s), dt)
        next_control_time = self.engine.time
        last_voltages = np.zeros(3, dtype=np.float64)
        last_update = time.time()
        step_count = 0
        sim_start_time = self.engine.time

        while self.running:
            # Check if max duration reached (only if > 0)
            if self.max_duration > 0 and (self.engine.time - sim_start_time) >= self.max_duration:
                break

            # update supply voltage on svm from engine
            supply_v = self.engine.supply_profile.get_voltage(self.engine.time)
            if self.svm:
                try:
                    self.svm.set_dc_voltage(supply_v)
                except AttributeError as exc:
                    self._warn_once(
                        "svm_set_dc_voltage_missing",
                        "SVM object does not expose set_dc_voltage",
                        exc,
                    )
                try:
                    if hasattr(self.engine, "get_controller_phase_currents"):
                        phase_currents = self.engine.get_controller_phase_currents()
                    else:
                        phase_currents = self.engine.motor.currents
                    self.svm.set_phase_currents(phase_currents)
                except AttributeError as exc:
                    self._warn_once(
                        "svm_set_phase_currents_missing",
                        "SVM object does not expose set_phase_currents",
                        exc,
                    )

            if self.engine.time >= (next_control_time - 1e-15):
                calc_start = time.perf_counter()

                if isinstance(self.controller, FOCController) and hasattr(
                    self.engine, "get_controller_phase_currents"
                ):
                    try:
                        self.controller.set_external_phase_currents(
                            self.engine.get_controller_phase_currents()
                        )
                    except Exception as exc:
                        self._warn_once(
                            "foc_external_phase_currents_failed",
                            "Failed to update FOC external phase currents",
                            exc,
                        )

                # Get control output (could be polar or cartesian)
                ctrl_out = self.controller.update(control_period_s)
                voltages = None
                if isinstance(ctrl_out, tuple) and len(ctrl_out) == 2:
                    a, b = ctrl_out
                    # determine whether cartesian
                    if (
                        hasattr(self.controller, "output_cartesian")
                        and self.controller.output_cartesian
                    ):
                        # use cartesian modulate if available
                        try:
                            modulate_cartesian = getattr(self.svm, "modulate_cartesian", None)
                            if not callable(modulate_cartesian):
                                raise AttributeError(
                                    "SVM object does not expose modulate_cartesian"
                                )
                            voltages = modulate_cartesian(valpha=a, vbeta=b)
                        except Exception as exc:
                            # fallback to manual inverse clarke
                            self._warn_once(
                                "svm_modulate_cartesian_failed",
                                "Cartesian modulation failed; falling back to inverse Clarke",
                                exc,
                            )
                            va, vb, vc = inverse_clarke(a, b)
                            voltages = np.array([va, vb, vc])
                    else:
                        # polar
                        voltages = self.svm.modulate(a, b)
                else:
                    raise ValueError("Controller returned unexpected output format")

                calc_duration_s = time.perf_counter() - calc_start
                self.engine.record_control_timing(calc_duration_s, control_period_s)
                last_voltages = voltages
                next_control_time += control_period_s

            if self.engine is not None:
                self.engine.set_inverter_telemetry(self.svm.get_last_telemetry())
            # Execute motor step
            self.engine.step(last_voltages, log_data=True)

            step_count += 1

            # Periodic snapshot update for GUI polling.
            current_time = time.time()
            if (current_time - last_update) >= self.update_interval:
                state = self.engine.get_current_state()
                info = self.engine.get_simulation_info()
                with self._latest_state_lock:
                    self._latest_state = {**state, **info}
                last_update = current_time

            # No GUI operations are executed in this loop; keep the control
            # path free from GUI timing jitter.

        self.finished_signal.emit()


class CurrentSpectrumWindow(QMainWindow):
    """Dedicated FFT window for controller-facing current harmonics."""

    closed = Signal()

    def __init__(self, window_size_samples: int = 512, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Current Harmonic FFT")
        self.setMinimumSize(920, 700)

        self._window_size_samples = int(max(64, window_size_samples))
        self._time_samples: deque[float] = deque(maxlen=self._window_size_samples)
        self._ia_samples: deque[float] = deque(maxlen=self._window_size_samples)
        self._last_freq = np.array([], dtype=np.float64)
        self._last_mag = np.array([], dtype=np.float64)
        self._last_phase_deg = np.array([], dtype=np.float64)
        self._last_mag_display = np.array([], dtype=np.float64)
        self._last_phase_display = np.array([], dtype=np.float64)

        self._grid_enabled = True
        self._mag_x_scale = "linear"
        self._mag_y_scale = "linear"
        self._phase_x_scale = "linear"
        self._phase_y_scale = "linear"
        self._amplitude_mode = "linear"
        self._phase_unit = "deg"

        central = QWidget()
        layout = QVBoxLayout()

        self.summary_label = QLabel(
            "Awaiting simulation samples. FFT summary will report dominant frequency and THD."
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        controls_row = QHBoxLayout()
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.grid_checkbox)

        controls_row.addWidget(QLabel("Mag X"))
        self.mag_x_scale_combo = QComboBox()
        self.mag_x_scale_combo.addItems(["linear", "log"])
        self.mag_x_scale_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.mag_x_scale_combo)

        controls_row.addWidget(QLabel("Mag Y"))
        self.mag_y_scale_combo = QComboBox()
        self.mag_y_scale_combo.addItems(["linear", "log"])
        self.mag_y_scale_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.mag_y_scale_combo)

        controls_row.addWidget(QLabel("Phase X"))
        self.phase_x_scale_combo = QComboBox()
        self.phase_x_scale_combo.addItems(["linear", "log"])
        self.phase_x_scale_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.phase_x_scale_combo)

        controls_row.addWidget(QLabel("Phase Y"))
        self.phase_y_scale_combo = QComboBox()
        self.phase_y_scale_combo.addItems(["linear", "log"])
        self.phase_y_scale_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.phase_y_scale_combo)

        controls_row.addWidget(QLabel("Amplitude"))
        self.amplitude_mode_combo = QComboBox()
        self.amplitude_mode_combo.addItems(["linear", "dB"])
        self.amplitude_mode_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.amplitude_mode_combo)

        controls_row.addWidget(QLabel("Phase Unit"))
        self.phase_unit_combo = QComboBox()
        self.phase_unit_combo.addItems(["deg", "rad"])
        self.phase_unit_combo.currentTextChanged.connect(self._on_display_options_changed)
        controls_row.addWidget(self.phase_unit_combo)

        btn_save_csv = AccessibleButton(
            "Save FFT CSV",
            "Save FFT frequency, magnitude, and phase as a CSV file.",
        )
        btn_save_csv.clicked.connect(self._save_fft_csv)
        controls_row.addWidget(btn_save_csv)

        btn_save_image = AccessibleButton(
            "Save FFT Graph Image",
            "Save FFT magnitude and phase graphs as an image file.",
        )
        btn_save_image.clicked.connect(self._save_fft_image)
        controls_row.addWidget(btn_save_image)

        layout.addLayout(controls_row)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(8, 6), dpi=90)
        self.canvas = FigureCanvas(self.figure)
        self.ax_mag = self.figure.add_subplot(211)
        self.ax_phase = self.figure.add_subplot(212)
        self.ax_mag.set_title("Phase-A Current FFT Magnitude (Controller-Facing)")
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel("Magnitude (A)")
        self.ax_phase.set_title("Phase-A Current FFT Phase")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_mag.grid(True, alpha=0.3)
        self.ax_phase.grid(True, alpha=0.3)
        layout.addWidget(self.canvas, 1)

        central.setLayout(layout)
        self.setCentralWidget(central)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_plot)
        self._refresh_timer.start(250)

    def set_window_size(self, samples: int) -> None:
        """Update FFT window length and preserve newest samples when possible."""
        samples = int(max(64, samples))
        if samples == self._window_size_samples:
            return
        self._window_size_samples = samples
        self._time_samples = deque(list(self._time_samples)[-samples:], maxlen=samples)
        self._ia_samples = deque(list(self._ia_samples)[-samples:], maxlen=samples)

    def push_snapshot(self, snapshot: dict) -> None:
        """Consume latest simulation snapshot without blocking control loop."""
        if not isinstance(snapshot, dict):
            return
        try:
            time_s = float(snapshot.get("time", 0.0))
            ia = float(snapshot.get("currents_a", 0.0))
        except (TypeError, ValueError):
            return
        self._time_samples.append(time_s)
        self._ia_samples.append(ia)

    def apply_display_settings(
        self,
        grid_enabled: bool,
        mag_x_scale: str,
        mag_y_scale: str,
        phase_x_scale: str,
        phase_y_scale: str,
        amplitude_mode: str,
        phase_unit: str,
    ) -> None:
        """Apply FFT display configuration from parent GUI controls."""
        grid_value = bool(grid_enabled)
        mag_x_value = str(mag_x_scale).lower()
        mag_y_value = str(mag_y_scale).lower()
        phase_x_value = str(phase_x_scale).lower()
        phase_y_value = str(phase_y_scale).lower()
        amp_value = str(amplitude_mode).lower()
        phase_unit_value = str(phase_unit).lower()

        controls = [
            self.grid_checkbox,
            self.mag_x_scale_combo,
            self.mag_y_scale_combo,
            self.phase_x_scale_combo,
            self.phase_y_scale_combo,
            self.amplitude_mode_combo,
            self.phase_unit_combo,
        ]
        for ctrl in controls:
            ctrl.blockSignals(True)
        try:
            self.grid_checkbox.setChecked(grid_value)
            self.mag_x_scale_combo.setCurrentText(mag_x_value)
            self.mag_y_scale_combo.setCurrentText(mag_y_value)
            self.phase_x_scale_combo.setCurrentText(phase_x_value)
            self.phase_y_scale_combo.setCurrentText(phase_y_value)
            self.amplitude_mode_combo.setCurrentText("dB" if amp_value == "db" else "linear")
            self.phase_unit_combo.setCurrentText("rad" if phase_unit_value == "rad" else "deg")
        finally:
            for ctrl in controls:
                ctrl.blockSignals(False)

        self._grid_enabled = grid_value
        self._mag_x_scale = mag_x_value
        self._mag_y_scale = mag_y_value
        self._phase_x_scale = phase_x_value
        self._phase_y_scale = phase_y_value
        self._amplitude_mode = "db" if amp_value == "db" else "linear"
        self._phase_unit = "rad" if phase_unit_value == "rad" else "deg"

    def _on_display_options_changed(self, *_args) -> None:
        self._grid_enabled = self.grid_checkbox.isChecked()
        self._mag_x_scale = self.mag_x_scale_combo.currentText().lower()
        self._mag_y_scale = self.mag_y_scale_combo.currentText().lower()
        self._phase_x_scale = self.phase_x_scale_combo.currentText().lower()
        self._phase_y_scale = self.phase_y_scale_combo.currentText().lower()
        self._amplitude_mode = self.amplitude_mode_combo.currentText().lower()
        self._phase_unit = self.phase_unit_combo.currentText().lower()

    def _refresh_plot(self) -> None:  # pragma: no cover - GUI timer/rendering path
        """Compute and draw FFT from buffered samples in this window thread context."""
        sample_count = len(self._ia_samples)
        if sample_count < 16:
            return

        time_arr = np.asarray(self._time_samples, dtype=np.float64)
        current_arr = np.asarray(self._ia_samples, dtype=np.float64)

        dt = float(np.median(np.diff(time_arr))) if sample_count > 2 else 0.0
        if dt <= 0.0:
            return

        centered = current_arr - np.mean(current_arr)
        spec = np.fft.rfft(centered)
        freq = np.fft.rfftfreq(centered.size, d=dt)
        mag = np.abs(spec) * (2.0 / max(centered.size, 1))
        phase_deg = np.degrees(np.angle(spec))

        # Ignore DC when selecting dominant component.
        if mag.size > 1:
            dominant_idx = int(np.argmax(mag[1:]) + 1)
            fundamental_mag = float(max(mag[dominant_idx], 1e-12))
            harmonic_energy = float(np.sqrt(np.sum(np.square(mag[dominant_idx + 1 :]))))
            thd = (harmonic_energy / fundamental_mag) * 100.0
            dominant_freq = float(freq[dominant_idx])
        else:
            dominant_freq = 0.0
            thd = 0.0

        self._last_freq = np.asarray(freq, dtype=np.float64)
        self._last_mag = np.asarray(mag, dtype=np.float64)
        self._last_phase_deg = np.asarray(phase_deg, dtype=np.float64)

        if self._amplitude_mode == "db":
            mag_display = 20.0 * np.log10(np.maximum(mag, 1e-12))
            mag_label = "Magnitude (dB)"
        else:
            mag_display = mag
            mag_label = "Magnitude (A)"

        if self._phase_unit == "rad":
            phase_display = np.radians(phase_deg)
            phase_label = "Phase (rad)"
        else:
            phase_display = phase_deg
            phase_label = "Phase (deg)"

        self._last_mag_display = mag_display
        self._last_phase_display = phase_display

        self.ax_mag.clear()
        self.ax_phase.clear()

        freq_mag_plot = np.where(freq <= 0.0, 1e-9, freq)
        mag_plot = np.asarray(mag_display, dtype=np.float64)
        if self._mag_y_scale == "log":
            mag_plot = np.where(mag_plot <= 0.0, 1e-12, mag_plot)

        phase_plot = np.asarray(phase_display, dtype=np.float64)
        if self._phase_y_scale == "log":
            phase_plot = np.where(np.abs(phase_plot) <= 1e-9, 1e-9, np.abs(phase_plot))

        self.ax_mag.plot(freq_mag_plot, mag_plot, color="#1E88E5", linewidth=1.5)
        self.ax_phase.plot(freq_mag_plot, phase_plot, color="#D32F2F", linewidth=1.3)

        self.ax_mag.set_title("Phase-A Current FFT Magnitude (Controller-Facing)")
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel(mag_label)
        self.ax_phase.set_title("Phase-A Current FFT Phase")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel(phase_label)

        self.ax_mag.set_xscale(self._mag_x_scale)
        self.ax_mag.set_yscale(self._mag_y_scale)
        self.ax_phase.set_xscale(self._phase_x_scale)
        self.ax_phase.set_yscale(self._phase_y_scale)

        self.ax_mag.grid(self._grid_enabled, alpha=0.3)
        self.ax_phase.grid(self._grid_enabled, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.summary_label.setText(
            f"FFT window: {sample_count} samples, dominant frequency: {dominant_freq:.1f} Hz, THD: {thd:.2f}%"  # noqa: E501
        )

    def _save_fft_csv(self) -> None:  # pragma: no cover - GUI file dialog path
        """Save latest FFT spectrum arrays to CSV."""
        if self._last_freq.size == 0:
            QMessageBox.information(self, "No FFT Data", "Run the simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FFT Data",
            str(Path("data") / "logs" / "current_fft_data.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return

        file_path = Path(path)
        if file_path.suffix.lower() != ".csv":
            file_path = file_path.with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            mag_col = "magnitude_db" if self._amplitude_mode == "db" else "magnitude_a"
            phase_col = "phase_rad" if self._phase_unit == "rad" else "phase_deg"
            writer.writerow(["frequency_hz", mag_col, phase_col])
            for f_hz, mag_value, phase_value in zip(
                self._last_freq, self._last_mag_display, self._last_phase_display
            ):
                writer.writerow([float(f_hz), float(mag_value), float(phase_value)])

        QMessageBox.information(self, "FFT Data Saved", f"Saved: {file_path.name}")

    def _save_fft_image(self) -> None:  # pragma: no cover - GUI file dialog path
        """Save latest FFT graphs as an image."""
        if self._last_freq.size == 0:
            QMessageBox.information(self, "No FFT Data", "Run the simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FFT Graph",
            str(Path("data") / "plots" / "current_fft.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg)",
        )
        if not path:
            return

        file_path = Path(path)
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            file_path = file_path.with_suffix(".png")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(file_path, dpi=160)
        QMessageBox.information(self, "FFT Graph Saved", f"Saved: {file_path.name}")

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI teardown path
        self._refresh_timer.stop()
        self.closed.emit()
        super().closeEvent(event)


class BLDCMotorControlGUI(QMainWindow):
    """
    Main SPINOTOR GUI
    ===========================

    Comprehensive application for BLDC motor simulation and control.

    Features:
    - Screen reader accessible
    - Real-time motor simulation
    - V/f speed control
    - Data logging and export
    - Visualization tools
    """

    def __init__(self):
        super().__init__()

        APP_NAME = "BLIND SYSTEMS SPINOTOR"
        APP_VERSION = "0.12.0"

        self.setWindowTitle(f"{APP_NAME} - SPINOTOR (v{APP_VERSION})")
        self.setGeometry(100, 100, 1500, 950)

        # Accessibility
        self.setAccessibleName("BLIND SYSTEMS SPINOTOR")
        self.setAccessibleDescription(
            "Comprehensive BLDC motor simulator with V/f and FOC control. "
            "Use Tab to navigate between sections, arrow keys in list views."
        )

        # Simulation components
        self.motor: BLDCMotor | None = None
        self.engine: SimulationEngine | None = None
        self.svm: SVMGenerator | None = None
        self.controller: BaseController | None = None
        self.builtin_motor_menu: QMenu | None = None
        self.audio_assistance_action: QAction | None = None
        self.sim_thread: SimulationThread | None = None
        self.logger = DataLogger()

        # Legacy single-profile calibration backend kept for compatibility.
        # The UI now exposes only the unified Auto-Calibrate action.
        self.calib_process: QProcess | None = None
        self.calib_output_path: Path | None = None

        # Auto-calibrate-all process state (two-stage: FOC PI gains → FW loaded-point)
        self.auto_calib_process: QProcess | None = None
        self._auto_calib_stage: int = 0  # 0=idle, 1=step1_foc, 2=step2_fw

        # UI state
        self.is_running = False
        self._last_stability_severity_announced: str | None = None

        # Speed curve history for live plotting / CSV export compatibility
        self.speed_history_time: list[float] = []
        self.speed_history_rpm: list[float] = []

        # Real-time oscilloscope widget (created in _create_monitoring_tab)
        self.oscilloscope = None  # type: ignore[assignment]

        # Last generated post-sim figures — used by the Customize buttons
        self._last_fig_3phase = None
        self._last_fig_currents = None
        self._last_fig_pfc = None
        self._last_fig_efficiency = None
        self._last_fig_inverter = None
        self._last_fig_measured_vs_true = None
        self._last_fig_custom = None

        # Optional FFT analysis window for measured currents.
        self.current_fft_window: CurrentSpectrumWindow | None = None

        # Process lifecycle tracking (prevent simultaneous simulation/calibration)
        self._task_lock = Lock()  # Protects concurrent access to task state
        self._running_task_name: str | None = None  # simulation, calibration, or auto_calibration

        # Create UI
        self._create_ui()
        self._initialize_defaults()

        # Timer for GUI updates when not using thread
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._poll_simulation_state)

    def _create_ui(self):
        """Create main UI layout."""
        # Create menu bar
        self._create_menu_bar()
        # Central widget with permanent info panel (left: main UI, right: info)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Left column: title, tabs, buttons
        left_layout = QVBoxLayout()

        title = QLabel("BLIND SYSTEMS SPINOTOR - Advanced Motor Control")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAccessibleName("Application Title")
        left_layout.addWidget(title)

        # Tab widget
        self.tabs = AccessibleTabWidget()
        self._create_motor_drive_tab()
        self._create_controller_tab()
        self._create_observer_startup_tab()
        self._create_advanced_tab()
        self._create_analysis_tab()

        left_layout.addWidget(self.tabs, 1)

        # Control buttons
        button_layout = QHBoxLayout()

        self.btn_start = AccessibleButton(
            "Start Simulation (F5)",
            "Begin BLDC motor simulation with current parameters",
        )
        self.btn_start.setShortcut("F5")
        self.btn_start.clicked.connect(self._start_simulation)
        button_layout.addWidget(self.btn_start)

        self.btn_stop = AccessibleButton("Stop Simulation (F6)", "Stop running simulation")
        self.btn_stop.setShortcut("F6")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_simulation)
        button_layout.addWidget(self.btn_stop)

        self.btn_reset = AccessibleButton("Reset (F7)", "Reset simulation to initial state")
        self.btn_reset.setShortcut("F7")
        self.btn_reset.clicked.connect(self._reset_simulation)
        button_layout.addWidget(self.btn_reset)

        btn_export = AccessibleButton(
            "Export Data (Ctrl+S)", "Save simulation results to CSV and metadata"
        )
        btn_export.setShortcut("Ctrl+S")
        btn_export.clicked.connect(self._export_data)
        button_layout.addWidget(btn_export)

        button_layout.addStretch()
        left_layout.addLayout(button_layout)

        main_layout.addLayout(left_layout, 3)

        # Right column: permanent info panel
        info_group = AccessibleGroupBox("Quick Info", "Persistent simulation parameters and units")
        info_layout = QVBoxLayout()

        # dt and time constants
        self.lbl_dt = QLabel("dt: -- s")
        self.lbl_dt.setAccessibleName("Time step")
        info_layout.addWidget(self.lbl_dt)

        self.lbl_tau_e = QLabel("Electrical time constant (L/R): -- s")
        self.lbl_tau_e.setAccessibleName("Electrical time constant")
        info_layout.addWidget(self.lbl_tau_e)

        self.lbl_tau_m = QLabel("Mechanical time constant (J/b): -- s")
        self.lbl_tau_m.setAccessibleName("Mechanical time constant")
        info_layout.addWidget(self.lbl_tau_m)

        self.lbl_stability = QLabel("RK4 stability advisory: --")
        self.lbl_stability.setWordWrap(True)
        self.lbl_stability.setAccessibleName("RK4 stability advisory")
        self.lbl_stability.setStyleSheet("color: #455A64;")
        info_layout.addWidget(self.lbl_stability)

        # Parameter units summary
        self.lbl_param_units = QLabel()
        self.lbl_param_units.setWordWrap(True)
        self.lbl_param_units.setAccessibleName("Parameter units summary")
        info_layout.addWidget(self.lbl_param_units)

        # Ld / Lq quick entry note will be below parameters tab; show current values here
        self.lbl_ld_lq = QLabel("Ld: -- H, Lq: -- H")
        self.lbl_ld_lq.setAccessibleName("Ld and Lq values")
        info_layout.addWidget(self.lbl_ld_lq)

        info_layout.addStretch()
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group, 1)

        central_widget.setLayout(main_layout)

        # Add status bar with simulation parameters and telemetry
        self.status_bar_dt = QLabel("dt: -- s")
        self.status_bar_dt.setAccessibleName("Simulation time step")
        self.status_bar_dt.setAccessibleDescription("Integration step size in seconds.")
        self.status_bar_tau_e = QLabel("τ_e: -- s")
        self.status_bar_tau_e.setAccessibleName("Electrical time constant")
        self.status_bar_tau_e.setAccessibleDescription(
            "Motor electrical time constant L/R in seconds."
        )
        self.status_bar_tau_m = QLabel("τ_m: -- s")
        self.status_bar_tau_m.setAccessibleName("Mechanical time constant")
        self.status_bar_tau_m.setAccessibleDescription(
            "Motor mechanical time constant J/b in seconds."
        )
        self.status_bar_stability = QLabel("RK4: --")
        self.status_bar_stability.setStyleSheet("color: #455A64;")
        self.status_bar_stability.setAccessibleName("RK4 stability advisory")
        self.status_bar_stability.setAccessibleDescription(
            "Warns when the simulation time step may be too large for stable integration."
        )
        self.status_bar_state = QLabel("State: Ready")
        self.status_bar_state.setAccessibleName("Simulation state")
        self.status_bar_state.setAccessibleDescription(
            "Current simulation state: Ready, Running, or Stopped."
        )
        self.status_bar_time_remaining = QLabel("Remaining: -- s")
        self.status_bar_time_remaining.setAccessibleName("Estimated time remaining")
        self.status_bar_time_remaining.setAccessibleDescription(
            "Estimated seconds remaining to complete the simulation."
        )
        self.status_bar_cpu_load = QLabel("CPU: -- %")
        self.status_bar_cpu_load.setAccessibleName("CPU load")
        self.status_bar_cpu_load.setAccessibleDescription(
            "Percentage of one CPU core consumed by the simulation thread."
        )
        self.status_bar_task = QLabel("Task: None")
        self.status_bar_task.setAccessibleName("Active task")
        self.status_bar_task.setAccessibleDescription(
            "Name of the currently running background task."
        )
        self.status_bar_backend = QLabel("Backend: --")
        self.status_bar_backend.setAccessibleName("Compute backend")
        self.status_bar_backend.setAccessibleDescription(
            "Active computation backend, e.g. NumPy or CuPy."
        )

        status_bar = self.statusBar()
        assert status_bar is not None
        status_bar.addWidget(self.status_bar_state)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_task)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_backend)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_time_remaining)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_cpu_load)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_dt)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_tau_e)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_tau_m)
        status_bar.addWidget(QLabel("|"))  # Separator
        status_bar.addWidget(self.status_bar_stability)

    def _create_menu_bar(self):
        """Create application menu bar with File, Option, and Help menus."""
        menubar = self.menuBar()
        assert menubar is not None

        # File Menu (mnemonic Alt+F)
        file_menu = menubar.addMenu("&File")
        assert file_menu is not None

        export_action = file_menu.addAction("&Export Simulation Data")
        assert export_action is not None
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self._export_data)

        file_menu.addSeparator()

        import_motor_action = file_menu.addAction("&Import Motor Parameters...")
        assert import_motor_action is not None
        import_motor_action.triggered.connect(self._import_motor_parameters)

        save_motor_action = file_menu.addAction("&Save Motor Parameters...")
        assert save_motor_action is not None
        save_motor_action.triggered.connect(self._save_motor_parameters)

        save_sim_params_action = file_menu.addAction("Save &Simulation Parameters...")
        assert save_sim_params_action is not None
        save_sim_params_action.triggered.connect(self._save_simulation_parameters)

        self.builtin_motor_menu = file_menu.addMenu("Load Built-in Motor Profile")
        assert self.builtin_motor_menu is not None
        self._refresh_builtin_motor_menu()

        file_menu.addSeparator()

        params_action = file_menu.addAction("&Simulation Parameters")
        assert params_action is not None
        params_action.triggered.connect(self._show_simulation_params)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("&Quit")
        assert quit_action is not None
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        # Option Menu (mnemonic Alt+O)
        option_menu = menubar.addMenu("&Option")
        assert option_menu is not None
        reset_action = option_menu.addAction("Reset Simulation (F7)")
        assert reset_action is not None
        reset_action.setShortcut("F7")
        reset_action.triggered.connect(self._reset_simulation)

        self.audio_assistance_action = option_menu.addAction("Audio Assistance Enabled")
        assert self.audio_assistance_action is not None
        self.audio_assistance_action.setCheckable(True)
        self.audio_assistance_action.setChecked(is_audio_assistance_enabled())
        self.audio_assistance_action.triggered.connect(self._toggle_audio_assistance)

        # Help Menu (mnemonic Alt+H)
        help_menu = menubar.addMenu("&Help")
        assert help_menu is not None
        html_help_action = help_menu.addAction("Open HTML Help (Sphinx)")
        assert html_help_action is not None
        html_help_action.triggered.connect(self._open_html_help)

        pdf_manual_action = help_menu.addAction("Open PDF User Manual")
        assert pdf_manual_action is not None
        pdf_manual_action.triggered.connect(self._open_user_manual_pdf)

        help_menu.addSeparator()
        about_action = help_menu.addAction("About")
        assert about_action is not None
        about_action.triggered.connect(self._show_about)

    def _refresh_builtin_motor_menu(self):
        """Populate built-in motor profile submenu from data/motor_profiles."""
        assert self.builtin_motor_menu is not None
        self.builtin_motor_menu.clear()
        profile_paths = list_motor_profiles(MOTOR_PROFILES_DIR)
        if not profile_paths:
            empty_action = self.builtin_motor_menu.addAction("No built-in profiles found")
            assert empty_action is not None
            empty_action.setEnabled(False)
            return

        for profile_path in profile_paths:
            action = self.builtin_motor_menu.addAction(profile_path.stem)
            assert action is not None
            action.triggered.connect(
                lambda checked=False, p=profile_path: self._load_motor_profile_from_path(p)
            )

    def _collect_current_motor_parameters(self) -> dict:
        """Collect current motor parameter values from UI widgets."""
        return {
            "nominal_voltage": float(self.param_voltage.value()),
            "phase_resistance": float(self.param_resistance.value()),
            "phase_inductance": float(self.param_inductance.value()),
            "back_emf_constant": float(self.param_emf.value()),
            "torque_constant": float(self.param_kt.value()),
            "rotor_inertia": float(self.param_inertia.value()),
            "friction_coefficient": float(self.param_friction.value()),
            "num_poles": int(self.param_poles.value()),
            "ld": float(self.param_ld.value()),
            "lq": float(self.param_lq.value()),
            "model_type": self.param_model_type.currentText(),
            "emf_shape": self.param_emf_shape.currentText(),
        }

    def _apply_motor_profile(self, profile: dict):
        """Apply a loaded motor profile to the UI controls."""
        params = profile["motor_params"]
        self.param_voltage.setValue(float(params["nominal_voltage"]))
        self.param_resistance.setValue(float(params["phase_resistance"]))
        self.param_inductance.setValue(float(params["phase_inductance"]))
        self.param_emf.setValue(float(params["back_emf_constant"]))
        self.param_kt.setValue(float(params["torque_constant"]))
        self.param_inertia.setValue(float(params["rotor_inertia"]))
        self.param_friction.setValue(float(params["friction_coefficient"]))
        self.param_poles.setValue(int(params["num_poles"]))
        self.param_ld.setValue(float(params.get("ld", params["phase_inductance"])))
        self.param_lq.setValue(float(params.get("lq", params["phase_inductance"])))
        self.param_model_type.setCurrentText(params.get("model_type", "dq"))
        self.param_emf_shape.setCurrentText(params.get("emf_shape", "sinusoidal"))

    def _import_motor_parameters(self):
        """Import motor parameters from a JSON profile file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Motor Parameters",
            str(MOTOR_PROFILES_DIR),
            "JSON Files (*.json)",
        )
        if not filename:
            return

        try:
            profile = load_motor_profile(Path(filename))
            self._apply_motor_profile(profile)
            self._apply_to_simulation()
            QMessageBox.information(
                self,
                "Motor Parameters Imported",
                f"Loaded profile: {profile['profile_name']}",
            )
            speak("Motor parameters imported successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Failed to import profile: {exc}")

    def _save_motor_parameters(self):
        """Save current motor parameters to a JSON profile file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Motor Parameters",
            str(MOTOR_PROFILES_DIR / "custom_motor_profile.json"),
            "JSON Files (*.json)",
        )
        if not filename:
            return

        file_path = Path(filename)
        if file_path.suffix.lower() != ".json":
            file_path = file_path.with_suffix(".json")

        profile_name = file_path.stem
        rated_info = {
            "rated_voltage_v": float(self.param_voltage.value()),
            "pole_pairs": int(self.param_poles.value() / 2),
        }
        source_info = {
            "origin": "user_saved_profile",
        }

        try:
            save_motor_profile(
                file_path=file_path,
                motor_params=self._collect_current_motor_parameters(),
                profile_name=profile_name,
                rated_info=rated_info,
                source_info=source_info,
            )
            self._refresh_builtin_motor_menu()
            QMessageBox.information(
                self,
                "Motor Parameters Saved",
                f"Saved profile: {file_path.name}",
            )
            speak("Motor parameters saved successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save profile: {exc}")

    def _load_motor_profile_from_path(self, profile_path: Path):
        """Load a built-in motor profile by path."""
        try:
            profile = load_motor_profile(profile_path)
            self._apply_motor_profile(profile)
            self._apply_to_simulation()
            QMessageBox.information(
                self,
                "Built-in Motor Profile Loaded",
                f"Loaded profile: {profile['profile_name']}",
            )
            speak("Built-in motor profile loaded.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load profile: {exc}")

    def _toggle_audio_assistance(self, enabled: bool):
        """Enable or disable spoken assistance from the option menu."""
        set_audio_assistance_enabled(bool(enabled))
        state = "enabled" if enabled else "disabled"
        QMessageBox.information(
            self,
            "Audio Assistance",
            f"Audio assistance is now {state}.",
        )
        speak(f"Audio assistance {state}.")

    def _open_html_help(self):
        """Open generated Sphinx HTML documentation index."""
        html_index = Path(__file__).resolve().parents[2] / "docs" / "_build" / "html" / "index.html"
        if not html_index.exists():
            QMessageBox.warning(
                self,
                "HTML Help Not Found",
                "Sphinx HTML help was not found at docs/_build/html/index.html.\n"
                "Generate it with: sphinx-build -b html docs docs/_build/html",
            )
            speak("HTML help not found. Please build Sphinx documentation first.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_index)))
        speak("HTML help opened.")

    def _ensure_user_manual_pdf(self) -> Path:
        """Generate a PDF user manual when missing and return its path."""
        project_root = Path(__file__).resolve().parents[2]
        pdf_path = project_root / "docs" / "BLIND_SYSTEMS_User_Manual.pdf"
        if pdf_path.exists():
            return pdf_path

        html = (
            "<h1>BLIND SYSTEMS SPINOTOR - User Manual</h1>"
            "<p><b>Version:</b> 0.12.0</p>"
            "<p><b>Author:</b> Amine Khettat</p>"
            "<h2>1. Getting Started</h2>"
            "<p>Configure motor, load and controller parameters, then start simulation.</p>"
            "<h2>2. Accessibility</h2>"
            "<p>Use Option -> Audio Assistance Enabled to toggle speech output.</p>"
            "<h2>3. Running and Monitoring</h2>"
            "<p>Start: F5, Stop: F6, Reset: F7. Monitor speed, torque, convergence and CPU load metrics.</p>"  # noqa: E501
            "<h2>4. Data Export</h2>"
            "<p>Use File -> Export Simulation Data or Ctrl+S.</p>"
            "<h2>5. Help</h2>"
            "<p>Open HTML help from Sphinx or this PDF from the Help menu.</p>"
            "<hr/>"
            "<p>Copyright 2026 BLIND SYSTEMS</p>"
        )

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(str(pdf_path))
        document = QTextDocument()
        document.setHtml(html)
        document.print_(printer)
        return pdf_path

    def _open_user_manual_pdf(self):
        """Open generated PDF user manual, generating it if required."""
        pdf_path = self._ensure_user_manual_pdf()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))
        speak("PDF user manual opened.")

    def _show_simulation_params(self):
        """Show current simulation time step and motor time constants."""
        msg = self.get_simulation_params_info()
        QMessageBox.information(self, "Simulation Parameters", msg)
        speak("Simulation parameters displayed.")

    def get_simulation_params_info(self) -> str:
        """Return a string with current dt and motor time constants (for testing)."""
        # Determine dt
        if self.engine:
            dt_val = self.engine.dt
            motor_params = self.engine.motor.params
        else:
            from src.core.motor_model import MotorParameters

            dt_val = _as_float(SIMULATION_PARAMS.get("dt", 0.0001), 0.0001)
            motor_params = MotorParameters()

        # Electrical time constant (L/R) and mechanical (J/b)
        try:
            tau_e = motor_params.phase_inductance / motor_params.phase_resistance
        except Exception:
            tau_e = None

        try:
            tau_m = motor_params.rotor_inertia / motor_params.friction_coefficient
        except Exception:
            tau_m = None

        pwm_hz = 1.0 / dt_val if dt_val > 0.0 else 0.0
        stability = self._build_dt_pwm_stability_advisory(dt_val, motor_params, pwm_hz)
        msg = f"Simulation time step (dt): {dt_val} s\n"
        msg += f"PWM frequency: {pwm_hz:.1f} Hz\n"
        if tau_e is not None:
            msg += f"Electrical time constant (L/R): {tau_e:.6f} s\n"
        else:
            msg += "Electrical time constant (L/R): n/a\n"
        if tau_m is not None:
            msg += f"Mechanical time constant (J/b): {tau_m:.6f} s\n"
        else:
            msg += "Mechanical time constant (J/b): n/a\n"
        msg += f"RK4 advisory: {stability['severity']}\n"
        msg += f"Recommended dt <= {stability['dt_recommended_s']:.3e} s\n"
        msg += f"Recommended PWM >= {stability['pwm_recommended_min_hz']:.1f} Hz\n"
        msg += "Recommended action: adjust dt or PWM frequency (motor parameters are fixed).\n"
        msg += str(stability["message"])

        return msg

    def _build_dt_pwm_stability_advisory(
        self, dt_s: float, motor_params: MotorParameters, pwm_hz: float
    ) -> dict[str, float | str]:
        """Compute RK4 dt/PWM advisory for current motor constants.

        User-facing guidance is restricted to simulation settings: dt and PWM frequency.
        """
        rk4_limit = 2.785
        recommended_margin = 5.0

        try:
            tau_e = float(motor_params.phase_inductance) / float(motor_params.phase_resistance)
        except Exception:
            tau_e = float("inf")
        try:
            tau_m = float(motor_params.rotor_inertia) / float(motor_params.friction_coefficient)
        except Exception:
            tau_m = float("inf")

        dt_limit_e = rk4_limit * tau_e if np.isfinite(tau_e) else float("inf")
        dt_limit_m = rk4_limit * tau_m if np.isfinite(tau_m) else float("inf")
        dt_limit = min(dt_limit_e, dt_limit_m)

        if np.isfinite(dt_limit) and dt_limit > 0.0:
            dt_recommended = dt_limit / recommended_margin
            pwm_recommended_min_hz = 1.0 / dt_recommended
            margin = dt_limit / dt_s if dt_s > 0.0 else float("inf")
            if dt_s > dt_limit:
                severity = "unstable"
            elif dt_s > 0.5 * dt_limit:
                severity = "marginal"
            else:
                severity = "stable"
        else:
            dt_recommended = float("inf")
            pwm_recommended_min_hz = 0.0
            margin = float("inf")
            severity = "unknown"

        if severity == "unstable":
            message = (
                f"dt={dt_s:.3e} s exceeds RK4 limit {dt_limit:.3e} s. "
                f"Decrease dt to <= {dt_recommended:.3e} s or raise PWM to >= "
                f"{pwm_recommended_min_hz:.1f} Hz."
            )
        elif severity == "marginal":
            message = (
                f"RK4 margin is low ({margin:.2f}x). Decrease dt toward <= "
                f"{dt_recommended:.3e} s or raise PWM toward >= {pwm_recommended_min_hz:.1f} Hz."
            )
        elif severity == "stable":
            message = f"RK4 margin is healthy ({margin:.2f}x)."
        else:
            message = "RK4 advisory unavailable due to missing positive R/L or J/b pair."

        return {
            "severity": severity,
            "dt_limit_s": float(dt_limit) if np.isfinite(dt_limit) else float("inf"),
            "dt_recommended_s": float(dt_recommended)
            if np.isfinite(dt_recommended)
            else float("inf"),
            "pwm_recommended_min_hz": float(pwm_recommended_min_hz),
            "stability_margin": float(margin) if np.isfinite(margin) else float("inf"),
            "message": message,
            "pwm_hz": float(pwm_hz),
        }

    def _announce_stability_advisory_if_needed(
        self,
        severity: str,
        dt_recommended_s: float,
        pwm_recommended_min_hz: float,
        margin: float,
    ) -> None:
        """Speak RK4 advisory changes when audio assistance is enabled.

        Announcements are edge-triggered by severity transitions to avoid
        repeating speech every UI refresh.
        """
        if not is_audio_assistance_enabled():
            return

        if severity == self._last_stability_severity_announced:
            return

        self._last_stability_severity_announced = severity
        if severity == "unstable":
            speak(
                "Warning. RK4 stability is unstable. "
                f"Decrease time step to at most {dt_recommended_s:.2e} seconds, "
                f"or increase PWM frequency to at least {pwm_recommended_min_hz:.0f} hertz."
            )
        elif severity == "marginal":
            speak(
                "Caution. RK4 stability margin is low. "
                f"Current margin is {margin:.1f} times. "
                f"Consider decreasing time step to {dt_recommended_s:.2e} seconds, "
                f"or increasing PWM frequency to {pwm_recommended_min_hz:.0f} hertz."
            )
        elif severity == "stable":
            speak("RK4 stability is now healthy.")
        else:
            speak("RK4 stability advisory is currently unavailable.")

    def _collect_simulation_configuration(self) -> dict:
        """Collect complete simulation configuration from current UI state."""
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        switching_frequency_hz = float(self.inverter_switching_frequency.value())
        dt_s = 1.0 / switching_frequency_hz if switching_frequency_hz > 0.0 else None

        config = {
            "schema": "bldc.simulation_config.v1",
            "created_utc": now_utc,
            "control_mode": self.ctrl_mode.currentText(),
            "simulation": {
                "duration_s": float(self.sim_duration.value()),
                "switching_frequency_hz": switching_frequency_hz,
                "dt_s": dt_s,
                "dc_voltage_v": float(self.supply_constant_voltage.value())
                if hasattr(self, "supply_constant_voltage")
                else _as_float(SIMULATION_PARAMS.get("dc_voltage", 48.0), 48.0),
            },
            "motor_params": {
                "nominal_voltage": float(self.param_voltage.value()),
                "phase_resistance": float(self.param_resistance.value()),
                "phase_inductance": float(self.param_inductance.value()),
                "back_emf_constant": float(self.param_emf.value()),
                "torque_constant": float(self.param_kt.value()),
                "rotor_inertia": float(self.param_inertia.value()),
                "friction_coefficient": float(self.param_friction.value()),
                "num_poles": int(self.param_poles.value()),
                "pole_pairs": int(self.param_poles.value() / 2),
                "ld": float(self.param_ld.value()),
                "lq": float(self.param_lq.value()),
                "model_type": self.param_model_type.currentText(),
                "emf_shape": self.param_emf_shape.currentText(),
            },
            "load_profile": {
                "type": self.load_type.currentText(),
                "constant_torque_nm": float(self.load_constant_torque.value()),
                "ramp_initial_torque_nm": float(self.load_initial_torque.value()),
                "ramp_final_torque_nm": float(self.load_final_torque.value()),
                "ramp_duration_s": float(self.load_ramp_duration.value()),
            },
            "supply_profile": {
                "type": self.supply_type.currentText()
                if hasattr(self, "supply_type")
                else "Constant",
                "constant_voltage_v": float(self.supply_constant_voltage.value())
                if hasattr(self, "supply_constant_voltage")
                else _as_float(SIMULATION_PARAMS.get("dc_voltage", 48.0), 48.0),
                "ramp_initial_v": float(self.supply_ramp_initial.value())
                if hasattr(self, "supply_ramp_initial")
                else 0.0,
                "ramp_final_v": float(self.supply_ramp_final.value())
                if hasattr(self, "supply_ramp_final")
                else 0.0,
                "ramp_duration_s": float(self.supply_ramp_duration.value())
                if hasattr(self, "supply_ramp_duration")
                else 0.0,
            },
            "vf_controller": {
                "v_nominal": float(self.vf_v_nominal.value()),
                "f_nominal": float(self.vf_f_nominal.value()),
                "speed_ref_rpm": float(self.vf_speed_ref.value()),
                "startup_voltage_v": float(self.vf_startup_voltage.value()),
                "frequency_slew_hz_per_s": float(self.vf_freq_slew.value()),
                "startup_sequence_enabled": self.vf_startup_sequence_mode.currentText()
                == "Enabled",
                "align_duration_s": float(self.vf_align_time.value()),
                "align_voltage_v": float(self.vf_align_voltage.value()),
                "align_angle_deg": float(self.vf_align_angle.value()),
                "ramp_initial_frequency_hz": float(self.vf_ramp_initial_frequency.value()),
            },
            "foc_controller": {
                "transform": self.foc_transform.currentText(),
                "output_mode": self.foc_output_mode.currentText(),
                "current_feedback_source": self.foc_current_feedback_source.currentText(),
                "speed_loop_mode": self.foc_speed_loop_mode.currentText(),
                "id_ref_a": float(self.foc_id_ref.value()),
                "iq_ref_a": float(self.foc_iq_ref.value()),
                "speed_ref_rpm": float(self.foc_speed_ref.value()),
                "iq_limit_a": float(self.foc_iq_limit.value()),
                "speed_pi": {
                    "kp": float(self.foc_speed_kp.value()),
                    "ki": float(self.foc_speed_ki.value()),
                },
                "current_pi": {
                    "d_kp": float(self.foc_d_kp.value()),
                    "d_ki": float(self.foc_d_ki.value()),
                    "q_kp": float(self.foc_q_kp.value()),
                    "q_ki": float(self.foc_q_ki.value()),
                },
                "decoupling": {
                    "enable_d": self.foc_decouple_d_mode.currentText() == "Enabled",
                    "enable_q": self.foc_decouple_q_mode.currentText() == "Enabled",
                },
                "observer": {
                    "mode": self.foc_angle_observer_mode.currentText(),
                    "solver": self.foc_solver_mode.currentText(),
                    "pll_kp": float(self.foc_pll_kp.value()),
                    "pll_ki": float(self.foc_pll_ki.value()),
                    "smo_k_slide": float(self.foc_smo_k_slide.value()),
                    "smo_lpf_alpha": float(self.foc_smo_lpf_alpha.value()),
                    "smo_boundary": float(self.foc_smo_boundary.value()),
                    "stsmo_k1": float(self.foc_stsmo_k1.value()),
                    "stsmo_k2_min": float(self.foc_stsmo_k2_min.value()),
                    "stsmo_k2_factor": float(self.foc_stsmo_k2_factor.value()),
                    "stsmo_rated_rpm": float(self.foc_stsmo_rated_rpm.value()),
                    "af_dc_cutoff_hz": float(self.foc_af_dc_cutoff.value()),
                },
                "startup_sequence": {
                    "enabled": self.foc_startup_sequence_mode.currentText() == "Enabled",
                    "align_duration_s": float(self.foc_align_time.value()),
                    "align_current_a": float(self.foc_align_current.value()),
                    "align_angle_deg": float(self.foc_align_angle.value()),
                    "open_loop_initial_speed_rpm": float(self.foc_open_loop_initial_speed.value()),
                    "open_loop_target_speed_rpm": float(self.foc_open_loop_target_speed.value()),
                    "open_loop_ramp_time_s": float(self.foc_open_loop_ramp_time.value()),
                    "open_loop_id_ref_a": float(self.foc_open_loop_id_ref.value()),
                    "open_loop_iq_ref_a": float(self.foc_open_loop_iq_ref.value()),
                },
                "startup_transition": {
                    "enabled": self.foc_startup_transition_mode.currentText() == "Enabled",
                    "initial_mode": self.foc_startup_initial_observer.currentText(),
                    "min_speed_rpm": float(self.foc_startup_min_speed.value()),
                    "min_elapsed_s": float(self.foc_startup_min_time.value()),
                    "min_emf_v": float(self.foc_startup_min_emf.value()),
                    "min_confidence": float(self.foc_startup_min_confidence.value()),
                    "confidence_hold_s": float(self.foc_startup_confidence_hold.value()),
                    "confidence_hysteresis": float(self.foc_startup_confidence_hysteresis.value()),
                    "fallback_enabled": self.foc_startup_fallback_mode.currentText() == "Enabled",
                    "fallback_hold_s": float(self.foc_startup_fallback_hold.value()),
                },
                "field_weakening": {
                    "enabled": self.foc_field_weakening_mode.currentText() == "Enabled",
                    "start_speed_rpm": float(self.foc_field_weakening_start_speed.value()),
                    "gain": float(self.foc_field_weakening_gain.value()),
                    "max_negative_id_a": float(self.foc_field_weakening_max_id.value()),
                    "headroom_target_v": float(self.foc_field_weakening_headroom_target.value()),
                },
            },
            "pfc": {
                "enabled": self.pfc_mode.currentText() == "Enabled",
                "target_pf": float(self.pfc_target_pf.value()),
                "kp": float(self.pfc_kp.value()),
                "ki": float(self.pfc_ki.value()),
                "max_compensation_var": float(self.pfc_max_var.value()),
                "window_samples": int(self.pfc_window_samples.value()),
            },
            "inverter_params": self.svm.get_realism_state() if self.svm else {},
            "communication_params": {
                "enabled": bool(
                    hasattr(self, "hardware_enable_backend")
                    and self.hardware_enable_backend.isChecked()
                ),
                "backend": self.hardware_backend_type.currentText()
                if hasattr(self, "hardware_backend_type")
                else "none",
                "mock_noise_std": float(self.hardware_noise_std.value())
                if hasattr(self, "hardware_noise_std")
                else 0.0,
                "mock_seed": int(self.hardware_seed.value())
                if hasattr(self, "hardware_seed")
                else 0,
            },
            "current_measurement": {
                "enabled": bool(self.current_sense_enable.isChecked()),
                "topology": self.current_sense_topology.currentText(),
                "shunt_resistance_ohm": float(self.current_sense_r_shunt.value()),
                "nominal_gain": float(self.current_sense_nominal_gain.value()),
                "nominal_offset_v": float(self.current_sense_nominal_offset.value()),
                "fft_window_samples": int(self.current_sense_fft_window_samples.value()),
                "fft_show_grid": bool(self.current_sense_fft_show_grid.isChecked()),
                "fft_mag_x_scale": self.current_sense_fft_mag_x_scale.currentText(),
                "fft_mag_y_scale": self.current_sense_fft_mag_y_scale.currentText(),
                "fft_phase_x_scale": self.current_sense_fft_phase_x_scale.currentText(),
                "fft_phase_y_scale": self.current_sense_fft_phase_y_scale.currentText(),
                "fft_amplitude_mode": self.current_sense_fft_amplitude_mode.currentText(),
                "fft_phase_unit": self.current_sense_fft_phase_unit.currentText(),
            },
        }

        return config

    def _save_simulation_parameters(self):
        """Save full simulation configuration as JSON."""
        default_path = Path("data") / "logs" / "simulation_parameters.json"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Parameters",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not filename:
            return

        file_path = Path(filename)
        if file_path.suffix.lower() != ".json":
            file_path = file_path.with_suffix(".json")

        try:
            payload = self._collect_simulation_configuration()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            QMessageBox.information(
                self,
                "Simulation Parameters Saved",
                f"Saved configuration: {file_path.name}",
            )
            speak("Simulation parameters saved successfully.")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save simulation parameters: {exc}",
            )

    def _show_about(self):
        """Show about dialog."""
        about_text = (
            "<h2>BLIND SYSTEMS SPINOTOR v0.12.0</h2>"
            "<p><b>Advanced SPINOTOR</b></p>"
            "<p><b>Author:</b> Amine Khettat</p>"
            "<p><b>Copyright:</b> 2026 BLIND SYSTEMS</p>"
            "<p>Includes accessible audio assistance and real-time control diagnostics.</p>"
        )

        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(about_text)

        project_root = Path(__file__).resolve().parents[2]
        candidate_logos = [
            project_root / "docs" / "logo.png",
            project_root / "data" / "logo.png",
        ]
        logo_path = next((p for p in candidate_logos if p.exists()), None)
        if logo_path is not None:
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                msg.setIconPixmap(pix.scaled(220, 80, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            # Fallback synthetic logo if no file is present yet.
            pix = QPixmap(240, 70)
            pix.fill(QColor("white"))
            painter = QPainter(pix)
            painter.setPen(QPen(QColor("#0A0A0A"), 2))
            painter.drawRect(1, 1, 238, 68)
            painter.setPen(QPen(QColor("#0A3D91"), 2))
            painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "BLIND SYSTEMS")
            painter.end()
            msg.setIconPixmap(pix)

        msg.exec()

    def _show_guide(self):
        """Show quick start guide."""
        guide_text = (
            "<h3>Quick Start Guide - BLIND SYSTEMS SPINOTOR</h3>"
            "<p><b>1. Configure Motor Parameters:</b></p>"
            "<ul><li>Set nominal voltage, resistance, inductance, pole pairs, Back-EMF constant</li></ul>"  # noqa: E501
            "<p><b>2. Set Load Profile:</b></p>"
            "<ul><li>Choose load type (Constant, Ramp, Inertial) and configure parameters</li></ul>"
            "<p><b>3. Setup Supply Profile (Optional):</b></p>"
            "<ul><li>Select supply type and voltage variation parameters</li></ul>"
            "<p><b>4. Select Control Mode:</b></p>"
            "<ul><li>Choose V/f or FOC control, configure parameters</li></ul>"
            "<p><b>5. Set Simulation Duration:</b></p>"
            "<ul><li>Enter duration (0 = infinite) and click Start (F5)</li></ul>"
            "<p><b>6. Monitor & Analyze:</b></p>"
            "<ul><li>Watch Monitoring tab for real-time values</li>"
            "<li>Use Plotting tab to visualize results</li></ul>"
            "<p><b>7. Export Results:</b></p>"
            "<ul><li>Use File → Export or Ctrl+S to save data</li></ul>"
        )
        QMessageBox.information(self, "Quick Start Guide", guide_text)

    def _create_parameters_tab(self):
        """Create motor parameters configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Motor parameters group
        group = AccessibleGroupBox(
            "Motor Parameters",
            "Configure BLDC motor specifications. Default values from literature.",
        )
        group_layout = QVBoxLayout()

        self.param_voltage = LabeledSpinBox(
            "Nominal Voltage",
            min_val=12,
            max_val=300,
            initial=DEFAULT_MOTOR_PARAMS["nominal_voltage"],
            step=1,
            decimals=1,
            suffix=" V",
            description="Motor nominal DC voltage. Typical: 24-48V for hobby, 48-400V for industrial.",  # noqa: E501
        )
        group_layout.addWidget(self.param_voltage)

        self.param_resistance = LabeledSpinBox(
            "Phase Resistance",
            min_val=0.1,
            max_val=50,
            initial=DEFAULT_MOTOR_PARAMS["phase_resistance"],
            step=0.1,
            decimals=3,
            suffix=" Ω",
            description="Winding resistance per phase. Affects current and losses.",
        )
        group_layout.addWidget(self.param_resistance)

        self.param_inductance = LabeledSpinBox(
            "Phase Inductance",
            min_val=0.0001,
            max_val=0.1,
            initial=DEFAULT_MOTOR_PARAMS["phase_inductance"],
            step=0.0001,
            decimals=5,
            suffix=" H",
            description="Winding inductance per phase. Affects current rise time.",
        )
        group_layout.addWidget(self.param_inductance)

        # d/q inductances (Ld/Lq) for FOC users
        self.param_ld = LabeledSpinBox(
            "Ld (d-axis inductance)",
            min_val=0.00005,
            max_val=0.2,
            initial=DEFAULT_MOTOR_PARAMS.get("phase_inductance", 0.005),
            step=0.00001,
            decimals=6,
            suffix=" H",
            description="d-axis inductance. Use for FOC models (Ld).",
        )
        group_layout.addWidget(self.param_ld)

        self.param_lq = LabeledSpinBox(
            "Lq (q-axis inductance)",
            min_val=0.00005,
            max_val=0.2,
            initial=DEFAULT_MOTOR_PARAMS.get("phase_inductance", 0.005),
            step=0.00001,
            decimals=6,
            suffix=" H",
            description="q-axis inductance. Use for FOC models (Lq).",
        )
        group_layout.addWidget(self.param_lq)

        self.param_emf = LabeledSpinBox(
            "Back-EMF Constant",
            min_val=0.01,
            max_val=1.0,
            initial=DEFAULT_MOTOR_PARAMS["back_emf_constant"],
            step=0.01,
            decimals=4,
            suffix=" V·s/rad",
            description="Back-EMF voltage per angular velocity unit.",
        )
        group_layout.addWidget(self.param_emf)

        self.param_kt = LabeledSpinBox(
            "Torque Constant",
            min_val=0.01,
            max_val=1.0,
            initial=DEFAULT_MOTOR_PARAMS["torque_constant"],
            step=0.01,
            decimals=4,
            suffix=" N·m/A",
            description="Torque produced per unit current.",
        )
        group_layout.addWidget(self.param_kt)

        self.param_inertia = LabeledSpinBox(
            "Rotor Inertia",
            min_val=0.00001,
            max_val=0.01,
            initial=DEFAULT_MOTOR_PARAMS["rotor_inertia"],
            step=0.00001,
            decimals=6,
            suffix=" kg·m²",
            description="Moment of inertia. Affects acceleration response.",
        )
        group_layout.addWidget(self.param_inertia)

        self.param_friction = LabeledSpinBox(
            "Friction Coefficient",
            min_val=0,
            max_val=0.1,
            initial=DEFAULT_MOTOR_PARAMS["friction_coefficient"],
            step=0.001,
            decimals=5,
            suffix=" N·m·s/rad",
            description="Viscous friction damping.",
        )
        group_layout.addWidget(self.param_friction)

        self.param_poles = LabeledSpinBox(
            "Number of Poles",
            min_val=2,
            max_val=20,
            initial=DEFAULT_MOTOR_PARAMS["num_poles"],
            step=1,
            decimals=0,
            suffix="",
            description="Total magnetic poles in motor (must be even).",
        )
        group_layout.addWidget(self.param_poles)

        self.param_model_type = LabeledComboBox(
            "Motor Model",
            items=["scalar", "dq"],
            description="Select motor model type. 'scalar' for 3-phase model, 'dq' for d-q axis model.",  # noqa: E501
        )
        group_layout.addWidget(self.param_model_type)

        self.param_emf_shape = LabeledComboBox(
            "Back-EMF Shape",
            items=["trapezoidal", "sinusoidal"],
            description="Select back-EMF waveform shape. 'trapezoidal' for BLDC, 'sinusoidal' for PMSM.",  # noqa: E501
        )
        group_layout.addWidget(self.param_emf_shape)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()

        widget.setLayout(layout)
        return widget  # Motor & Drive tab picks this up

    def _create_load_tab(self):
        """Create load profile configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        group = AccessibleGroupBox("Load Profile", "Define mechanical load applied to motor shaft.")
        group_layout = QVBoxLayout()

        self.load_type = LabeledComboBox(
            "Load Type",
            items=["Constant", "Ramp", "Variable"],
            description="Select load profile type. Constant: fixed torque. "
            "Ramp: linear increase. Variable: custom profile.",
        )
        self.load_type.currentTextChanged.connect(self._on_load_type_changed)
        group_layout.addWidget(self.load_type)

        self.load_constant_torque = LabeledSpinBox(
            "Constant Load Torque",
            min_val=0,
            max_val=10,
            initial=_as_float(DEFAULT_LOAD_PROFILE["torque"]),
            step=0.1,
            decimals=2,
            suffix=" N·m",
            description="Constant load torque applied to motor shaft.",
        )
        group_layout.addWidget(self.load_constant_torque)

        self.load_initial_torque = LabeledSpinBox(
            "Initial Ramp Torque",
            min_val=0,
            max_val=10,
            initial=_as_float(DEFAULT_LOAD_PROFILE["initial_torque"]),
            step=0.1,
            decimals=2,
            suffix=" N·m",
            description="Starting load torque for ramp profile.",
        )
        group_layout.addWidget(self.load_initial_torque)

        self.load_final_torque = LabeledSpinBox(
            "Final Ramp Torque",
            min_val=0,
            max_val=10,
            initial=_as_float(DEFAULT_LOAD_PROFILE["final_torque"]),
            step=0.1,
            decimals=2,
            suffix=" N·m",
            description="Ending load torque for ramp profile.",
        )
        group_layout.addWidget(self.load_final_torque)

        self.load_ramp_duration = LabeledSpinBox(
            "Ramp Duration",
            min_val=0.1,
            max_val=10,
            initial=_as_float(DEFAULT_LOAD_PROFILE["ramp_duration"]),
            step=0.1,
            decimals=2,
            suffix=" s",
            description="Time to complete load ramp (affects ramp slope).",
        )
        group_layout.addWidget(self.load_ramp_duration)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()

        widget.setLayout(layout)
        return widget  # Motor & Drive tab picks this up
        # also create supply tab right after load
        # Supply tab is built separately by Motor & Drive composite tab

    def _create_supply_tab(self):
        """Create power supply configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        group = AccessibleGroupBox(
            "Supply Profile", "Define DC bus voltage profile feeding the inverter."
        )
        group_layout = QVBoxLayout()

        self.supply_type = LabeledComboBox(
            "Profile Type",
            items=["Constant", "Ramp"],
            description="Select supply voltage profile type. Variable/custom profiles not implemented yet.",  # noqa: E501
        )
        self.supply_type.currentTextChanged.connect(self._on_supply_type_changed)
        group_layout.addWidget(self.supply_type)

        self.supply_constant_voltage = LabeledSpinBox(
            "Constant Voltage",
            min_val=0,
            max_val=500,
            initial=_as_float(SIMULATION_PARAMS.get("dc_voltage", 48.0), 48.0),
            step=1,
            decimals=1,
            suffix=" V",
            description="Fixed DC bus voltage.",
        )
        group_layout.addWidget(self.supply_constant_voltage)

        self.supply_ramp_initial = LabeledSpinBox(
            "Initial Voltage",
            min_val=0,
            max_val=500,
            initial=_as_float(SIMULATION_PARAMS.get("dc_voltage", 48.0), 48.0),
            step=1,
            decimals=1,
            suffix=" V",
            description="Start voltage for ramp profile.",
        )
        group_layout.addWidget(self.supply_ramp_initial)

        self.supply_ramp_final = LabeledSpinBox(
            "Final Voltage",
            min_val=0,
            max_val=500,
            initial=_as_float(SIMULATION_PARAMS.get("dc_voltage", 48.0), 48.0),
            step=1,
            decimals=1,
            suffix=" V",
            description="End voltage for ramp profile.",
        )
        group_layout.addWidget(self.supply_ramp_final)

        self.supply_ramp_duration = LabeledSpinBox(
            "Ramp Duration",
            min_val=0.1,
            max_val=60,
            initial=1.0,
            step=0.1,
            decimals=2,
            suffix=" s",
            description="Time to complete voltage ramp.",
        )
        group_layout.addWidget(self.supply_ramp_duration)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget  # Motor & Drive tab picks this up

    def _on_supply_type_changed(self, text: str) -> None:
        """Show/hide appropriate supply parameters."""
        if text == "Constant":
            self.supply_constant_voltage.setVisible(True)
            self.supply_ramp_initial.setVisible(False)
            self.supply_ramp_final.setVisible(False)
            self.supply_ramp_duration.setVisible(False)
        else:
            self.supply_constant_voltage.setVisible(False)
            self.supply_ramp_initial.setVisible(True)
            self.supply_ramp_final.setVisible(True)
            self.supply_ramp_duration.setVisible(True)

    def _create_control_tab(self):
        """Create controller configuration tab with mode selection."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Simulation duration control
        duration_group = AccessibleGroupBox(
            "Simulation Duration", "Set how long the simulation runs (0 = infinite)"
        )
        duration_layout = QVBoxLayout()

        self.sim_duration = LabeledSpinBox(
            "Duration",
            min_val=0,
            max_val=300,
            initial=10.0,
            step=0.5,
            decimals=1,
            suffix=" s",
            description="Simulation runtime in seconds. Set to 0 for infinite/continuous simulation.",  # noqa: E501
        )
        duration_layout.addWidget(self.sim_duration)

        info_label = QLabel(
            "💡 Tip: Set to 0 seconds for infinite simulation (run until you press Stop)"
        )
        info_label.setWordWrap(True)
        duration_layout.addWidget(info_label)

        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)

        # allow choice between control algorithms
        self.ctrl_mode = LabeledComboBox(
            "Control Mode",
            items=["V/f", "FOC"],
            description="Select the control algorithm to use for simulation.",
        )
        self.ctrl_mode.currentTextChanged.connect(self._on_control_mode_changed)
        layout.addWidget(self.ctrl_mode)

        # V/f parameters group (shown when mode == "V/f")
        self.vf_group = AccessibleGroupBox(
            "V/f Speed Controller", "Configure Voltage-to-Frequency control algorithm."
        )
        self.vf_group_layout = QVBoxLayout()

        self.vf_v_nominal = LabeledSpinBox(
            "Nominal Voltage",
            min_val=1,
            max_val=100,
            initial=VF_CONTROLLER_PARAMS["v_nominal"],
            step=1,
            decimals=1,
            suffix=" V",
            description="Motor rated voltage (volts at nominal frequency).",
        )
        self.vf_group_layout.addWidget(self.vf_v_nominal)

        self.vf_f_nominal = LabeledSpinBox(
            "Nominal Frequency",
            min_val=1,
            max_val=500,
            initial=VF_CONTROLLER_PARAMS["f_nominal"],
            step=1,
            decimals=1,
            suffix=" Hz",
            description="Motor rated frequency (frequency at rated voltage).",
        )
        self.vf_group_layout.addWidget(self.vf_f_nominal)

        self.vf_startup_voltage = LabeledSpinBox(
            "Startup Voltage",
            min_val=0,
            max_val=20,
            initial=VF_CONTROLLER_PARAMS["v_startup"],
            step=0.1,
            decimals=2,
            suffix=" V",
            description="Initial voltage to overcome static friction. Typical: 0.5-2V.",
        )
        self.vf_group_layout.addWidget(self.vf_startup_voltage)

        self.vf_freq_slew = LabeledSpinBox(
            "Frequency Slew Rate",
            min_val=1,
            max_val=500,
            initial=VF_CONTROLLER_PARAMS["frequency_slew_rate"],
            step=1,
            decimals=1,
            suffix=" Hz/s",
            description="Maximum frequency change rate. Higher = faster acceleration.",
        )
        self.vf_group_layout.addWidget(self.vf_freq_slew)

        # Speed reference (will be shown in monitoring tab)
        self.vf_speed_ref = LabeledSpinBox(
            "Speed Reference",
            min_val=0,
            max_val=500,
            initial=50.0,
            step=1,
            decimals=1,
            suffix=" Hz",
            description="Desired motor operating frequency (speed command).",
        )
        self.vf_group_layout.addWidget(self.vf_speed_ref)

        self.vf_startup_sequence_mode = LabeledComboBox(
            "Startup Sequence",
            items=["Disabled", "Enabled"],
            description="Apply a standard V/f startup with rotor alignment, open-loop ramp, then steady V/f operation.",  # noqa: E501
        )
        self.vf_startup_sequence_mode.setCurrentText(
            "Enabled" if VF_CONTROLLER_PARAMS["startup_sequence_enabled"] else "Disabled"
        )
        self.vf_group_layout.addWidget(self.vf_startup_sequence_mode)

        self.vf_align_time = LabeledSpinBox(
            "Alignment Time",
            min_val=0.0,
            max_val=2.0,
            initial=VF_CONTROLLER_PARAMS["startup_align_duration_s"],
            step=0.005,
            decimals=3,
            suffix=" s",
            description="Fixed-angle pre-magnetization time before the V/f ramp starts.",
        )
        self.vf_group_layout.addWidget(self.vf_align_time)

        self.vf_align_voltage = LabeledSpinBox(
            "Alignment Voltage",
            min_val=0.0,
            max_val=20.0,
            initial=VF_CONTROLLER_PARAMS["startup_align_voltage_v"],
            step=0.1,
            decimals=2,
            suffix=" V",
            description="Voltage magnitude applied during rotor alignment.",
        )
        self.vf_group_layout.addWidget(self.vf_align_voltage)

        self.vf_align_angle = LabeledSpinBox(
            "Alignment Angle",
            min_val=0.0,
            max_val=360.0,
            initial=VF_CONTROLLER_PARAMS["startup_align_angle_deg"],
            step=1.0,
            decimals=1,
            suffix=" deg",
            description="Electrical angle held during the alignment phase.",
        )
        self.vf_group_layout.addWidget(self.vf_align_angle)

        self.vf_ramp_initial_frequency = LabeledSpinBox(
            "Ramp Initial Frequency",
            min_val=0.0,
            max_val=100.0,
            initial=VF_CONTROLLER_PARAMS["startup_ramp_initial_frequency_hz"],
            step=0.5,
            decimals=2,
            suffix=" Hz",
            description="Initial forced frequency used immediately after alignment.",
        )
        self.vf_group_layout.addWidget(self.vf_ramp_initial_frequency)

        self.vf_group.setLayout(self.vf_group_layout)
        layout.addWidget(self.vf_group)

        # FOC parameters group (hidden by default)
        self.foc_group = AccessibleGroupBox(
            "FOC Controller", "Configure Field-Oriented Control algorithm."
        )
        self.foc_group_layout = QVBoxLayout()

        self.foc_transform = LabeledComboBox(
            "Transform",
            items=["Clarke", "Concordia"],
            description="Choose coordinate transform for current measurement.",
        )
        self.foc_group_layout.addWidget(self.foc_transform)

        self.foc_output_mode = LabeledComboBox(
            "Output Mode",
            items=["Polar", "Cartesian"],
            description="Polar returns (mag,angle), Cartesian returns (v_alpha,v_beta).",
        )
        self.foc_group_layout.addWidget(self.foc_output_mode)

        self.foc_current_feedback_source = LabeledComboBox(
            "Current Feedback Source",
            items=["Motor True Currents", "Reconstructed (Shunt)"],
            description="Select whether FOC current loops use true model currents or reconstructed shunt measurements.",  # noqa: E501
        )
        self.foc_group_layout.addWidget(self.foc_current_feedback_source)

        self.foc_id_ref = LabeledSpinBox(
            "D-axis Current Ref",
            min_val=-10,
            max_val=10,
            initial=0.0,
            step=0.1,
            decimals=2,
            suffix=" A",
            description="Reference for d-axis current (field weakening).",
        )
        self.foc_group_layout.addWidget(self.foc_id_ref)

        self.foc_iq_ref = LabeledSpinBox(
            "Q-axis Current Ref",
            min_val=-10,
            max_val=10,
            initial=0.0,
            step=0.1,
            decimals=2,
            suffix=" A",
            description="Reference for q-axis current (torque-producing).",
        )
        self.foc_group_layout.addWidget(self.foc_iq_ref)

        self.foc_speed_ref = LabeledSpinBox(
            "Speed Ref (rpm)",
            min_val=0,
            max_val=10000,
            initial=0.0,
            step=10,
            decimals=0,
            suffix=" RPM",
            description="Speed reference for closed-loop control (mapped to iq).",
        )
        self.foc_group_layout.addWidget(self.foc_speed_ref)

        self.foc_field_weakening_mode = LabeledComboBox(
            "Field Weakening",
            items=["Disabled", "Enabled"],
            description="Independent field-weakening feature toggle. When enabled, additional negative d-axis current is injected above the configured speed threshold.",  # noqa: E501
        )
        self.foc_field_weakening_mode.setCurrentText(
            "Enabled" if FOC_FIELD_WEAKENING_PARAMS["enabled"] else "Disabled"
        )
        self.foc_group_layout.addWidget(self.foc_field_weakening_mode)

        self.foc_field_weakening_start_speed = LabeledSpinBox(
            "FW Start Speed",
            min_val=0.0,
            max_val=100000.0,
            initial=FOC_FIELD_WEAKENING_PARAMS["start_speed_rpm"],
            step=50.0,
            decimals=1,
            suffix=" RPM",
            description="Speed threshold above which field weakening begins injecting negative d-axis current.",  # noqa: E501
        )
        self.foc_group_layout.addWidget(self.foc_field_weakening_start_speed)

        self.foc_field_weakening_gain = LabeledSpinBox(
            "FW Gain",
            min_val=0.0,
            max_val=10.0,
            initial=FOC_FIELD_WEAKENING_PARAMS["gain"],
            step=0.05,
            decimals=3,
            suffix="",
            description="Scaling factor used by the field-weakening scheduler.",
        )
        self.foc_group_layout.addWidget(self.foc_field_weakening_gain)

        self.foc_field_weakening_max_id = LabeledSpinBox(
            "FW Max Negative Id",
            min_val=0.0,
            max_val=100.0,
            initial=FOC_FIELD_WEAKENING_PARAMS["max_negative_id_a"],
            step=0.1,
            decimals=2,
            suffix=" A",
            description="Maximum additional negative d-axis current magnitude injected by field weakening.",  # noqa: E501
        )
        self.foc_group_layout.addWidget(self.foc_field_weakening_max_id)

        self.foc_field_weakening_headroom_target = LabeledSpinBox(
            "FW Target Headroom",
            min_val=0.0,
            max_val=100.0,
            initial=FOC_FIELD_WEAKENING_PARAMS["headroom_target_v"],
            step=0.05,
            decimals=3,
            suffix=" V",
            description="Desired dq voltage reserve maintained by FW. Lower values maximize speed range; higher values preserve control margin.",  # noqa: E501
        )
        self.foc_group_layout.addWidget(self.foc_field_weakening_headroom_target)

        self.foc_speed_loop_mode = LabeledComboBox(
            "Speed Loop Mode",
            items=["Legacy iq mapping", "Cascaded PI"],
            description="Legacy keeps previous iq mapping behavior. Cascaded PI enables speed-loop to iq generation.",  # noqa: E501
        )
        self.foc_group_layout.addWidget(self.foc_speed_loop_mode)

        self.foc_iq_limit = LabeledSpinBox(
            "Iq Limit",
            min_val=0.1,
            max_val=100.0,
            initial=30.0,
            step=0.1,
            decimals=2,
            suffix=" A",
            description="Absolute q-axis current clamp used by cascaded speed loop.",
        )
        self.foc_group_layout.addWidget(self.foc_iq_limit)

        self.foc_speed_kp = LabeledSpinBox(
            "Speed PI Kp",
            min_val=0.0,
            max_val=10.0,
            initial=0.02,
            step=0.001,
            decimals=4,
            suffix="",
            description="Proportional gain for cascaded speed PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_speed_kp)

        self.foc_speed_ki = LabeledSpinBox(
            "Speed PI Ki",
            min_val=0.0,
            max_val=200.0,
            initial=1.0,
            step=0.1,
            decimals=3,
            suffix="",
            description="Integral gain for cascaded speed PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_speed_ki)

        self.foc_d_kp = LabeledSpinBox(
            "D-axis PI Kp",
            min_val=0.0,
            max_val=50.0,
            initial=1.0,
            step=0.01,
            decimals=3,
            suffix="",
            description="Proportional gain for d-axis current PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_d_kp)

        self.foc_d_ki = LabeledSpinBox(
            "D-axis PI Ki",
            min_val=0.0,
            max_val=500.0,
            initial=0.1,
            step=0.01,
            decimals=3,
            suffix="",
            description="Integral gain for d-axis current PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_d_ki)

        self.foc_q_kp = LabeledSpinBox(
            "Q-axis PI Kp",
            min_val=0.0,
            max_val=50.0,
            initial=1.0,
            step=0.01,
            decimals=3,
            suffix="",
            description="Proportional gain for q-axis current PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_q_kp)

        self.foc_q_ki = LabeledSpinBox(
            "Q-axis PI Ki",
            min_val=0.0,
            max_val=500.0,
            initial=0.1,
            step=0.01,
            decimals=3,
            suffix="",
            description="Integral gain for q-axis current PI loop.",
        )
        self.foc_group_layout.addWidget(self.foc_q_ki)

        self.foc_decouple_d_mode = LabeledComboBox(
            "D-axis Decoupling",
            items=["Disabled", "Enabled"],
            description="Enable d-axis feed-forward compensation term (-omega*Lq*iq).",
        )
        self.foc_group_layout.addWidget(self.foc_decouple_d_mode)

        self.foc_decouple_q_mode = LabeledComboBox(
            "Q-axis Decoupling",
            items=["Disabled", "Enabled"],
            description="Enable q-axis feed-forward compensation term (omega*Ld*id).",
        )
        self.foc_group_layout.addWidget(self.foc_decouple_q_mode)

        self.foc_group.setLayout(self.foc_group_layout)
        layout.addWidget(self.foc_group)

        # ── Observer group (moved to Observer & Startup tab) ─────────────────
        self.foc_observer_group = AccessibleGroupBox(
            "Angle Observer",
            "Select and configure the rotor angle observer. "
            "Parameters show/hide based on the selected observer type.",
        )
        self.foc_observer_group_layout = QVBoxLayout()

        self.foc_angle_observer_mode = LabeledComboBox(
            "Angle Observer",
            items=["Measured", "PLL", "SMO", "STSMO", "ActiveFlux",
                   "Auto (recommend from motor)"],
            description=(
                "Select rotor electrical angle source. "
                "Measured: direct sensor (always safe — default). "
                "PLL: phase-locked loop on back-EMF (best at high speed). "
                "SMO: first-order sliding-mode observer (good all-round sensorless). "
                "STSMO: Super-Twisting SMO with backward-Euler integration "
                "(best accuracy, unconditionally stable). "
                "ActiveFlux: Boldea active-flux integrator (robust in field-weakening, "
                "recommended for IPM salient motors). "
                "Auto (recommend from motor): Auto-Calibrate will analyse motor saliency "
                "and rated speed, "
                "then automatically select and configure the most appropriate observer. "
                "Default is Measured for safety; change to Auto before clicking Auto Calibrate "
                "to let the calibration pipeline choose the best observer "
                "for the loaded motor profile."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_angle_observer_mode)

        self.foc_solver_mode = LabeledComboBox(
            "Integration Solver",
            items=["Backward Euler (stable)", "Forward Euler"],
            description=(
                "Integration method for the STSMO current estimator. "
                "Backward Euler is unconditionally stable for any gain and is the default. "
                "Forward Euler requires k1 < L/(√2·dt) and can become unstable "
                "at high gains or low inductance."
            ),
        )
        self.foc_solver_mode.setCurrentText("Backward Euler (stable)")
        self.foc_observer_group_layout.addWidget(self.foc_solver_mode)

        self.foc_pll_kp = LabeledSpinBox(
            "PLL Kp",
            min_val=0.0,
            max_val=5000.0,
            initial=80.0,
            step=1.0,
            decimals=2,
            suffix="",
            description="Proportional gain for back-EMF PLL angle observer.",
        )
        self.foc_observer_group_layout.addWidget(self.foc_pll_kp)

        self.foc_pll_ki = LabeledSpinBox(
            "PLL Ki",
            min_val=0.0,
            max_val=50000.0,
            initial=2000.0,
            step=10.0,
            decimals=2,
            suffix="",
            description="Integral gain for back-EMF PLL angle observer.",
        )
        self.foc_observer_group_layout.addWidget(self.foc_pll_ki)

        # ── EEMF Model toggle (IPM saliency compensation) ─────────────────────
        self.foc_smo_eemf_model = LabeledComboBox(
            "EEMF Model (IPM)",
            items=["Disabled", "Enabled"],
            description=(
                "Enable Extended EMF (EEMF) model for IPM salient-pole motors "
                "(Chen 2003). Uses Lq instead of Ld in the di/dt term, removing "
                "the iq-dependent saliency bias (~13° at no-load for Lq/Ld=2). "
                "Auto-enabled by auto-calibration when Lq/Ld > 1.2."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_smo_eemf_model)

        # ── SOGI Filter toggle (zero-phase EMF bandpass) ──────────────────────
        self.foc_smo_sogi_filter = LabeledComboBox(
            "SOGI Filter",
            items=["Disabled", "Enabled"],
            description=(
                "Enable Second-Order Generalized Integrator (SOGI) bandpass filter "
                "for back-EMF reconstruction. Replaces the fixed-frequency LPF with "
                "a resonant filter that tracks ωe: zero phase lag at the electrical "
                "frequency vs ~14° lag from the standard LPF at 3000 RPM. "
                "Recommended for IPM motors and high-speed sensorless operation. "
                "Auto-enabled by auto-calibration when Lq/Ld > 1.2."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_smo_sogi_filter)

        self.foc_smo_k_slide = LabeledSpinBox(
            "SMO Kslide",
            min_val=0.0,
            max_val=10000.0,
            initial=600.0,
            step=5.0,
            decimals=2,
            suffix="",
            description="Sliding gain for SMO-inspired angle observer correction.",
        )
        self.foc_observer_group_layout.addWidget(self.foc_smo_k_slide)

        self.foc_smo_lpf_alpha = LabeledSpinBox(
            "SMO LPF Alpha",
            min_val=0.001,
            max_val=1.0,
            initial=0.08,
            step=0.001,
            decimals=3,
            suffix="",
            description="Low-pass blending factor for SMO estimated electrical speed.",
        )
        self.foc_observer_group_layout.addWidget(self.foc_smo_lpf_alpha)

        self.foc_smo_boundary = LabeledSpinBox(
            "SMO Boundary",
            min_val=0.001,
            max_val=1.0,
            initial=0.06,
            step=0.001,
            decimals=3,
            suffix=" rad",
            description="Boundary layer width used in SMO switching nonlinearity.",
        )
        self.foc_observer_group_layout.addWidget(self.foc_smo_boundary)

        # ── STSMO parameters (shown only when STSMO observer is selected) ──────
        self.foc_stsmo_k1 = LabeledSpinBox(
            "STSMO k1 (injection)",
            min_val=0.1,
            max_val=500.0,
            initial=18.0,
            step=0.5,
            decimals=2,
            suffix="",
            description=(
                "Super-Twisting injection gain k1. Controls how aggressively the "
                "observer reacts to current error. Higher values → faster convergence "
                "but more chattering. Use 'Auto-calibrate STSMO' to set analytically."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_stsmo_k1)

        self.foc_stsmo_k2_min = LabeledSpinBox(
            "STSMO k2 floor",
            min_val=1.0,
            max_val=50000.0,
            initial=500.0,
            step=10.0,
            decimals=1,
            suffix=" V/s",
            description=(
                "Minimum integral gain k2 [V/s] for the STSMO. "
                "The effective k2 is speed-adaptive (k2 ≥ ke·ωm·ωe) but never "
                "falls below this floor, preventing loss of tracking at standstill."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_stsmo_k2_min)

        self.foc_stsmo_k2_factor = LabeledSpinBox(
            "STSMO k2 factor",
            min_val=0.1,
            max_val=10.0,
            initial=1.0,
            step=0.1,
            decimals=2,
            suffix="×",
            description=(
                "Speed-adaptive k2 scaling factor (Levant condition multiplier). "
                "Effective k2 = max(k2_floor, factor × ke × ωm × ωe). "
                "Values > 1 increase robustness at the cost of higher chattering. "
                "SOGI post-filter suppresses chattering, so 1.0 is optimal."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_stsmo_k2_factor)

        self.foc_stsmo_rated_rpm = LabeledSpinBox(
            "STSMO Rated Speed",
            min_val=100.0,
            max_val=100000.0,
            initial=3500.0,
            step=100.0,
            decimals=0,
            suffix=" RPM",
            description=(
                "Rated motor speed used for analytical gain calibration. "
                "Pressing 'Auto-calibrate STSMO' computes k1 and k2 from this speed "
                "and the motor back-EMF constant."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_stsmo_rated_rpm)

        self.foc_stsmo_autocalib_btn = AccessibleButton(
            "Auto-calibrate STSMO",
            tooltip=(
                "Compute ST-SMO gains analytically from motor parameters and rated speed. "
                "Sets k1 = λ·√(ke·ωm_max), k2_factor = 1.0 (Levant condition). "
                "Run this once after setting motor parameters before starting the simulation."
            ),
        )
        self.foc_stsmo_autocalib_btn.clicked.connect(self._on_stsmo_autocalib)
        self.foc_observer_group_layout.addWidget(self.foc_stsmo_autocalib_btn)

        # ── ActiveFlux parameter ──────────────────────────────────────────────
        self.foc_af_dc_cutoff = LabeledSpinBox(
            "ActiveFlux DC cutoff",
            min_val=0.01,
            max_val=10.0,
            initial=0.5,
            step=0.05,
            decimals=2,
            suffix=" Hz",
            description=(
                "Drift-correction pole frequency for the Active Flux integrator [Hz]. "
                "Must be much lower than the minimum electrical frequency (typically < 1 Hz). "
                "Lower values → less drift suppression; "
                "higher values → phase distortion at low speed."
            ),
        )
        self.foc_observer_group_layout.addWidget(self.foc_af_dc_cutoff)
        self.foc_observer_group.setLayout(self.foc_observer_group_layout)
        # Not added to layout — picked up by Observer & Startup tab

        # ── Startup Sequence group ────────────────────────────────────────────
        self.foc_startup_group = AccessibleGroupBox(
            "Startup Sequence & Transition",
            "Configure rotor alignment, open-loop ramp, and observer handoff criteria.",
        )
        self.foc_startup_group_layout = QVBoxLayout()

        self.foc_startup_sequence_mode = LabeledComboBox(
            "Startup Sequence",
            items=["Disabled", "Enabled"],
            description="Apply a standard FOC startup: align rotor, use open-loop ramp if sensorless, then hand off to closed-loop observer control.",  # noqa: E501
        )
        self.foc_startup_sequence_mode.setCurrentText(
            "Enabled" if FOC_STARTUP_PARAMS["startup_sequence_enabled"] else "Disabled"
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_sequence_mode)

        self.foc_align_time = LabeledSpinBox(
            "Alignment Time",
            min_val=0.0,
            max_val=2.0,
            initial=FOC_STARTUP_PARAMS["startup_align_duration_s"],
            step=0.005,
            decimals=3,
            suffix=" s",
            description="Duration of the initial rotor alignment phase.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_align_time)

        self.foc_align_current = LabeledSpinBox(
            "Alignment Current",
            min_val=0.0,
            max_val=20.0,
            initial=FOC_STARTUP_PARAMS["startup_align_current_a"],
            step=0.1,
            decimals=2,
            suffix=" A",
            description="d-axis current used to lock the rotor before acceleration.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_align_current)

        self.foc_align_angle = LabeledSpinBox(
            "Alignment Angle",
            min_val=0.0,
            max_val=360.0,
            initial=FOC_STARTUP_PARAMS["startup_align_angle_deg"],
            step=1.0,
            decimals=1,
            suffix=" deg",
            description="Electrical angle held during rotor alignment.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_align_angle)

        self.foc_open_loop_initial_speed = LabeledSpinBox(
            "Open-Loop Initial Speed",
            min_val=0.0,
            max_val=5000.0,
            initial=FOC_STARTUP_PARAMS["startup_open_loop_initial_speed_rpm"],
            step=5.0,
            decimals=1,
            suffix=" RPM",
            description="Forced mechanical speed used at the beginning of the sensorless ramp.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_open_loop_initial_speed)

        self.foc_open_loop_target_speed = LabeledSpinBox(
            "Open-Loop Target Speed",
            min_val=0.0,
            max_val=10000.0,
            initial=FOC_STARTUP_PARAMS["startup_open_loop_target_speed_rpm"],
            step=10.0,
            decimals=1,
            suffix=" RPM",
            description="Forced mechanical speed reached before closed-loop observer takeover.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_open_loop_target_speed)

        self.foc_open_loop_ramp_time = LabeledSpinBox(
            "Open-Loop Ramp Time",
            min_val=0.0,
            max_val=5.0,
            initial=FOC_STARTUP_PARAMS["startup_open_loop_ramp_time_s"],
            step=0.01,
            decimals=3,
            suffix=" s",
            description="Duration of the forced-angle acceleration ramp.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_open_loop_ramp_time)

        self.foc_open_loop_id_ref = LabeledSpinBox(
            "Open-Loop d-axis Ref",
            min_val=-20.0,
            max_val=20.0,
            initial=FOC_STARTUP_PARAMS["startup_open_loop_id_ref_a"],
            step=0.1,
            decimals=2,
            suffix=" A",
            description="d-axis current reference used during forced open-loop acceleration.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_open_loop_id_ref)

        self.foc_open_loop_iq_ref = LabeledSpinBox(
            "Open-Loop q-axis Ref",
            min_val=-20.0,
            max_val=20.0,
            initial=FOC_STARTUP_PARAMS["startup_open_loop_iq_ref_a"],
            step=0.1,
            decimals=2,
            suffix=" A",
            description="q-axis current reference used to generate torque during the forced ramp.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_open_loop_iq_ref)

        self.foc_startup_transition_mode = LabeledComboBox(
            "Observer Startup Transition",
            items=["Disabled", "Enabled"],
            description="Automatic observer takeover criteria used after the forced open-loop ramp or for legacy observer-only startup mode.",  # noqa: E501
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_transition_mode)

        self.foc_startup_initial_observer = LabeledComboBox(
            "Startup Initial Observer",
            items=["Measured", "PLL", "SMO", "STSMO", "ActiveFlux",
                   "Auto (recommend from motor)"],
            description=(
                "Observer mode used during the startup phase before handoff to the main observer. "
                "Measured: safe choice when a position sensor is available (default). "
                "PLL or SMO: suitable when back-EMF is already above the noise floor. "
                "STSMO: recommended for sensorless startups — backward-Euler guarantees stability. "
                "ActiveFlux: use when field-weakening starts immediately at low speed. "
                "Auto (recommend from motor): mirrors the main observer selection "
                "made by Auto Calibrate — "
                "for sensorless startup the calibration pipeline will set STSMO "
                "(unconditionally stable), "
                "or Measured when a sensor is detected."
            ),
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_initial_observer)

        self.foc_startup_min_speed = LabeledSpinBox(
            "Startup Min Speed",
            min_val=0.0,
            max_val=20000.0,
            initial=300.0,
            step=10.0,
            decimals=1,
            suffix=" RPM",
            description="Minimum speed required before observer handoff.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_min_speed)

        self.foc_startup_min_time = LabeledSpinBox(
            "Startup Min Time",
            min_val=0.0,
            max_val=5.0,
            initial=0.05,
            step=0.005,
            decimals=3,
            suffix=" s",
            description="Minimum startup dwell time before observer handoff.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_min_time)

        self.foc_startup_min_emf = LabeledSpinBox(
            "Startup Min Back-EMF",
            min_val=0.0,
            max_val=50.0,
            initial=0.5,
            step=0.05,
            decimals=3,
            suffix=" V",
            description="Minimum back-EMF magnitude required before observer handoff.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_min_emf)

        self.foc_startup_min_confidence = LabeledSpinBox(
            "Startup Min Confidence",
            min_val=0.0,
            max_val=1.0,
            initial=0.6,
            step=0.01,
            decimals=2,
            suffix="",
            description="Minimum observer confidence required before handoff.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_min_confidence)

        self.foc_startup_confidence_hold = LabeledSpinBox(
            "Confidence Hold Time",
            min_val=0.0,
            max_val=1.0,
            initial=0.02,
            step=0.005,
            decimals=3,
            suffix=" s",
            description="Time confidence must stay above threshold before handoff.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_confidence_hold)

        self.foc_startup_confidence_hysteresis = LabeledSpinBox(
            "Confidence Hysteresis",
            min_val=0.0,
            max_val=1.0,
            initial=0.1,
            step=0.01,
            decimals=2,
            suffix="",
            description="Lower-confidence margin before fallback to startup observer.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_confidence_hysteresis)

        self.foc_startup_fallback_mode = LabeledComboBox(
            "Observer Fallback",
            items=["Enabled", "Disabled"],
            description="Allow reverting to startup observer when confidence degrades.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_fallback_mode)

        self.foc_startup_fallback_hold = LabeledSpinBox(
            "Fallback Hold Time",
            min_val=0.0,
            max_val=1.0,
            initial=0.03,
            step=0.005,
            decimals=3,
            suffix=" s",
            description="Time degraded confidence must persist before fallback.",
        )
        self.foc_startup_group_layout.addWidget(self.foc_startup_fallback_hold)

        btn_auto_d = AccessibleButton("Auto-tune d-axis", "Auto-tune d-axis PI controller")
        btn_auto_d.clicked.connect(lambda: self._auto_tune_axis("d"))
        self.foc_startup_group_layout.addWidget(btn_auto_d)

        btn_auto_q = AccessibleButton("Auto-tune q-axis", "Auto-tune q-axis PI controller")
        btn_auto_q.clicked.connect(lambda: self._auto_tune_axis("q"))
        self.foc_startup_group_layout.addWidget(btn_auto_q)

        self.foc_startup_group.setLayout(self.foc_startup_group_layout)
        # Not added to layout — picked up by Observer & Startup tab

        # ── Advanced groups (moved to Advanced tab) ───────────────────────────
        _adv_widget = QWidget()
        _adv_layout = QVBoxLayout()

        self.inverter_group = AccessibleGroupBox(
            "Inverter Non-Idealities",
            "Optional inverter realism settings. Set to zero to keep ideal inverter behavior.",
        )
        inverter_layout = QVBoxLayout()

        # Feature toggles let the user dial the realism level up or down without
        # losing any tuned numerical parameters.
        self.inverter_enable_device_drop = QCheckBox("Enable Device Drop")
        self.inverter_enable_device_drop.setChecked(False)
        self.inverter_enable_device_drop.setAccessibleName("Enable device voltage drop")
        self.inverter_enable_device_drop.setAccessibleDescription(
            "When checked, applies a per-phase voltage drop from switching devices (IGBT, MOSFET)."
        )
        inverter_layout.addWidget(self.inverter_enable_device_drop)

        self.inverter_enable_dead_time = QCheckBox("Enable Dead-Time Distortion")
        self.inverter_enable_dead_time.setChecked(False)
        self.inverter_enable_dead_time.setAccessibleName("Enable dead-time distortion")
        self.inverter_enable_dead_time.setAccessibleDescription(
            "When checked, simulates gate dead-time as a voltage distortion "
            "proportional to current sign."
        )
        inverter_layout.addWidget(self.inverter_enable_dead_time)

        self.inverter_enable_conduction = QCheckBox("Enable Conduction Loss")
        self.inverter_enable_conduction.setChecked(False)
        self.inverter_enable_conduction.setAccessibleName("Enable conduction loss")
        self.inverter_enable_conduction.setAccessibleDescription(
            "When checked, models I²R conduction loss through inverter on-resistance."
        )
        inverter_layout.addWidget(self.inverter_enable_conduction)

        self.inverter_enable_switching = QCheckBox("Enable Switching Loss")
        self.inverter_enable_switching.setChecked(False)
        self.inverter_enable_switching.setAccessibleName("Enable switching loss")
        self.inverter_enable_switching.setAccessibleDescription(
            "When checked, models switching loss as a frequency-proportional voltage reduction."
        )
        inverter_layout.addWidget(self.inverter_enable_switching)

        self.inverter_enable_diode = QCheckBox("Enable Freewheel Diode Path")
        self.inverter_enable_diode.setChecked(False)
        self.inverter_enable_diode.setAccessibleName("Enable freewheel diode path")
        self.inverter_enable_diode.setAccessibleDescription(
            "When checked, models body-diode forward drop on phase legs carrying opposing current."
        )
        inverter_layout.addWidget(self.inverter_enable_diode)

        self.inverter_enable_min_pulse = QCheckBox("Enable Minimum Pulse Suppression")
        self.inverter_enable_min_pulse.setChecked(False)
        self.inverter_enable_min_pulse.setAccessibleName("Enable minimum pulse suppression")
        self.inverter_enable_min_pulse.setAccessibleDescription(
            "When checked, voltage commands smaller than the minimum pulse fraction are zeroed."
        )
        inverter_layout.addWidget(self.inverter_enable_min_pulse)

        self.inverter_enable_bus_ripple = QCheckBox("Enable DC-Link Ripple")
        self.inverter_enable_bus_ripple.setChecked(False)
        self.inverter_enable_bus_ripple.setAccessibleName("Enable DC-link bus ripple")
        self.inverter_enable_bus_ripple.setAccessibleDescription(
            "When checked, models capacitor ripple and source impedance on the DC bus voltage."
        )
        inverter_layout.addWidget(self.inverter_enable_bus_ripple)

        self.inverter_enable_thermal = QCheckBox("Enable Thermal Coupling")
        self.inverter_enable_thermal.setChecked(False)
        self.inverter_enable_thermal.setAccessibleName("Enable thermal coupling")
        self.inverter_enable_thermal.setAccessibleDescription(
            "When checked, tracks junction temperature and scales device resistance "
            "with temperature."
        )
        inverter_layout.addWidget(self.inverter_enable_thermal)

        self.inverter_enable_phase_asymmetry = QCheckBox("Enable Phase Asymmetry")
        self.inverter_enable_phase_asymmetry.setChecked(False)
        self.inverter_enable_phase_asymmetry.setAccessibleName("Enable phase asymmetry")
        self.inverter_enable_phase_asymmetry.setAccessibleDescription(
            "When checked, allows independent voltage scale and drop offsets per phase."
        )
        inverter_layout.addWidget(self.inverter_enable_phase_asymmetry)

        self.inverter_device_drop = LabeledSpinBox(
            "Device Drop",
            min_val=0.0,
            max_val=10.0,
            initial=0.0,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Approximate per-phase effective voltage drop from switching devices.",
        )
        inverter_layout.addWidget(self.inverter_device_drop)

        self.inverter_dead_time_fraction = LabeledSpinBox(
            "Dead-Time Loss",
            min_val=0.0,
            max_val=0.2,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" pu",
            description="Duty loss fraction from dead-time effects. 0 keeps ideal modulation.",
        )
        inverter_layout.addWidget(self.inverter_dead_time_fraction)

        self.inverter_conduction_resistance = LabeledSpinBox(
            "Conduction Resistance",
            min_val=0.0,
            max_val=5.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" Ohm",
            description="Effective inverter conduction path resistance used for current-dependent voltage drop.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_conduction_resistance)

        self.inverter_switching_frequency = LabeledSpinBox(
            "Switching Frequency",
            min_val=0.0,
            max_val=200000.0,
            initial=_as_float(SIMULATION_PARAMS.get("pwm_frequency_hz", 20000.0), 20000.0),
            step=500.0,
            decimals=1,
            suffix=" Hz",
            description="PWM switching frequency; also sets control update period for FOC and V/f loops.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_switching_frequency)

        self.inverter_switching_loss_coeff = LabeledSpinBox(
            "Switching Loss Coeff",
            min_val=0.0,
            max_val=1.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" V/A/kHz",
            description="Voltage-loss coefficient per ampere and kHz for switching-dependent drop modeling.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_switching_loss_coeff)

        self.inverter_diode_drop = LabeledSpinBox(
            "Diode Drop",
            min_val=0.0,
            max_val=10.0,
            initial=0.0,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Additional freewheel diode drop when current polarity opposes commanded phase voltage.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_diode_drop)

        self.inverter_diode_resistance = LabeledSpinBox(
            "Diode Resistance",
            min_val=0.0,
            max_val=5.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" Ohm",
            description="Series resistance used in the freewheel diode path model.",
        )
        inverter_layout.addWidget(self.inverter_diode_resistance)

        self.inverter_min_pulse_fraction = LabeledSpinBox(
            "Min Pulse Fraction",
            min_val=0.0,
            max_val=0.5,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" pu",
            description="Commands smaller than this fraction of half-bus voltage are suppressed to emulate minimum PWM on-time.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_min_pulse_fraction)

        self.inverter_dc_link_capacitance = LabeledSpinBox(
            "DC-Link Capacitance",
            min_val=0.0,
            max_val=1.0,
            initial=0.0,
            step=0.0001,
            decimals=5,
            suffix=" F",
            description="Capacitance used by the reduced-order DC-link ripple model.",
        )
        inverter_layout.addWidget(self.inverter_dc_link_capacitance)

        self.inverter_dc_link_source_resistance = LabeledSpinBox(
            "DC-Link Source Resistance",
            min_val=0.0,
            max_val=5.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" Ohm",
            description="Source resistance that recharges the DC-link capacitor in the ripple model.",  # noqa: E501
        )
        inverter_layout.addWidget(self.inverter_dc_link_source_resistance)

        self.inverter_dc_link_esr = LabeledSpinBox(
            "DC-Link ESR",
            min_val=0.0,
            max_val=5.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" Ohm",
            description="Equivalent series resistance used to compute instantaneous DC-link sag.",
        )
        inverter_layout.addWidget(self.inverter_dc_link_esr)

        self.inverter_thermal_resistance = LabeledSpinBox(
            "Thermal Resistance",
            min_val=0.0,
            max_val=20.0,
            initial=0.0,
            step=0.01,
            decimals=3,
            suffix=" K/W",
            description="Equivalent junction-to-ambient thermal resistance for inverter devices.",
        )
        inverter_layout.addWidget(self.inverter_thermal_resistance)

        self.inverter_thermal_capacitance = LabeledSpinBox(
            "Thermal Capacitance",
            min_val=0.001,
            max_val=10000.0,
            initial=1.0,
            step=0.1,
            decimals=3,
            suffix=" J/K",
            description="Thermal capacitance used by the reduced-order junction temperature model.",
        )
        inverter_layout.addWidget(self.inverter_thermal_capacitance)

        self.inverter_ambient_temp = LabeledSpinBox(
            "Ambient Temperature",
            min_val=-40.0,
            max_val=200.0,
            initial=25.0,
            step=1.0,
            decimals=1,
            suffix=" C",
            description="Ambient temperature used by the inverter thermal model.",
        )
        inverter_layout.addWidget(self.inverter_ambient_temp)

        self.inverter_temp_coeff_resistance = LabeledSpinBox(
            "Resistance Temp Coeff",
            min_val=0.0,
            max_val=0.05,
            initial=0.0,
            step=0.0005,
            decimals=5,
            suffix=" 1/C",
            description="Relative resistance increase per degree C in the thermal coupling model.",
        )
        inverter_layout.addWidget(self.inverter_temp_coeff_resistance)

        self.inverter_temp_coeff_drop = LabeledSpinBox(
            "Drop Temp Coeff",
            min_val=0.0,
            max_val=0.05,
            initial=0.0,
            step=0.0005,
            decimals=5,
            suffix=" 1/C",
            description="Relative device-drop increase per degree C in the thermal coupling model.",
        )
        inverter_layout.addWidget(self.inverter_temp_coeff_drop)

        self.inverter_phase_voltage_scale_a = LabeledSpinBox(
            "Phase A Voltage Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase A mismatch multiplier applied to commanded phase voltage.",
        )
        inverter_layout.addWidget(self.inverter_phase_voltage_scale_a)

        self.inverter_phase_voltage_scale_b = LabeledSpinBox(
            "Phase B Voltage Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase B mismatch multiplier applied to commanded phase voltage.",
        )
        inverter_layout.addWidget(self.inverter_phase_voltage_scale_b)

        self.inverter_phase_voltage_scale_c = LabeledSpinBox(
            "Phase C Voltage Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase C mismatch multiplier applied to commanded phase voltage.",
        )
        inverter_layout.addWidget(self.inverter_phase_voltage_scale_c)

        self.inverter_phase_drop_scale_a = LabeledSpinBox(
            "Phase A Drop Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase A mismatch multiplier applied to inverter loss/drop terms.",
        )
        inverter_layout.addWidget(self.inverter_phase_drop_scale_a)

        self.inverter_phase_drop_scale_b = LabeledSpinBox(
            "Phase B Drop Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase B mismatch multiplier applied to inverter loss/drop terms.",
        )
        inverter_layout.addWidget(self.inverter_phase_drop_scale_b)

        self.inverter_phase_drop_scale_c = LabeledSpinBox(
            "Phase C Drop Scale",
            min_val=0.5,
            max_val=1.5,
            initial=1.0,
            step=0.001,
            decimals=4,
            suffix="",
            description="Phase C mismatch multiplier applied to inverter loss/drop terms.",
        )
        inverter_layout.addWidget(self.inverter_phase_drop_scale_c)

        self.inverter_group.setLayout(inverter_layout)
        _adv_layout.addWidget(self.inverter_group)

        self.current_sense_group = AccessibleGroupBox(
            "Current Measurement and FFT",
            "Topology-aware shunt sensing controls and separate current harmonic FFT window.",
        )
        current_sense_layout = QVBoxLayout()

        self.current_sense_enable = QCheckBox("Enable Current Sensing Model")
        self.current_sense_enable.setChecked(False)
        self.current_sense_enable.setAccessibleName("Enable current sensing model")
        self.current_sense_enable.setAccessibleDescription(
            "When checked, applies shunt amplifier gain, offset, and anti-alias filter "
            "to the measured phase currents before they reach the controller."
        )
        current_sense_layout.addWidget(self.current_sense_enable)

        self.current_sense_topology = LabeledComboBox(
            "Sensing Topology",
            items=["single", "double", "triple"],
            description="Single, double, or triple-shunt current measurement topology.",
        )
        self.current_sense_topology.currentTextChanged.connect(
            lambda _text: self._update_bridge_visualization({})
        )
        current_sense_layout.addWidget(self.current_sense_topology)

        self.current_sense_r_shunt = LabeledSpinBox(
            "Shunt Resistance",
            min_val=0.00005,
            max_val=0.05,
            initial=0.001,
            step=0.0001,
            decimals=5,
            suffix=" Ohm",
            description="Per-shunt resistance used in current sensing.",
        )
        current_sense_layout.addWidget(self.current_sense_r_shunt)

        self.current_sense_nominal_gain = LabeledSpinBox(
            "Nominal Gain",
            min_val=1.0,
            max_val=500.0,
            initial=20.0,
            step=0.5,
            decimals=2,
            suffix=" V/V",
            description="Gain used by controller reconstruction path.",
        )
        current_sense_layout.addWidget(self.current_sense_nominal_gain)

        self.current_sense_nominal_offset = LabeledSpinBox(
            "Nominal Offset",
            min_val=0.0,
            max_val=5.0,
            initial=1.65,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Nominal offset used by controller reconstruction path.",
        )
        current_sense_layout.addWidget(self.current_sense_nominal_offset)

        self.current_sense_cutoff_hz = LabeledSpinBox(
            "Sensing Cutoff",
            min_val=100.0,
            max_val=200000.0,
            initial=20000.0,
            step=100.0,
            decimals=1,
            suffix=" Hz",
            description="First-order analog anti-aliasing cutoff frequency.",
        )
        current_sense_layout.addWidget(self.current_sense_cutoff_hz)

        self.current_sense_vcc = LabeledSpinBox(
            "ADC Vcc",
            min_val=1.0,
            max_val=5.0,
            initial=3.3,
            step=0.01,
            decimals=2,
            suffix=" V",
            description="ADC full-scale clamp voltage for sensing channels.",
        )
        current_sense_layout.addWidget(self.current_sense_vcc)

        self.current_sense_actual_gain_a = LabeledSpinBox(
            "Actual Gain A",
            min_val=1.0,
            max_val=500.0,
            initial=20.0,
            step=0.5,
            decimals=2,
            suffix=" V/V",
            description="Runtime actual gain for shunt channel A.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_gain_a)

        self.current_sense_actual_gain_b = LabeledSpinBox(
            "Actual Gain B",
            min_val=1.0,
            max_val=500.0,
            initial=20.0,
            step=0.5,
            decimals=2,
            suffix=" V/V",
            description="Runtime actual gain for shunt channel B.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_gain_b)

        self.current_sense_actual_gain_c = LabeledSpinBox(
            "Actual Gain C",
            min_val=1.0,
            max_val=500.0,
            initial=20.0,
            step=0.5,
            decimals=2,
            suffix=" V/V",
            description="Runtime actual gain for shunt channel C.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_gain_c)

        self.current_sense_actual_offset_a = LabeledSpinBox(
            "Actual Offset A",
            min_val=0.0,
            max_val=5.0,
            initial=1.65,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Runtime actual offset for shunt channel A.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_offset_a)

        self.current_sense_actual_offset_b = LabeledSpinBox(
            "Actual Offset B",
            min_val=0.0,
            max_val=5.0,
            initial=1.65,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Runtime actual offset for shunt channel B.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_offset_b)

        self.current_sense_actual_offset_c = LabeledSpinBox(
            "Actual Offset C",
            min_val=0.0,
            max_val=5.0,
            initial=1.65,
            step=0.01,
            decimals=3,
            suffix=" V",
            description="Runtime actual offset for shunt channel C.",
        )
        current_sense_layout.addWidget(self.current_sense_actual_offset_c)

        self.current_sense_fft_window_samples = LabeledSpinBox(
            "FFT Window Samples",
            min_val=64,
            max_val=16384,
            initial=512,
            step=64,
            decimals=0,
            suffix=" samples",
            description="Sample window length used by the asynchronous FFT window.",
        )
        current_sense_layout.addWidget(self.current_sense_fft_window_samples)

        self.current_sense_fft_show_grid = QCheckBox("Show FFT Grids")
        self.current_sense_fft_show_grid.setChecked(True)
        self.current_sense_fft_show_grid.setAccessibleName("Show FFT grids")
        self.current_sense_fft_show_grid.setAccessibleDescription(
            "When checked, overlays a frequency grid on the FFT magnitude and phase plots."
        )
        current_sense_layout.addWidget(self.current_sense_fft_show_grid)

        fft_scale_row_1 = QHBoxLayout()
        fft_scale_row_1.addWidget(QLabel("Magnitude X Scale"))
        self.current_sense_fft_mag_x_scale = QComboBox()
        self.current_sense_fft_mag_x_scale.addItems(["linear", "log"])
        fft_scale_row_1.addWidget(self.current_sense_fft_mag_x_scale)
        fft_scale_row_1.addWidget(QLabel("Magnitude Y Scale"))
        self.current_sense_fft_mag_y_scale = QComboBox()
        self.current_sense_fft_mag_y_scale.addItems(["linear", "log"])
        fft_scale_row_1.addWidget(self.current_sense_fft_mag_y_scale)
        current_sense_layout.addLayout(fft_scale_row_1)

        fft_scale_row_2 = QHBoxLayout()
        fft_scale_row_2.addWidget(QLabel("Phase X Scale"))
        self.current_sense_fft_phase_x_scale = QComboBox()
        self.current_sense_fft_phase_x_scale.addItems(["linear", "log"])
        fft_scale_row_2.addWidget(self.current_sense_fft_phase_x_scale)
        fft_scale_row_2.addWidget(QLabel("Phase Y Scale"))
        self.current_sense_fft_phase_y_scale = QComboBox()
        self.current_sense_fft_phase_y_scale.addItems(["linear", "log"])
        fft_scale_row_2.addWidget(self.current_sense_fft_phase_y_scale)
        current_sense_layout.addLayout(fft_scale_row_2)

        fft_units_row = QHBoxLayout()
        fft_units_row.addWidget(QLabel("Amplitude Unit"))
        self.current_sense_fft_amplitude_mode = QComboBox()
        self.current_sense_fft_amplitude_mode.addItems(["linear", "dB"])
        fft_units_row.addWidget(self.current_sense_fft_amplitude_mode)
        fft_units_row.addWidget(QLabel("Phase Unit"))
        self.current_sense_fft_phase_unit = QComboBox()
        self.current_sense_fft_phase_unit.addItems(["deg", "rad"])
        fft_units_row.addWidget(self.current_sense_fft_phase_unit)
        current_sense_layout.addLayout(fft_units_row)

        for combo in (
            self.current_sense_fft_mag_x_scale,
            self.current_sense_fft_mag_y_scale,
            self.current_sense_fft_phase_x_scale,
            self.current_sense_fft_phase_y_scale,
            self.current_sense_fft_amplitude_mode,
            self.current_sense_fft_phase_unit,
        ):
            combo.currentTextChanged.connect(self._apply_fft_display_settings)
        self.current_sense_fft_show_grid.stateChanged.connect(self._apply_fft_display_settings)

        bridge_title = QLabel("Live Inverter Bridge / Shunt Topology")
        current_sense_layout.addWidget(bridge_title)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self.bridge_figure = Figure(figsize=(6.4, 2.8), dpi=90)
        self.bridge_canvas = FigureCanvas(self.bridge_figure)
        self.bridge_ax = self.bridge_figure.add_subplot(111)
        self.bridge_ax.set_axis_off()
        _bridge_container = QWidget()
        _bridge_vbox = QVBoxLayout(_bridge_container)
        _bridge_vbox.setContentsMargins(0, 0, 0, 0)
        _bridge_vbox.addWidget(self.bridge_canvas)
        current_sense_layout.addWidget(_bridge_container)

        self.current_sense_status_label = QLabel(
            "Current sensing disabled. Enable to expose measured-vs-true current telemetry."
        )
        self.current_sense_status_label.setWordWrap(True)
        current_sense_layout.addWidget(self.current_sense_status_label)

        btn_fft_window = AccessibleButton(
            "Open Current FFT Window",
            "Open separate real-time FFT analysis window for controller-facing currents.",
        )
        btn_fft_window.clicked.connect(self._open_current_fft_window)
        current_sense_layout.addWidget(btn_fft_window)

        btn_fft_csv = AccessibleButton(
            "Save FFT Data CSV",
            "Save FFT frequency, magnitude, and phase data to CSV.",
        )
        btn_fft_csv.clicked.connect(self._save_current_fft_csv)
        current_sense_layout.addWidget(btn_fft_csv)

        btn_fft_image = AccessibleButton(
            "Save FFT Graph Image",
            "Save FFT magnitude and phase graphs to an image file.",
        )
        btn_fft_image.clicked.connect(self._save_current_fft_image)
        current_sense_layout.addWidget(btn_fft_image)

        self.current_sense_group.setLayout(current_sense_layout)
        self._update_bridge_visualization({})
        _adv_layout.addWidget(self.current_sense_group)

        self.mcu_budget_group = AccessibleGroupBox(
            "MCU Budget Estimator",
            "Estimate target MCU control-loop load from measured calculation duration.",
        )
        mcu_layout = QVBoxLayout()

        self.mcu_perf_ratio = LabeledSpinBox(
            "Host-to-MCU Slowdown",
            min_val=0.1,
            max_val=1000.0,
            initial=40.0,
            step=0.5,
            decimals=2,
            suffix=" x",
            description="Estimated slowdown from host runtime to target MCU runtime.",
        )
        mcu_layout.addWidget(self.mcu_perf_ratio)

        self.mcu_reference_clock_mhz = LabeledSpinBox(
            "Reference Clock",
            min_val=1.0,
            max_val=1000.0,
            initial=120.0,
            step=1.0,
            decimals=1,
            suffix=" MHz",
            description="Reference clock used for slowdown normalization.",
        )
        mcu_layout.addWidget(self.mcu_reference_clock_mhz)

        self.mcu_target_clock_1_mhz = LabeledSpinBox(
            "Target Clock 1",
            min_val=1.0,
            max_val=1000.0,
            initial=48.0,
            step=1.0,
            decimals=1,
            suffix=" MHz",
            description="First target MCU clock for load estimation.",
        )
        mcu_layout.addWidget(self.mcu_target_clock_1_mhz)

        self.mcu_target_clock_2_mhz = LabeledSpinBox(
            "Target Clock 2",
            min_val=1.0,
            max_val=1000.0,
            initial=72.0,
            step=1.0,
            decimals=1,
            suffix=" MHz",
            description="Second target MCU clock for load estimation.",
        )
        mcu_layout.addWidget(self.mcu_target_clock_2_mhz)

        self.mcu_target_clock_3_mhz = LabeledSpinBox(
            "Target Clock 3",
            min_val=1.0,
            max_val=1000.0,
            initial=120.0,
            step=1.0,
            decimals=1,
            suffix=" MHz",
            description="Third target MCU clock for load estimation.",
        )
        mcu_layout.addWidget(self.mcu_target_clock_3_mhz)

        self.mcu_budget_group.setLayout(mcu_layout)
        _adv_layout.addWidget(self.mcu_budget_group)

        self.pfc_group = AccessibleGroupBox(
            "Power Factor Correction",
            "Configure simulation-level closed-loop power factor telemetry control.",
        )
        pfc_layout = QVBoxLayout()

        self.pfc_mode = LabeledComboBox(
            "PFC Mode",
            items=["Disabled", "Enabled"],
            description="Enable or disable closed-loop power factor correction telemetry.",
        )
        pfc_layout.addWidget(self.pfc_mode)

        self.pfc_target_pf = LabeledSpinBox(
            "Target Power Factor",
            min_val=0.50,
            max_val=1.00,
            initial=0.95,
            step=0.01,
            decimals=3,
            suffix="",
            description="Desired target power factor for the PFC controller.",
        )
        pfc_layout.addWidget(self.pfc_target_pf)

        self.pfc_kp = LabeledSpinBox(
            "PFC Kp",
            min_val=0.0,
            max_val=10.0,
            initial=0.10,
            step=0.01,
            decimals=3,
            suffix="",
            description="Proportional gain for PFC compensation command.",
        )
        pfc_layout.addWidget(self.pfc_kp)

        self.pfc_ki = LabeledSpinBox(
            "PFC Ki",
            min_val=0.0,
            max_val=50.0,
            initial=1.0,
            step=0.1,
            decimals=3,
            suffix="",
            description="Integral gain for PFC compensation command.",
        )
        pfc_layout.addWidget(self.pfc_ki)

        self.pfc_max_var = LabeledSpinBox(
            "Max Compensation",
            min_val=10.0,
            max_val=200000.0,
            initial=10000.0,
            step=10.0,
            decimals=1,
            suffix=" VAR",
            description="Upper clamp for reactive compensation command.",
        )
        pfc_layout.addWidget(self.pfc_max_var)

        self.pfc_window_samples = LabeledSpinBox(
            "PF Window Samples",
            min_val=8,
            max_val=5000,
            initial=128,
            step=1,
            decimals=0,
            suffix="",
            description="Rolling window size used for power-factor metric estimation.",
        )
        pfc_layout.addWidget(self.pfc_window_samples)

        self.pfc_group.setLayout(pfc_layout)
        _adv_layout.addWidget(self.pfc_group)

        self.hardware_group = AccessibleGroupBox(
            "Communication Interface",
            "Optional communication backend settings (for example CAN/LIN style I/O). Disable to run pure software simulation.",  # noqa: E501
        )
        hardware_layout = QVBoxLayout()

        self.hardware_enable_backend = QCheckBox("Enable Communication Backend")
        self.hardware_enable_backend.setChecked(False)
        self.hardware_enable_backend.setAccessibleName("Enable hardware communication backend")
        self.hardware_enable_backend.setAccessibleDescription(
            "When checked, the simulation sends real-time telemetry to the selected "
            "hardware backend (e.g., serial, CAN). Disable for pure software simulation."
        )
        hardware_layout.addWidget(self.hardware_enable_backend)

        self.hardware_backend_type = LabeledComboBox(
            "Communication Backend",
            items=["Mock DAQ"],
            description="Communication backend implementation used for command/feedback I/O.",
        )
        hardware_layout.addWidget(self.hardware_backend_type)

        self.hardware_noise_std = LabeledSpinBox(
            "Mock Noise Std",
            min_val=0.0,
            max_val=10.0,
            initial=0.0,
            step=0.001,
            decimals=4,
            suffix=" V",
            description="Standard deviation of Gaussian feedback voltage noise for the Mock DAQ backend.",  # noqa: E501
        )
        hardware_layout.addWidget(self.hardware_noise_std)

        self.hardware_seed = LabeledSpinBox(
            "Mock Random Seed",
            min_val=0,
            max_val=1000000,
            initial=0,
            step=1,
            decimals=0,
            suffix="",
            description="Deterministic random seed used by the Mock DAQ noise generator.",
        )
        hardware_layout.addWidget(self.hardware_seed)

        self.hardware_group.setLayout(hardware_layout)
        _adv_layout.addWidget(self.hardware_group)

        _adv_layout.addStretch()
        _adv_widget.setLayout(_adv_layout)
        _adv_scroll = QScrollArea()
        _adv_scroll.setWidget(_adv_widget)
        _adv_scroll.setWidgetResizable(True)
        _adv_scroll.setAccessibleName("Advanced settings scroll area")
        _adv_scroll.setAccessibleDescription(
            "Inverter non-idealities, current sensing, MCU budget, PFC, "
            "and hardware backend settings."
        )
        self._advanced_scroll = _adv_scroll

        layout.addStretch()
        widget.setLayout(layout)
        return widget  # caller adds to the appropriate tab

    def _create_monitoring_tab(self):
        """Create real-time monitoring tab with speed curve."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Navigation instructions for accessibility
        instructions = QLabel(
            "📋 Monitoring Tab: Real-time motor values on the left, Speed profile on the right.\n"
            "Navigate with Tab key. In the monitoring list, use arrow keys or Tab+Shift to navigate values."  # noqa: E501
        )
        instructions.setWordWrap(True)
        instructions.setAccessibleName("Navigation Instructions")
        layout.addWidget(instructions)

        # Status group with more comprehensive monitoring
        group = AccessibleGroupBox(
            "Real-Time Monitoring",
            "Current motor state and performance metrics. Use arrow keys to navigate through values.",  # noqa: E501
        )
        group_layout = QHBoxLayout()

        # Left side: Status labels in accessible list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAccessibleName("Monitoring Values List")
        scroll.setAccessibleDescription(
            "Scrollable list of current motor state values. Use arrow keys to navigate."
        )

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        self.status_labels = {}
        self.status_blocks = {}  # Store text blocks for easier access
        status_items = [
            ("speed_rpm", "⚡ Rotor Speed", "RPM"),
            ("omega", "Angular Velocity", "rad/s"),
            ("theta", "Rotor Position", "rad"),
            ("currents_a", "Phase A Current", "A"),
            ("currents_b", "Phase B Current", "A"),
            ("currents_c", "Phase C Current", "A"),
            ("torque", "Electromagnetic Torque", "N·m"),
            ("back_emf_a", "Back-EMF Phase A", "V"),
            ("back_emf_b", "Back-EMF Phase B", "V"),
            ("back_emf_c", "Back-EMF Phase C", "V"),
            ("id_ref", "d-axis Current Ref", "A"),
            ("iq_ref", "q-axis Current Ref", "A"),
            ("speed_error", "Speed Error", "rad/s"),
            ("v_d_ff", "d-axis Feedforward", "V"),
            ("v_q_ff", "q-axis Feedforward", "V"),
            ("speed_loop_enabled", "Speed Loop Enabled", "0/1"),
            ("decouple_d_enabled", "D-axis Decoupling", "0/1"),
            ("decouple_q_enabled", "Q-axis Decoupling", "0/1"),
            ("observer_mode_code", "Observer Mode Code", "0/1/2"),
            ("theta_electrical", "Estimated Electrical Angle", "rad"),
            ("theta_meas_emf", "Back-EMF Angle", "rad"),
            ("theta_error_pll", "PLL Angle Error", "rad"),
            ("theta_error_smo", "SMO Angle Error", "rad"),
            ("smo_omega_est", "SMO Estimated Speed", "rad/s"),
            ("observer_confidence", "Observer Confidence", "0-1"),
            ("observer_confidence_emf", "Confidence from EMF", "0-1"),
            ("observer_confidence_speed", "Confidence from Speed", "0-1"),
            ("observer_confidence_coherence", "Confidence from Coherence", "0-1"),
            ("observer_confidence_ema", "Confidence EMA", "0-1"),
            ("observer_confidence_trend", "Confidence Trend", "delta"),
            (
                "observer_confidence_above_threshold_time_s",
                "Confidence Above Threshold Time",
                "s",
            ),
            (
                "observer_confidence_below_threshold_time_s",
                "Confidence Below Threshold Time",
                "s",
            ),
            ("observer_confidence_crossings_up", "Confidence Crossings Up", "count"),
            (
                "observer_confidence_crossings_down",
                "Confidence Crossings Down",
                "count",
            ),
            ("startup_sequence_enabled", "Startup Sequence Enabled", "0/1"),
            ("startup_phase_code", "Startup Phase Code", "0/1/2/3"),
            ("startup_sequence_elapsed_s", "Startup Sequence Elapsed", "s"),
            ("startup_phase_elapsed_s", "Startup Phase Elapsed", "s"),
            ("startup_handoff_count", "Startup Handoff Count", "count"),
            (
                "startup_last_handoff_time_s",
                "Last Handoff Time",
                "s",
            ),
            (
                "startup_last_handoff_confidence",
                "Last Handoff Confidence",
                "0-1",
            ),
            (
                "startup_handoff_confidence_peak",
                "Handoff Confidence Peak",
                "0-1",
            ),
            ("startup_handoff_quality", "Handoff Quality KPI", "0-1"),
            (
                "startup_handoff_stability_ratio",
                "Handoff Stability Ratio",
                "0-1",
            ),
            ("pfc_enabled", "PFC Enabled", "0/1"),
            ("pfc_target_pf", "PFC Target PF", "0-1"),
            ("pfc_power_factor", "Input Power Factor", "-1..1"),
            ("pfc_active_power_w", "Input Active Power", "W"),
            ("pfc_reactive_power_var", "Input Reactive Power", "VAR"),
            ("pfc_command_var", "PFC Compensation Command", "VAR"),
            ("efficiency", "System Efficiency", "0-1"),
            ("mechanical_output_power_w", "Mechanical Output Power", "W"),
            ("total_loss_power_w", "Estimated Total Loss", "W"),
            ("effective_dc_voltage", "Effective DC-Link Voltage", "V"),
            ("dc_link_ripple_v", "DC-Link Ripple", "V"),
            ("dc_link_bus_current_a", "DC-Link Bus Current", "A"),
            ("inverter_total_loss_power_w", "Inverter Total Loss", "W"),
            ("junction_temperature_c", "Inverter Junction Temp", "C"),
            ("common_mode_voltage", "Common-Mode Voltage", "V"),
            ("control_calc_duration_us", "Control Calc Duration", "us"),
            ("control_cpu_load_pct", "Control CPU Load", "%"),
            ("control_cpu_load_avg_pct", "Control CPU Load Avg", "%"),
            ("mcu_load_target_1_pct", "MCU Load @ Target 1", "%"),
            ("mcu_load_target_2_pct", "MCU Load @ Target 2", "%"),
            ("mcu_load_target_3_pct", "MCU Load @ Target 3", "%"),
            ("hardware_enabled", "Communication Enabled", "0/1"),
            ("hardware_connected", "Communication Connected", "0/1"),
            ("hardware_backend_code", "Communication Backend Code", "0/1/2"),
            ("hardware_write_count", "Communication Write Count", "count"),
            ("hardware_read_count", "Communication Read Count", "count"),
            ("hardware_io_error_flag", "Communication I/O Error Flag", "0/1"),
            ("time", "Simulation Time", "s"),
        ]

        total_items = len(status_items)
        for idx, (key, name, unit) in enumerate(status_items):
            # Create accessible text block component
            text_block = AccessibleTextBlock(name, unit, idx, total_items)
            self.status_blocks[key] = text_block
            self.status_labels[key] = text_block.value_label  # Keep for backward compat
            scroll_layout.addWidget(text_block)

        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        group_layout.addWidget(scroll, 1)

        # Right side: Real-time multi-channel oscilloscope
        from src.ui.widgets.oscilloscope_widget import OscilloscopeWidget

        self.oscilloscope = OscilloscopeWidget()
        self.oscilloscope.setAccessibleName("Real-time Oscilloscope")
        self.oscilloscope.setAccessibleDescription(
            "Multi-channel live oscilloscope. "
            "Select signal channels from the drop-downs, choose a time window, "
            "pause/resume the display, or reset the Y-axis scale at any time."
        )
        group_layout.addWidget(self.oscilloscope, 2)

        group.setLayout(group_layout)
        layout.addWidget(group)

        widget.setLayout(layout)
        return widget  # Analysis tab picks this up

    def _create_plotting_tab(self):
        """Create plotting and visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        group = AccessibleGroupBox("Visualization", "Generate plots of simulation results.")
        group_layout = QVBoxLayout()

        info = QLabel(
            "Plots are generated from recorded simulation data.\n"
            "Run simulation first, then select plot type or custom variables to generate visualization."  # noqa: E501
        )
        info.setWordWrap(True)
        group_layout.addWidget(info)

        # Grid controls for plots
        grid_layout = QHBoxLayout()
        from PySide6.QtWidgets import QCheckBox

        self.plot_grid_checkbox = QCheckBox("Show Grid")
        self.plot_grid_checkbox.setAccessibleName("Show major grid on plots")
        self.plot_grid_checkbox.setAccessibleDescription(
            "When checked, draws major grid lines aligned to axis ticks on all generated plots."
        )
        self.plot_grid_checkbox.setChecked(True)
        grid_layout.addWidget(self.plot_grid_checkbox)

        self.plot_minor_grid_checkbox = QCheckBox("Minor Grid")
        self.plot_minor_grid_checkbox.setAccessibleName("Show minor grid on plots")
        self.plot_minor_grid_checkbox.setAccessibleDescription(
            "When checked, draws a finer minor grid between major tick lines "
            "on all generated plots."
        )
        self.plot_minor_grid_checkbox.setChecked(False)
        grid_layout.addWidget(self.plot_minor_grid_checkbox)

        self.plot_grid_spacing = LabeledSpinBox(
            "Grid Spacing (s)",
            0.001,
            10.0,
            0.5,
            0.001,
            3,
            " s",
            "Grid spacing in seconds (X-axis)",
        )
        grid_layout.addWidget(self.plot_grid_spacing)

        self.plot_grid_spacing_y = LabeledSpinBox(
            "Y Spacing",
            0.0,
            100.0,
            0.0,
            0.1,
            2,
            " units",
            "Grid spacing for Y-axis (0=auto)",
        )
        grid_layout.addWidget(self.plot_grid_spacing_y)

        group_layout.addLayout(grid_layout)

        # custom variable selector
        self.plot_var_list = AccessibleTableWidget(
            "Variable Selection",
            "Select variables to plot. Use arrow keys to navigate, Space or Enter to select/deselect items.",  # noqa: E501
        )
        self.plot_var_list.setColumnCount(1)
        self.plot_var_list.setHorizontalHeaderLabels(["Variable"])
        self.plot_var_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plot_var_list.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        # will populate when simulation ends / when plotting
        group_layout.addWidget(self.plot_var_list)

        button_layout = QHBoxLayout()

        btn_plot_3phase = AccessibleButton(
            "Plot 3-Phase Overview",
            "Generate comprehensive 3-phase motor variables plot",
        )
        btn_plot_3phase.clicked.connect(self._plot_3phase)
        button_layout.addWidget(btn_plot_3phase)

        btn_plot_current = AccessibleButton(
            "Plot Currents", "Generate detailed 3-phase current analysis plot"
        )
        btn_plot_current.clicked.connect(self._plot_currents)
        button_layout.addWidget(btn_plot_current)

        btn_plot_pfc = AccessibleButton(
            "Plot PFC Analysis",
            "Generate power-factor, active/reactive power, and command trends",
        )
        btn_plot_pfc.clicked.connect(self._plot_pfc_analysis)
        button_layout.addWidget(btn_plot_pfc)

        btn_plot_efficiency = AccessibleButton(
            "Plot Efficiency Analysis",
            "Generate input power, output power, loss, and efficiency trends",
        )
        btn_plot_efficiency.clicked.connect(self._plot_efficiency_analysis)
        button_layout.addWidget(btn_plot_efficiency)

        btn_plot_inverter = AccessibleButton(
            "Plot Inverter Analysis",
            "Generate bus voltage, inverter loss, temperature, and common-mode trends",
        )
        btn_plot_inverter.clicked.connect(self._plot_inverter_analysis)
        button_layout.addWidget(btn_plot_inverter)

        btn_efficiency_tips = AccessibleButton(
            "Suggest Efficiency Tuning",
            "Show heuristic efficiency recommendations from current settings",
        )
        btn_efficiency_tips.clicked.connect(self._show_efficiency_recommendations)
        button_layout.addWidget(btn_efficiency_tips)

        btn_plot_measured_vs_true = AccessibleButton(
            "Plot Measured vs True Currents",
            "Overlay measured and true physics phase currents with per-phase RMS error (requires current sensing enabled)",  # noqa: E501
        )
        btn_plot_measured_vs_true.clicked.connect(self._plot_measured_vs_true)
        button_layout.addWidget(btn_plot_measured_vs_true)

        btn_plot_custom = AccessibleButton(
            "Plot Selected", "Generate plot for user-selected variables"
        )
        btn_plot_custom.clicked.connect(self._plot_custom)
        button_layout.addWidget(btn_plot_custom)

        group_layout.addLayout(button_layout)

        # ── Customize buttons (post-sim plot style editor) ─────────────
        customize_info = QLabel(
            "After generating a plot, click the matching Customize button to open the "
            "style editor (font sizes, colors, line widths, DPI, export presets)."
        )
        customize_info.setWordWrap(True)
        customize_info.setAccessibleName("Customize plots information")
        group_layout.addWidget(customize_info)

        customize_layout = QHBoxLayout()

        _cust_specs = [
            (
                "Customize 3-Phase",
                "_last_fig_3phase",
                "Open style editor for the 3-phase overview plot",
            ),
            (
                "Customize Currents",
                "_last_fig_currents",
                "Open style editor for the current analysis plot",
            ),
            ("Customize PFC", "_last_fig_pfc", "Open style editor for the PFC analysis plot"),
            (
                "Customize Efficiency",
                "_last_fig_efficiency",
                "Open style editor for the efficiency analysis plot",
            ),
            (
                "Customize Inverter",
                "_last_fig_inverter",
                "Open style editor for the inverter analysis plot",
            ),
            (
                "Customize Meas/True",
                "_last_fig_measured_vs_true",
                "Open style editor for the measured-vs-true current plot",
            ),
            (
                "Customize Selected",
                "_last_fig_custom",
                "Open style editor for the custom multi-axis plot",
            ),
        ]
        for _label, _attr, _desc in _cust_specs:
            _btn = AccessibleButton(_label, _desc)
            _btn.clicked.connect(lambda _checked=False, _a=_attr: self._open_plot_customizer(_a))
            customize_layout.addWidget(_btn)

        group_layout.addLayout(customize_layout)
        group_layout.addStretch()

        group.setLayout(group_layout)
        layout.addWidget(group)

        widget.setLayout(layout)
        return widget  # Analysis tab picks this up

    def _open_plot_customizer(self, fig_attr: str) -> None:
        """Open PlotCustomizerDialog for the figure stored in *fig_attr*."""
        fig = getattr(self, fig_attr, None)
        if fig is None:
            from PySide6.QtWidgets import QMessageBox as _QMB

            _QMB.information(
                self,
                "No Plot Available",
                "Generate the corresponding plot first, then click Customize.",
            )
            return
        try:
            from src.ui.widgets.plot_customizer_dialog import PlotCustomizerDialog

            dlg = PlotCustomizerDialog(figure=fig, parent=self)
            dlg.exec()
        except Exception as _exc:
            logger.error("PlotCustomizerDialog error: %s", _exc)

    # ------------------------------------------------------------------
    # Calibration tab
    # ------------------------------------------------------------------

    def _create_calibration_tab(self):
        """Create the FW loaded-point calibration tab."""
        widget = QScrollArea()
        inner = QWidget()
        layout = QVBoxLayout()

        # ── Motor Profile & Session selection ────────────────────────
        profile_group = AccessibleGroupBox(
            "Motor Profile & Session",
            "Select the motor profile and auto-tuning session for calibration",
        )
        pg_layout = QVBoxLayout()

        profiles = sorted(
            p for p in MOTOR_PROFILES_DIR.glob("*.json") if not p.name.startswith("_")
        )
        profile_names = [p.name for p in profiles] or ["(no profiles found)"]

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Motor profile:"))
        self.calib_profile_combo = QComboBox()
        self.calib_profile_combo.addItems(profile_names)
        self.calib_profile_combo.setAccessibleName("Motor profile for calibration")
        self.calib_profile_combo.setAccessibleDescription(
            "Select motor profile JSON for field-weakening loaded-point calibration"
        )
        self.calib_profile_combo.currentTextChanged.connect(self._on_calib_profile_changed)
        row1.addWidget(self.calib_profile_combo, 1)
        pg_layout.addLayout(row1)

        self.calib_session_label = QLabel("Session: (auto-detected)")
        self.calib_session_label.setAccessibleName("Tuning session file")
        self.calib_session_label.setAccessibleDescription(
            "Path to the tuning session file for the selected motor profile"
        )
        pg_layout.addWidget(self.calib_session_label)

        self.calib_output_label = QLabel("Output: (auto)")
        self.calib_output_label.setAccessibleName("Calibration output file path")
        self.calib_output_label.setAccessibleDescription(
            "Path where calibration results will be saved"
        )
        pg_layout.addWidget(self.calib_output_label)

        profile_group.setLayout(pg_layout)
        layout.addWidget(profile_group)

        # ── Auto-Calibrate All — single action for full analytic pipeline ──
        auto_calib_group = AccessibleGroupBox(
            "Full Auto-Calibration (Analytic + Physics)",
            "Run both calibration stages for all motor profiles: "
            "step 1 analytically computes FOC PI gains (current/speed loops) "
            "from motor parameters (L, R, J, B, Kt), then refines them with a "
            "frequency-domain grid search; step 2 derives field-weakening "
            "parameters physically from nameplate data (Ke, rated speed, rated "
            "current, Vdc) and validates them with a closed-loop sweep.",
        )
        auto_btn_row = QHBoxLayout()

        self.btn_auto_calib_all = AccessibleButton(
            "Auto-Calibrate",
            "Run complete auto-calibration pipeline: "
            "Stage 1 analytically computes FOC PI gains for all motor profiles; "
            "Stage 2 derives physics-based field-weakening parameters for all profiles",
        )
        self.btn_auto_calib_all.clicked.connect(self._start_auto_calibrate_all)
        auto_btn_row.addWidget(self.btn_auto_calib_all)

        self.btn_stop_auto_calib = AccessibleButton(
            "Stop Auto-Calibration",
            "Terminate the running auto-calibration pipeline",
        )
        self.btn_stop_auto_calib.setEnabled(False)
        self.btn_stop_auto_calib.clicked.connect(self._stop_auto_calibrate_all)
        auto_btn_row.addWidget(self.btn_stop_auto_calib)

        # Backward-compatible aliases for hidden single-profile calibration controls.
        self.btn_start_calib = self.btn_auto_calib_all
        self.btn_stop_calib = self.btn_stop_auto_calib

        auto_btn_row.addStretch()
        auto_calib_group.setLayout(auto_btn_row)
        layout.addWidget(auto_calib_group)

        # ── Progress log ─────────────────────────────────────────────
        prog_group = AccessibleGroupBox(
            "Calibration Progress",
            "Live output from the calibration process",
        )
        prog_layout = QVBoxLayout()
        self.calib_log = QTextEdit()
        self.calib_log.setReadOnly(True)
        self.calib_log.setMinimumHeight(220)
        mono_font = QFont("Courier New", 9)
        self.calib_log.setFont(mono_font)
        self.calib_log.setAccessibleName("Calibration progress log")
        self.calib_log.setAccessibleDescription(
            "Live text output from calibration; key milestones are announced via audio"
        )
        prog_layout.addWidget(self.calib_log)
        prog_group.setLayout(prog_layout)
        layout.addWidget(prog_group)

        # ── Results panel ────────────────────────────────────────────
        result_group = AccessibleGroupBox(
            "Calibration Results",
            "Key performance metrics from the last completed calibration run",
        )
        res_layout = QVBoxLayout()
        self.calib_result_status = QLabel("Status: Not run")
        self.calib_result_status.setAccessibleName("Calibration Status")
        self.calib_result_status.setAccessibleDescription(
            "Overall status of the auto-calibration pipeline"
        )
        self.calib_result_speed = QLabel("Achieved Speed: --")
        self.calib_result_speed.setAccessibleName("Achieved Speed")
        self.calib_result_speed.setAccessibleDescription(
            "Maximum motor speed achieved during calibration in RPM"
        )
        self.calib_result_load = QLabel("Load Torque: --")
        self.calib_result_load.setAccessibleName("Load Torque")
        self.calib_result_load.setAccessibleDescription(
            "Applied load torque during the calibration run in Newton-meters"
        )
        self.calib_result_efficiency = QLabel("Efficiency: --")
        self.calib_result_efficiency.setAccessibleName("Efficiency")
        self.calib_result_efficiency.setAccessibleDescription(
            "Motor efficiency achieved at calibration operating point in percent"
        )
        self.calib_result_fw = QLabel("FW Injection: --")
        self.calib_result_fw.setAccessibleName("Field Weakening Injection")
        self.calib_result_fw.setAccessibleDescription(
            "Field weakening current injection at the calibrated high-speed point in amperes"
        )
        for lbl in (
            self.calib_result_status,
            self.calib_result_speed,
            self.calib_result_load,
            self.calib_result_efficiency,
            self.calib_result_fw,
        ):
            res_layout.addWidget(lbl)
        res_layout.addStretch()
        result_group.setLayout(res_layout)
        layout.addWidget(result_group)

        inner.setLayout(layout)
        widget.setWidget(inner)
        widget.setWidgetResizable(True)
        return widget  # Analysis tab picks this up

        # Prime session/output labels for the initial selection
        if profile_names[0] != "(no profiles found)":
            self._on_calib_profile_changed(profile_names[0])

    # ── Five composite tab builders ────────────────────────────────────────────

    def _create_motor_drive_tab(self) -> None:
        """Tab 1 — Motor & Drive: Motor Parameters + Load Profile + Supply Profile as sub-tabs."""
        container = QWidget()
        container.setAccessibleName("Motor and Drive configuration")
        container.setAccessibleDescription(
            "Configure motor electrical and mechanical parameters, "
            "load profile, and supply voltage."
        )
        sub_tabs = AccessibleTabWidget()
        sub_tabs.setAccessibleName("Motor and Drive sub-tabs")
        sub_tabs.setAccessibleDescription(
            "Three configuration areas: motor parameters, load profile, and supply voltage profile."
        )
        motor_w = self._create_parameters_tab()
        load_w = self._create_load_tab()
        supply_w = self._create_supply_tab()
        sub_tabs.addTab(motor_w, "Motor")
        sub_tabs.addTab(load_w, "Load")
        sub_tabs.addTab(supply_w, "Supply")
        lay = QVBoxLayout()
        lay.addWidget(sub_tabs)
        container.setLayout(lay)
        self.tabs.addTab(container, "Motor & Drive")

    def _create_controller_tab(self) -> None:
        """Tab 2 — Controller: simulation duration + V/f + FOC current loops + FW + PI gains."""
        ctrl_widget = self._create_control_tab()  # returns widget (no addTab)
        scroll = QScrollArea()
        scroll.setWidget(ctrl_widget)
        scroll.setWidgetResizable(True)
        scroll.setAccessibleName("Controller settings scroll area")
        scroll.setAccessibleDescription(
            "Simulation duration, control mode (V/f or FOC), current references, "
            "field weakening, speed loop, PI gains, and decoupling settings."
        )
        container = QWidget()
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)
        container.setLayout(lay)
        self.tabs.addTab(container, "Controller")

    def _create_observer_startup_tab(self) -> None:
        """Tab 3 — Observer & Startup: context-sensitive observer params + startup sequence."""
        container = QWidget()
        container.setAccessibleName("Observer and Startup configuration")

        scroll = QScrollArea()
        inner = QWidget()
        layout = QVBoxLayout()

        # Intro label
        intro = QLabel(
            "Select the angle observer below. "
            "Only the parameters relevant to the chosen observer are shown."
        )
        intro.setWordWrap(True)
        intro.setAccessibleName("Observer tab introduction")
        layout.addWidget(intro)

        # Observer group (built in _create_control_tab, stored as self.foc_observer_group)
        layout.addWidget(self.foc_observer_group)

        # Startup group (built in _create_control_tab, stored as self.foc_startup_group)
        layout.addWidget(self.foc_startup_group)

        layout.addStretch()
        inner.setLayout(layout)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setAccessibleName("Observer and Startup scroll area")

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        container.setLayout(outer)

        # Connect context-sensitive visibility
        self.foc_angle_observer_mode.currentTextChanged.connect(self._on_observer_mode_changed)
        # Apply initial state
        self._on_observer_mode_changed(self.foc_angle_observer_mode.currentText())

        self.tabs.addTab(container, "Observer & Startup")

    def _create_advanced_tab(self) -> None:
        """Tab 4 — Advanced: Inverter, Current Sense, MCU Budget, PFC, Hardware Backend."""
        container = QWidget()
        container.setAccessibleName("Advanced settings")
        container.setAccessibleDescription(
            "Inverter non-idealities, current sensing model, MCU budget estimator, "
            "power factor correction, and hardware communication backend."
        )
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._advanced_scroll)  # built by _create_control_tab
        container.setLayout(outer)
        self.tabs.addTab(container, "Advanced")

    def _create_analysis_tab(self) -> None:
        """Tab 5 — Analysis: Monitoring + Plotting + Calibration as sub-tabs."""
        container = QWidget()
        container.setAccessibleName("Analysis and results")
        container.setAccessibleDescription(
            "Real-time monitoring, plot generation, and auto-calibration."
        )
        sub_tabs = AccessibleTabWidget()
        sub_tabs.setAccessibleName("Analysis sub-tabs")
        sub_tabs.setAccessibleDescription(
            "Three sections: real-time monitoring, plot generation, and calibration."
        )
        mon_w = self._create_monitoring_tab()
        plot_w = self._create_plotting_tab()
        cal_w = self._create_calibration_tab()
        sub_tabs.addTab(mon_w, "Monitoring")
        sub_tabs.addTab(plot_w, "Plotting")
        sub_tabs.addTab(cal_w, "Calibration")
        lay = QVBoxLayout()
        lay.addWidget(sub_tabs)
        container.setLayout(lay)
        self.tabs.addTab(container, "Analysis")

    # ── Observer context-sensitive visibility ──────────────────────────────────

    def _on_observer_mode_changed(self, mode: str) -> None:  # noqa: C901
        """Show/hide observer-specific parameter widgets based on selected observer mode.

        Called whenever the Angle Observer dropdown changes. Groups are sub-sections
        of self.foc_observer_group that contain per-observer parameters.
        """
        # All observer-specific spinboxes — grouped by observer type
        _pll_widgets = [self.foc_pll_kp, self.foc_pll_ki]
        _smo_widgets = [self.foc_smo_k_slide, self.foc_smo_lpf_alpha, self.foc_smo_boundary]
        _stsmo_widgets = [
            self.foc_stsmo_k1,
            self.foc_stsmo_k2_min,
            self.foc_stsmo_k2_factor,
            self.foc_stsmo_rated_rpm,
            self.foc_stsmo_autocalib_btn,
            self.foc_solver_mode,
        ]
        _af_widgets = [self.foc_af_dc_cutoff]

        # First hide everything
        for w in _pll_widgets + _smo_widgets + _stsmo_widgets + _af_widgets:
            w.hide()

        # Show only what the selected observer needs
        if mode == "PLL":
            for w in _pll_widgets:
                w.show()
        elif mode == "SMO":
            for w in _smo_widgets:
                w.show()
        elif mode == "STSMO":
            for w in _stsmo_widgets:
                w.show()
        elif mode == "ActiveFlux":
            for w in _af_widgets:
                w.show()
        elif mode == "Auto (recommend from motor)":
            # Show all observer parameters so the user can review the
            # auto-calibrated gains before running the simulation.
            # The actual observer will be selected when Auto Calibrate runs.
            for w in _pll_widgets + _smo_widgets + _stsmo_widgets + _af_widgets:
                w.show()
        # "Measured" → all hidden (no observer params needed)

    def _on_calib_profile_changed(self, profile_name: str) -> None:
        """Update session and output path labels when profile selection changes."""
        stem = Path(profile_name).stem
        session_dir = MOTOR_PROFILES_DIR.parent / "tuning_sessions" / "until_converged"
        session_path = session_dir / f"{stem}_until_converged.json"
        out_path = MOTOR_PROFILES_DIR.parent / "logs" / f"calibration_{stem}_fw_loaded_point.json"
        self.calib_output_path = out_path
        exists_tag = "" if session_path.exists() else " ⚠ not found"
        self.calib_session_label.setText(f"Session: {session_path.name}{exists_tag}")
        self.calib_output_label.setText(f"Output: {out_path.name}")

    def _start_calibration(self) -> None:
        """Launch the legacy single-profile field-weakening calibration backend."""
        if not self._can_start_task("calibration"):
            return

        profile_name = self.calib_profile_combo.currentText().strip()
        if not profile_name or profile_name == "(no profiles found)":
            QMessageBox.warning(
                self,
                "Calibration Profile Missing",
                "Select a motor profile before starting calibration.",
            )
            speak("Select a motor profile before starting calibration.")
            return

        profile_path = MOTOR_PROFILES_DIR / profile_name
        if not profile_path.exists():
            QMessageBox.critical(
                self,
                "Calibration Profile Not Found",
                f"Motor profile file was not found:\n{profile_path}",
            )
            speak("Selected motor profile file was not found.")
            return

        stem = profile_path.stem
        session_path = (
            MOTOR_PROFILES_DIR.parent
            / "tuning_sessions"
            / "until_converged"
            / f"{stem}_until_converged.json"
        )
        if not session_path.exists():
            QMessageBox.critical(
                self,
                "Calibration Session Not Found",
                f"Tuning session file was not found:\n{session_path}",
            )
            speak("Required tuning session file was not found.")
            return

        self.calib_output_path = (
            MOTOR_PROFILES_DIR.parent / "logs" / f"calibration_{stem}_fw_loaded_point.json"
        )
        self.calib_log.clear()
        self.calib_log.append(
            f"Starting single-profile field-weakening calibration for {profile_name}."
        )

        script_path = (
            Path(__file__).resolve().parents[2] / "examples" / "calibrate_fw_loaded_point.py"
        )
        self.calib_process = QProcess(self)
        self.calib_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.calib_process.readyReadStandardOutput.connect(self._on_calib_output)
        self.calib_process.finished.connect(self._on_calib_finished)
        self.calib_process.start(
            sys.executable,
            [
                str(script_path),
                "--profile",
                str(profile_path),
                "--session",
                str(session_path),
                "--output",
                str(self.calib_output_path),
            ],
        )
        self.btn_start_calib.setEnabled(False)
        self.btn_stop_calib.setEnabled(True)
        self._mark_task_running("calibration")
        self.status_bar_state.setText("State: Running")
        self.status_bar_task.setText("Task: Calibration")
        self.status_bar_time_remaining.setText("Remaining: -- s")
        speak("Calibration started.")

    def _on_calib_output(self) -> None:
        """Append single-profile calibration output to the log and announce milestones."""
        if self.calib_process is None:
            return

        raw = self.calib_process.readAllStandardOutput().data()
        text = bytes(raw).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.calib_log.append(line)

            if line.startswith("STEP1_START"):
                speak("Calibration step 1 started.")
            elif line.startswith("STEP2_START"):
                speak("Calibration step 2 started.")
            elif line.startswith("STEP3_START"):
                speak("Calibration step 3 started.")
            elif line.startswith("TORQUE_OK"):
                torque = line.split(maxsplit=1)[1] if " " in line else ""
                speak(f"Torque target reached at {torque}.")
            elif line.startswith("TORQUE_FAIL"):
                torque = line.split(maxsplit=1)[1] if " " in line else ""
                speak(f"Torque target not reached at {torque}.")
            elif line.startswith("REPORT_SAVED"):
                speak("Calibration report saved.")

        sb = self.calib_log.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _on_calib_finished(self, exit_code: int, exit_status) -> None:
        """Finalize single-profile calibration UI state and parse the result report."""
        if exit_code != 0:
            self.calib_result_status.setText(f"Status: Failed (exit {exit_code})")
            speak("Calibration failed.")
            self._reset_legacy_calib_ui()
            return

        if self.calib_output_path is None or not self.calib_output_path.exists():
            self.calib_result_status.setText("Status: Failed (report missing)")
            speak("Calibration failed because the result report was not found.")
            self._reset_legacy_calib_ui()
            return

        try:
            report = json.loads(self.calib_output_path.read_text(encoding="utf-8"))
        except Exception:
            self.calib_result_status.setText("Status: Error reading report")
            speak("Calibration report could not be read.")
            self._reset_legacy_calib_ui()
            return

        step3 = report.get("step3_final_working_point_tuning", {})
        metrics = step3.get("result_high_fidelity", {}).get("metrics", {})
        target_load = step3.get("target_load_nm", "--")
        mean_speed = metrics.get("mean_speed_rpm_last_1s", "--")
        efficiency = metrics.get("efficiency_pct_last_1s", "--")
        fw_injection = metrics.get("fw_injection_dc_a_last_1s", "--")
        passed = bool(step3.get("success", False))

        self.calib_result_status.setText(
            "Status: PASS" if passed else "Status: Complete (review results)"
        )
        self.calib_result_speed.setText(f"Achieved Speed: {mean_speed}")
        self.calib_result_load.setText(f"Load Torque: {target_load}")
        self.calib_result_efficiency.setText(f"Efficiency: {efficiency}")
        self.calib_result_fw.setText(f"FW Injection: {fw_injection}")
        speak(f"Calibration finished. Achieved speed {mean_speed} RPM.")
        self._reset_legacy_calib_ui()

    def _stop_calibration(self) -> None:
        """Stop the legacy single-profile calibration process if it is running."""
        if self.calib_process is not None:
            self._terminate_process(self.calib_process)
        self.calib_log.append("\n[Calibration stopped]\n")
        self.calib_result_status.setText("Status: Stopped")
        speak("Calibration stopped.")
        self._reset_legacy_calib_ui()

    def _reset_legacy_calib_ui(self) -> None:
        """Reset hidden single-profile calibration state."""
        self.calib_process = None
        self.btn_start_calib.setEnabled(True)
        self.btn_stop_calib.setEnabled(False)
        self.status_bar_state.setText("State: Stopped")
        self.status_bar_task.setText("Task: None")
        self.status_bar_time_remaining.setText("Remaining: -- s")
        self._mark_task_finished("calibration")

    # ------------------------------------------------------------------
    # Auto-Calibrate All (Analytic + Physics) — two-stage pipeline
    # ------------------------------------------------------------------

    def _start_auto_calibrate_all(self) -> None:
        """Launch the full two-stage analytic/physics auto-calibration pipeline.

        Stage 1 – ``auto_calibrate_all_motors.py``:
            Derives FOC PI gains analytically from motor parameters
            (L, R, J, B, Kt) then refines them with a frequency-domain
            multi-resolution grid search for all profiles.

        Stage 2 – ``auto_calibrate_fw_all_motors.py``:
            Derives field-weakening parameters physically from nameplate
            data (Ke, rated speed, rated current, Vdc) and validates them
            with a closed-loop simulation sweep for all profiles.
        """
        if not self._can_start_task("auto_calibration"):
            return

        self.calib_log.clear()
        self.calib_log.append(
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  AUTO-CALIBRATE ALL  —  Analytic + Physics pipeline          ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
            "\n[Stage 1/2]  FOC PI gains — analytic initial guess + frequency-domain refinement\n"
        )
        self.calib_result_status.setText("Status: Running — Stage 1/2 (FOC PI gains)…")
        self.btn_auto_calib_all.setEnabled(False)
        self.btn_stop_auto_calib.setEnabled(True)
        self._mark_task_running("auto_calibration")
        self.status_bar_state.setText("State: Running")
        self.status_bar_task.setText("Task: Auto-Calibration (Stage 1/2)")
        self.status_bar_time_remaining.setText("Remaining: -- s")

        script_path = (
            Path(__file__).resolve().parents[2] / "examples" / "auto_calibrate_all_motors.py"
        )
        self._auto_calib_stage = 1
        self.auto_calib_process = QProcess(self)
        self.auto_calib_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.auto_calib_process.readyReadStandardOutput.connect(self._on_auto_calib_output)
        self.auto_calib_process.finished.connect(self._on_auto_calib_step1_finished)
        self.auto_calib_process.start(sys.executable, [str(script_path)])
        speak("Auto-calibration stage one started: analytic FOC PI gains for all motors.")

    def _on_auto_calib_output(self) -> None:
        """Stream stdout from the running auto-calibration process to the log."""
        if self.auto_calib_process is None:
            return
        raw = self.auto_calib_process.readAllStandardOutput().data()
        text = bytes(raw).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.calib_log.append(line)
        sb = self.calib_log.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())
        # Speak key analytic milestones for accessibility
        lower = text.lower()
        if "analytical initial guess" in lower:
            speak("Analytic initial guess computed.")
        if "optimized gains" in lower:
            speak("Gain optimisation complete.")
        if "calibration complete" in lower or "all motors done" in lower:
            speak("All motors calibrated.")

    def _on_auto_calib_step1_finished(self, exit_code: int, exit_status) -> None:
        """Handle completion of stage 1 and launch stage 2 if successful."""
        if self._auto_calib_stage != 1:
            return  # Already cancelled

        if exit_code != 0:
            self.calib_log.append(
                f"\n[Stage 1 FAILED — exit code {exit_code}. Pipeline aborted.]\n"
            )
            self.calib_result_status.setText(f"Status: Failed at Stage 1 (exit {exit_code})")
            speak("Auto-calibration stage one failed. Pipeline aborted.")
            self._reset_auto_calib_ui()
            return

        self.calib_log.append(
            "\n[Stage 1 complete ✓]\n"
            "\n[Stage 2/2]  Field-weakening — physics-based derivation + closed-loop validation\n"
        )
        self.calib_result_status.setText("Status: Running — Stage 2/2 (Field-weakening)…")
        self.status_bar_task.setText("Task: Auto-Calibration (Stage 2/2)")
        speak(
            "Stage one complete. Starting stage two: physics-based field-weakening for all motors."
        )

        script_path = (
            Path(__file__).resolve().parents[2] / "examples" / "auto_calibrate_fw_all_motors.py"
        )
        self._auto_calib_stage = 2
        # Disconnect step-1 finished signal before reusing the process slot
        self.auto_calib_process = QProcess(self)
        self.auto_calib_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.auto_calib_process.readyReadStandardOutput.connect(self._on_auto_calib_output)
        self.auto_calib_process.finished.connect(self._on_auto_calib_step2_finished)
        self.auto_calib_process.start(sys.executable, [str(script_path)])

    def _on_auto_calib_step2_finished(self, exit_code: int, exit_status) -> None:
        """Handle completion of the full two-stage pipeline."""
        if self._auto_calib_stage != 2:
            return  # Already cancelled

        if exit_code == 0:
            self.calib_log.append("\n[Stage 2 complete ✓]\n\n[Auto-calibration pipeline DONE]\n")
            self.calib_result_status.setText("Status: Complete ✓ (Analytic + Physics)")
            # ── Apply all calibrated values to the GUI ─────────────────────
            # 1. Load Stage-1 current/speed PI gains from saved JSON
            self._load_stage1_gains_to_gui()
            # 2. Load Stage-2 field-weakening parameters from saved JSON
            self._load_stage2_fw_to_gui()
            # 3. Compute observer gains analytically + startup sequence
            #    + EEMF/SOGI/decoupling toggles for the current motor
            self._apply_observer_calibration_to_gui()
            speak(
                "Auto-calibration pipeline complete. "
                "FOC PI gains, field-weakening parameters, observer gains, "
                "and startup sequence updated for the current motor."
            )
            status_bar = self.statusBar()
            if status_bar is not None:
                status_bar.showMessage(
                    "Auto-calibration complete — all motor profiles and observer gains updated.",
                    10000,
                )
        else:
            self.calib_log.append(f"\n[Stage 2 FAILED — exit code {exit_code}]\n")
            self.calib_result_status.setText(f"Status: Failed at Stage 2 (exit {exit_code})")
            speak("Auto-calibration stage two failed.")

        self._reset_auto_calib_ui()

    @staticmethod
    def _recommend_observer(
        is_salient: bool,
        omega_e_max: float,
        v_bus: float,
        ke: float,
    ) -> str:
        """Recommend the most appropriate sensorless observer for a motor profile.

        Decision tree (in priority order):
        1. Salient-pole (IPM, Lq/Ld > 1.2) → ``"ActiveFlux"``
           ActiveFlux integrator is immune to cross-saturation bias and handles
           the reluctance-torque term naturally in field-weakening.
        2. Isotropic, high back-EMF speed (ωe_max > 1 500 rad/s) → ``"PLL"``
           Phase-locked loop is simple and very accurate when back-EMF is
           well above the noise floor across most of the speed range.
        3. Isotropic, lower speed range → ``"SMO"``
           First-order SMO is more robust than PLL when back-EMF is small
           and provides adequate accuracy up to moderate speeds.

        Parameters
        ----------
        is_salient : bool
            True when Lq/Ld > 1.2 (IPM motor).
        omega_e_max : float
            Maximum electrical angular speed [rad/s] at rated mechanical speed.
        v_bus : float
            DC bus voltage [V] (reserved for future SNR-based logic).
        ke : float
            Back-EMF constant [Vs/rad] (reserved for future SNR-based logic).

        Returns
        -------
        str
            One of ``"ActiveFlux"``, ``"PLL"``, or ``"SMO"``.
        """
        if is_salient:
            return "ActiveFlux"
        # Threshold: 1 500 rad/s ≈ 3 000 RPM with pp=5, or 7 500 RPM with pp=2
        if omega_e_max > 1500.0:
            return "PLL"
        return "SMO"

    def _apply_observer_calibration_to_gui(self) -> None:
        """Compute analytical observer gains from current motor params and apply to GUI.

        Called automatically after the two-stage auto-calibration pipeline
        finishes successfully.  Produces a physically consistent set of gains
        for the loaded motor, including full saliency awareness for IPM motors.

        Calibrated observers
        --------------------
        * PLL       — Kp, Ki from type-2 bandwidth target (ωn = ωe_max / 5)
        * SMO       — Kslide, LPF alpha, boundary from τe; for IPM (Lq/Ld>1.2):
                      lower k_slide (2×ωe_max) and wider boundary (0.08 rad) to
                      reduce chattering from saliency harmonics
        * STSMO     — k1, k2_min, k2_factor from Levant convergence conditions;
                      k2_min is motor-aware (scales with Ke×ωe_max); convergence
                      factor raised to 4.0 for salient motors
        * ActiveFlux — dc_cutoff derived from minimum electrical frequency at
                      the open-loop handoff speed
        * EEMF model — auto-enabled when Lq/Ld > 1.2 (IPM compensation)
        * SOGI filter — auto-enabled when Lq/Ld > 1.2 (zero phase-lag EMF filter)

        Startup sequence
        ----------------
        Alignment duration, align current, open-loop ramp speed/time, iq reference,
        and observer handoff thresholds are all derived analytically from motor
        parameters so the first simulation run uses safe default values without
        any manual entry.
        """
        try:
            import numpy as np

            from src.control.foc_controller import FOCController
            from src.core.motor_model import BLDCMotor, MotorParameters

            # ── Collect current motor parameters from GUI ──────────────────
            mp = self._collect_current_motor_parameters()
            rated_rpm = float(
                getattr(self, "foc_stsmo_rated_rpm", None)
                and self.foc_stsmo_rated_rpm.value()
                or 3000.0
            )

            # ── Derive key motor quantities ────────────────────────────────
            pp = max(1, int(mp["num_poles"]) // 2)
            ke = float(mp["back_emf_constant"])
            R  = float(mp["phase_resistance"])
            v_bus = float(mp["nominal_voltage"])
            ld = float(mp.get("ld") or mp["phase_inductance"])
            lq = float(mp.get("lq") or mp["phase_inductance"])
            # Average inductance for electrical time constant
            L_avg = (ld + lq) / 2.0
            tau_e = L_avg / max(R, 1e-9)          # electrical time constant [s]

            omega_m_max = rated_rpm / 60.0 * 2.0 * float(np.pi)
            omega_e_max = omega_m_max * pp         # max electrical angular speed [rad/s]

            # ── Saliency detection ─────────────────────────────────────────
            # IPM: Lq/Ld > 1.2 — requires EEMF model + SOGI to avoid angle bias
            saliency_ratio = lq / max(ld, 1e-12)
            is_salient = saliency_ratio > 1.2
            # Convergence factor: raise to 4.0 for salient motors (extra margin
            # against saliency-induced harmonic in the reconstructed EMF)
            convergence_factor = 4.0 if is_salient else 3.0

            # ── Observer auto-selection ────────────────────────────────────
            # If the user left the observer mode as "Auto (recommend from motor)"
            # we compute the best observer for this motor and programme it into
            # the combo.  If the user already picked a specific observer we
            # respect that choice and leave the combo unchanged.
            _user_obs_mode = self.foc_angle_observer_mode.currentText()
            if _user_obs_mode == "Auto (recommend from motor)":
                _recommended = self._recommend_observer(
                    is_salient=is_salient,
                    omega_e_max=omega_e_max,
                    v_bus=v_bus,
                    ke=ke,
                )
                self.foc_angle_observer_mode.setCurrentText(_recommended)
                # Startup observer: for sensorless modes use STSMO during
                # startup (unconditionally stable); for Measured keep Measured.
                if _recommended != "Measured":
                    self.foc_startup_initial_observer.setCurrentText("STSMO")
                else:
                    self.foc_startup_initial_observer.setCurrentText("Measured")
                self.calib_log.append(
                    f"\n[Observer auto-selection: recommended '{_recommended}' "
                    f"(saliency_ratio={saliency_ratio:.2f}, "
                    f"ωe_max={omega_e_max:.0f} rad/s)]\n"
                )
            elif self.foc_startup_initial_observer.currentText() == "Auto (recommend from motor)":
                # Main observer was set manually but startup is still Auto —
                # derive startup observer from the manually chosen main observer.
                _main_obs = _user_obs_mode
                if _main_obs in ("PLL", "SMO", "STSMO", "ActiveFlux"):
                    self.foc_startup_initial_observer.setCurrentText("STSMO")
                else:
                    self.foc_startup_initial_observer.setCurrentText("Measured")

            # ── Build transient FOCController for analytical formulas ───────
            params = MotorParameters(
                nominal_voltage=v_bus,
                phase_resistance=R,
                phase_inductance=mp["phase_inductance"],
                back_emf_constant=ke,
                torque_constant=float(mp["torque_constant"]),
                rotor_inertia=float(mp["rotor_inertia"]),
                friction_coefficient=float(mp["friction_coefficient"]),
                num_poles=int(mp["num_poles"]),
                poles_pairs=pp,   # must be set explicitly; default is 4
                ld=ld,
                lq=lq,
                model_type=mp.get("model_type", "dq"),
            )
            motor = BLDCMotor(params)
            ctrl = FOCController(motor=motor)
            ctrl.enable_sensorless_emf_reconstruction()

            # ── PLL ────────────────────────────────────────────────────────
            pll = ctrl.calibrate_pll_gains_analytical(rated_rpm=rated_rpm, apply=False)

            # ── SMO (saliency-aware) ────────────────────────────────────────
            dt_ui = 1.0 / max(float(self.inverter_switching_frequency.value()), 1.0)
            if is_salient:
                # IPM motors: reduce k_slide to 2×ωe_max (EEMF model does the
                # heavy lifting) and widen boundary to 0.08 rad to suppress
                # the reluctance-torque harmonic that would otherwise cause
                # excessive chattering with the standard 5×ωe_max gain.
                k_slide_smo = 2.0 * omega_e_max
                lpf_alpha_smo = float(dt_ui) / (float(dt_ui) + 3.0 * tau_e)
                lpf_alpha_smo = float(np.clip(lpf_alpha_smo, 1e-4, 0.5))
                smo = {
                    "k_slide":   k_slide_smo,
                    "lpf_alpha": lpf_alpha_smo,
                    "boundary":  0.08,
                }
            else:
                smo = ctrl.calibrate_smo_gains_analytical(
                    rated_rpm=rated_rpm, dt=dt_ui, apply=False
                )

            # ── STSMO (motor-aware k2_min) ─────────────────────────────────
            # k2_min governs STSMO tracking during the open-loop ramp when the
            # speed-adaptive formula (k2 = ke·ωm·ωe) approaches zero.
            # Formula: cover the EMF rate-of-change produced by accelerating from
            # 0 to 5 % of ωe_max in 0.1 s (typical ramp), × 3 safety factor.
            k2_min_motor = max(50.0, 1.5 * ke * omega_e_max)
            if is_salient:
                # Saliency adds a 2× harmonic in the reconstructed EMF whose
                # amplitude is ≈ (Lq-Ld)/Ld × Ke×ωe.  Extra margin needed.
                k2_min_motor *= 1.5
            k2_min_motor = float(np.clip(k2_min_motor, 50.0, 20000.0))

            stsmo = ctrl.calibrate_stsmo_gains_analytical(
                rated_rpm=rated_rpm,
                convergence_factor=convergence_factor,
                apply=False,
            )

            # ── ActiveFlux dc_cutoff ────────────────────────────────────────
            # Must be comfortably below the minimum electrical frequency.
            # We use the open-loop handoff speed (computed below) as reference.
            # f_e_min = handoff_rpm × pp / 60; dc_cutoff = f_e_min / 10
            # (clamped to [0.05, 0.5] Hz for safety).
            omega_m_emf_thresh = 0.05 * v_bus / max(ke, 1e-9)   # [rad/s mech]
            handoff_rpm = max(60.0, omega_m_emf_thresh * 60.0 / (2.0 * float(np.pi)))
            handoff_rpm = round(handoff_rpm / 10.0) * 10.0       # snap to 10 RPM
            f_e_min = handoff_rpm * pp / 60.0                     # [Hz electrical]
            dc_cutoff = float(np.clip(f_e_min / 10.0, 0.05, 0.5))

            # ── Startup sequence auto-tuning ────────────────────────────────
            # Alignment: settle in 5× electrical time constants (min 50 ms)
            align_duration = float(np.clip(5.0 * tau_e, 0.05, 0.5))
            # Alignment current: 10 % of bus voltage in amperes, capped [0.5, 3 A]
            align_current = float(np.clip(0.10 * v_bus, 0.5, 3.0))
            # Open-loop speed ramp: from 10 % to handoff speed
            initial_rpm = float(np.clip(0.10 * handoff_rpm, 5.0, 30.0))
            ramp_time = float(np.clip(handoff_rpm / max(rated_rpm, 1.0) * 0.5,
                                      0.10, 2.0))
            # Startup iq reference: enough to accelerate, capped [0.5, 3 A]
            iq_startup = float(np.clip(0.05 * v_bus, 0.5, 3.0))
            # Observer handoff thresholds
            min_emf_v    = float(np.clip(0.03 * v_bus, 0.05, 2.0))
            min_speed_rpm = handoff_rpm
            min_time_s   = float(np.clip(ramp_time * 0.8, 0.05, 1.0))

            # ── Apply all values to GUI widgets ────────────────────────────
            # EEMF / SOGI toggles (auto-enable for IPM motors)
            if hasattr(self, "foc_smo_eemf_model"):
                self.foc_smo_eemf_model.setCurrentText(
                    "Enabled" if is_salient else "Disabled"
                )
            if hasattr(self, "foc_smo_sogi_filter"):
                self.foc_smo_sogi_filter.setCurrentText(
                    "Enabled" if is_salient else "Disabled"
                )

            # D/Q decoupling — always beneficial for FOC; especially critical
            # for IPM motors where Lq >> Ld produces large cross-coupling terms.
            self.foc_decouple_d_mode.setCurrentText("Enabled")
            self.foc_decouple_q_mode.setCurrentText("Enabled")

            # ── PFC (Power Factor Controller) ──────────────────────────────
            # max_compensation_var: upper VAR clamp — scale with rated power.
            #   P_equiv ≈ Vbus² / (2R) gives a power proxy that works across
            #   motors from 12 V / 50 W to 48 V / 5 kW without knowing rated
            #   current explicitly.  30 % of that value is the reactive budget.
            p_equiv = v_bus ** 2 / max(2.0 * R, 1e-9)
            pfc_max_var = float(np.clip(p_equiv * 0.30, 100.0, 50000.0))

            # window_samples: must cover ≥2 electrical cycles at rated speed so
            #   the PF estimator averages out switching harmonics.
            #   f_e_rated = rated_rpm × pp / 60  [Hz]
            f_e_rated = rated_rpm * pp / 60.0
            dt_sim = 1.0 / max(float(self.inverter_switching_frequency.value()), 1.0)
            samples_per_cycle = 1.0 / max(f_e_rated * dt_sim, 1e-9)
            pfc_window = int(np.clip(round(2.0 * samples_per_cycle), 8, 2000))

            self.pfc_max_var.setValue(pfc_max_var)
            self.pfc_window_samples.setValue(pfc_window)
            # Enable PFC for diagnostic visibility; keep gains at defaults
            # (kp=0.10, ki=1.0 are dimensionless and motor-independent)
            self.pfc_mode.setCurrentText("Enabled")

            # PLL
            self.foc_pll_kp.setValue(float(pll["kp"]))
            self.foc_pll_ki.setValue(float(pll["ki"]))

            # SMO
            self.foc_smo_k_slide.setValue(float(smo["k_slide"]))
            self.foc_smo_lpf_alpha.setValue(float(smo["lpf_alpha"]))
            self.foc_smo_boundary.setValue(float(smo["boundary"]))

            # STSMO
            self.foc_stsmo_k1.setValue(float(stsmo["k1"]))
            self.foc_stsmo_k2_min.setValue(k2_min_motor)
            self.foc_stsmo_k2_factor.setValue(1.0)
            self.foc_stsmo_rated_rpm.setValue(float(rated_rpm))

            # ActiveFlux
            self.foc_af_dc_cutoff.setValue(dc_cutoff)

            # Startup sequence
            self.foc_startup_sequence_mode.setCurrentText("Enabled")
            self.foc_startup_transition_mode.setCurrentText("Enabled")
            self.foc_align_time.setValue(align_duration)
            self.foc_align_current.setValue(align_current)
            self.foc_open_loop_initial_speed.setValue(initial_rpm)
            self.foc_open_loop_target_speed.setValue(handoff_rpm)
            self.foc_open_loop_ramp_time.setValue(ramp_time)
            self.foc_open_loop_id_ref.setValue(0.0)
            self.foc_open_loop_iq_ref.setValue(iq_startup)
            self.foc_startup_min_speed.setValue(min_speed_rpm)
            self.foc_startup_min_emf.setValue(min_emf_v)
            self.foc_startup_min_time.setValue(min_time_s)

            # ── Log calibrated values ──────────────────────────────────────
            salient_tag = (
                f"  ⚑ IPM saliency detected: Lq/Ld={saliency_ratio:.2f} → "
                "EEMF model + SOGI filter enabled\n"
                if is_salient
                else f"  Isotropic motor (Lq/Ld={saliency_ratio:.2f})\n"
            )
            _active_obs = self.foc_angle_observer_mode.currentText()
            _startup_obs = self.foc_startup_initial_observer.currentText()
            self.calib_log.append(
                "\n[Observer gains auto-calibrated for current motor]\n"
                + salient_tag
                + f"  Active observer:  {_active_obs}\n"
                f"  Startup observer: {_startup_obs}\n"
                + f"  PLL:       Kp={pll['kp']:.2f}  Ki={pll['ki']:.1f}\n"
                f"  SMO:       Kslide={smo['k_slide']:.1f}  "
                f"LPF_alpha={smo['lpf_alpha']:.4f}  "
                f"Boundary={smo['boundary']:.3f} rad\n"
                f"  STSMO:     k1={stsmo['k1']:.3f}  k2_min={k2_min_motor:.1f} V/s"
                f"  k2_factor=1.0  (λ={convergence_factor:.1f})\n"
                f"  ActiveFlux: dc_cutoff={dc_cutoff:.3f} Hz\n"
                f"  Startup:   align={align_duration*1000:.0f} ms  "
                f"Ialign={align_current:.1f} A  "
                f"ramp {initial_rpm:.0f}→{handoff_rpm:.0f} RPM in {ramp_time:.2f} s  "
                f"min_EMF={min_emf_v:.2f} V\n"
                "  Decoupling: D=Enabled  Q=Enabled\n"
                f"  PFC:       Enabled  max_var={pfc_max_var:.0f} VAR"
                f"  window={pfc_window} samples\n"
            )

        except Exception as exc:  # noqa: BLE001
            self.calib_log.append(
                f"\n[Observer calibration to GUI failed: {exc}]\n"
                "  Observer gains were NOT updated — please use the per-observer "
                "Auto-Calibrate buttons manually.\n"
            )

    # ── Calibration result loaders ────────────────────────────────────────────

    def _match_calib_file(
        self, calib_dir: "Path", prefix: str, mp: dict, tol: float = 0.03
    ) -> "dict | None":
        """Find the best-matching calibration JSON in *calib_dir*.

        Looks for files whose name starts with *prefix* and whose stored motor
        parameters are within *tol* (relative) of the current GUI motor params.

        Parameters
        ----------
        calib_dir : Path
            Directory to search.
        prefix : str
            File-name prefix, e.g. ``"auto_calibrated_"`` or ``"fw_calibrated_"``.
        mp : dict
            Current motor params from ``_collect_current_motor_parameters()``.
        tol : float
            Maximum relative difference allowed for each key parameter (default 3 %).

        Returns
        -------
        Parsed JSON dict of the best-matching file, or ``None`` if no match.
        """
        import json as _json

        def _rel_close(a: float, b: float) -> bool:
            if max(abs(a), abs(b)) < 1e-12:
                return True
            return abs(a - b) / max(abs(a), abs(b)) <= tol

        best: dict | None = None
        best_score = -1

        for path in sorted(calib_dir.glob(f"{prefix}*.json")):
            try:
                with path.open() as f:
                    data = _json.load(f)
            except Exception:  # noqa: BLE001  # nosec B112
                continue

            # Stage-1 files use "motor_params" key; Stage-2 uses "motor_params_summary"
            stored = data.get("motor_params") or {}
            summary = data.get("motor_params_summary") or {}

            # Build a normalised comparison dict from whichever key is present
            cmp: dict = {}
            if stored:
                cmp = {
                    "R": stored.get("phase_resistance"),
                    "L": stored.get("phase_inductance"),
                    "Ke": stored.get("back_emf_constant"),
                    "V": stored.get("nominal_voltage"),
                    "pp": stored.get("poles_pairs"),
                }
            elif summary:
                cmp = {
                    "R": summary.get("R"),
                    "L": summary.get("L"),
                    "Ke": summary.get("Ke"),
                    "V": summary.get("Vnom"),
                    "pp": summary.get("pp"),
                }

            if not all(v is not None for v in cmp.values()):
                continue

            gui = {
                "R":  mp["phase_resistance"],
                "L":  mp["phase_inductance"],
                "Ke": mp["back_emf_constant"],
                "V":  mp["nominal_voltage"],
                "pp": max(1, int(mp["num_poles"]) // 2),
            }

            matches = sum(
                1 for k in gui if cmp[k] is not None and _rel_close(float(gui[k]), float(cmp[k]))
            )
            if matches == len(gui) and matches > best_score:
                best_score = matches
                best = data

        return best

    def _load_stage1_gains_to_gui(self) -> None:
        """Apply Stage-1 calibration results (current PI + speed PI) to the GUI.

        Searches ``data/tuning_sessions/`` for an ``auto_calibrated_*.json``
        file whose motor parameters match the currently loaded motor (within
        3 % relative tolerance on R, L, Ke, Vnom, pp).  If found, populates:

        * ``foc_d_kp``, ``foc_d_ki`` — d-axis current PI (same as q-axis: unified calibration)
        * ``foc_q_kp``, ``foc_q_ki`` — q-axis current PI
        * ``foc_speed_kp``, ``foc_speed_ki`` — outer speed loop PI
        """
        try:
            calib_dir = (
                Path(__file__).resolve().parents[2] / "data" / "tuning_sessions"
            )
            mp = self._collect_current_motor_parameters()
            data = self._match_calib_file(calib_dir, "auto_calibrated_", mp)

            if data is None:
                self.calib_log.append(
                    "\n[Stage 1 gains: no matching calibration file found — "
                    "current/speed PI gains unchanged]\n"
                )
                return

            res = data.get("tuning_result", {})
            current_kp = float(res["current_kp"])
            current_ki = float(res["current_ki"])
            speed_kp   = float(res["speed_kp"])
            speed_ki   = float(res["speed_ki"])

            # Stage 1 produces a single unified current_kp/ki; apply to both
            # d- and q-axis controllers.  The d-axis (flux) and q-axis (torque)
            # share the same bandwidth in the symmetric FOC design.
            self.foc_d_kp.setValue(current_kp)
            self.foc_d_ki.setValue(current_ki)
            self.foc_q_kp.setValue(current_kp)
            self.foc_q_ki.setValue(current_ki)
            self.foc_speed_kp.setValue(speed_kp)
            self.foc_speed_ki.setValue(speed_ki)

            profile_name = data.get("motor_profile_name", "unknown")
            self.calib_log.append(
                f"\n[Stage 1 gains loaded — profile: {profile_name}]\n"
                f"  Current PI: Kp={current_kp:.4f}  Ki={current_ki:.3f}  "
                f"(applied to d- and q-axes)\n"
                f"  Speed PI:   Kp={speed_kp:.4f}  Ki={speed_ki:.4f}\n"
            )

        except Exception as exc:  # noqa: BLE001
            self.calib_log.append(
                f"\n[Stage 1 gains load failed: {exc}]\n"
                "  Current / speed PI gains unchanged.\n"
            )

    def _load_stage2_fw_to_gui(self) -> None:
        """Apply Stage-2 calibration results (field-weakening parameters) to the GUI.

        Searches ``data/tuning_sessions/`` for a ``fw_calibrated_*.json`` file
        whose motor parameters match the currently loaded motor (within 3 %
        relative tolerance).  If found, enables field-weakening and sets:

        * ``foc_field_weakening_mode`` → "Enabled"
        * ``foc_field_weakening_start_speed`` → ``physics_params.fw_start_rpm``
        * ``foc_field_weakening_gain``        → ``selected_gain``
        * ``foc_field_weakening_max_id``      → ``selected_fw_id_max_a``
        * ``foc_field_weakening_headroom_target`` → ``physics_params.fw_headroom_target_v``
        * ``foc_iq_limit``                   → rated current from motor_params_summary
          (capped at the spinbox maximum for safety)

        If field-weakening is not needed (``fw_needed = False``), FW mode is set
        to "Disabled" and the remaining FW controls are still populated with the
        physics-based analytical values for reference.
        """
        try:
            calib_dir = (
                Path(__file__).resolve().parents[2] / "data" / "tuning_sessions"
            )
            mp = self._collect_current_motor_parameters()
            data = self._match_calib_file(calib_dir, "fw_calibrated_", mp)

            if data is None:
                self.calib_log.append(
                    "\n[Stage 2 FW: no matching calibration file found — "
                    "field-weakening parameters unchanged]\n"
                )
                return

            pp        = data.get("physics_params", {})
            fw_start  = float(pp.get("fw_start_rpm", 0.0))
            fw_gain   = float(data.get("selected_gain", pp.get("fw_gain", 1.0)))
            fw_id_max = float(data.get("selected_fw_id_max_a", pp.get("fw_id_max_a", 5.0)))
            fw_head   = float(pp.get("fw_headroom_target_v", 1.0))
            fw_needed = bool(pp.get("fw_needed", fw_start > 0))

            # Rated current from motor_params_summary (used for Iq limit)
            summary    = data.get("motor_params_summary", {})
            i_rated    = float(summary.get("I_rated", 0.0))

            self.foc_field_weakening_mode.setCurrentText(
                "Enabled" if fw_needed else "Disabled"
            )
            self.foc_field_weakening_start_speed.setValue(fw_start)
            self.foc_field_weakening_gain.setValue(fw_gain)
            self.foc_field_weakening_max_id.setValue(abs(fw_id_max))
            self.foc_field_weakening_headroom_target.setValue(fw_head)

            # Set Iq limit from rated current (capped at spinbox max)
            if i_rated > 0:
                iq_limit_max = float(self.foc_iq_limit.spinner.maximum())
                self.foc_iq_limit.setValue(min(i_rated, iq_limit_max))

            profile_name = data.get("motor_profile_name", "unknown")
            all_passed   = data.get("all_passed", False)
            self.calib_log.append(
                f"\n[Stage 2 FW parameters loaded — profile: {profile_name}]\n"
                f"  FW mode:    {'Enabled' if fw_needed else 'Disabled (no FW needed)'}\n"
                f"  Start RPM:  {fw_start:.1f}\n"
                f"  Gain:       {fw_gain:.3f}\n"
                f"  Max -Id:    {fw_id_max:.1f} A\n"
                f"  Headroom:   {fw_head:.3f} V\n"
                f"  All operating-point checks: {'PASS' if all_passed else 'PARTIAL'}\n"
                + (f"  Iq limit:   "
                   f"{min(i_rated, float(self.foc_iq_limit.spinner.maximum())):.1f} A\n"
                   if i_rated > 0 else "")
            )

        except Exception as exc:  # noqa: BLE001
            self.calib_log.append(
                f"\n[Stage 2 FW load failed: {exc}]\n"
                "  Field-weakening parameters unchanged.\n"
            )

    def _stop_auto_calibrate_all(self) -> None:
        """Terminate the running auto-calibration pipeline gracefully."""
        if (
            self.auto_calib_process is not None
            and self.auto_calib_process.state() != QProcess.ProcessState.NotRunning
        ):
            self._terminate_process(self.auto_calib_process)
        self.calib_log.append("\n[Auto-calibration stopped by user]\n")
        self.calib_result_status.setText("Status: Stopped")
        speak("Auto-calibration stopped.")
        self._reset_auto_calib_ui()

    def _terminate_process(self, process: QProcess) -> None:
        """Terminate a QProcess gracefully, then force-kill if needed."""
        process.terminate()
        if not process.waitForFinished(2000):
            process.kill()
            process.waitForFinished(1000)

    def _reset_auto_calib_ui(self) -> None:
        """Restore auto-calibration button states after completion or stop."""
        self._auto_calib_stage = 0
        self.auto_calib_process = None
        self.btn_auto_calib_all.setEnabled(True)
        self.btn_stop_auto_calib.setEnabled(False)
        self.status_bar_state.setText("State: Stopped")
        self.status_bar_task.setText("Task: None")
        self.status_bar_time_remaining.setText("Remaining: -- s")
        self._mark_task_finished("auto_calibration")

    def _cleanup_background_resources(self) -> None:
        """Stop running workers, processes, timers, and child windows before close."""
        if self.is_running:
            if self.sim_thread is not None:
                self.sim_thread.stop_simulation()
                self.sim_thread.wait(5000)
            self.is_running = False

        if self.auto_calib_process is not None:
            if self.auto_calib_process.state() != QProcess.ProcessState.NotRunning:
                self._terminate_process(self.auto_calib_process)
            self.auto_calib_process = None

        if self.calib_process is not None:
            if self.calib_process.state() != QProcess.ProcessState.NotRunning:
                self._terminate_process(self.calib_process)
            self.calib_process = None

        self.update_timer.stop()
        if hasattr(self, "plot_timer"):
            self.plot_timer.stop()
        if self.current_fft_window is not None:
            self.current_fft_window.close()
            self.current_fft_window = None

    def closeEvent(self, event) -> None:
        """
        Override closeEvent to ensure all background processes are properly terminated.
        This prevents orphaned processes when the GUI is closed.
        """
        try:
            self._cleanup_background_resources()

        except Exception as exc:
            # Log the error but don't block the window from closing
            print(f"Error during cleanup in closeEvent: {exc}")

        # Call parent closeEvent to complete the window closure
        event.accept()

    def _initialize_defaults(self):
        """Initialize simulation with default values."""
        self.ctrl_mode.setCurrentText("V/f")
        self.foc_group.setVisible(False)
        if hasattr(self, "pfc_mode"):
            self.pfc_mode.setCurrentText("Disabled")
        # initialize supply controls visibility
        if hasattr(self, "supply_type"):
            self.supply_type.setCurrentText("Constant")
            self._on_supply_type_changed("Constant")
        if hasattr(self, "hardware_enable_backend"):
            self.hardware_enable_backend.setChecked(False)
        self._apply_to_simulation()

    def _apply_to_simulation(self):  # noqa: C901
        """Apply current UI parameters to simulation."""
        requested_pwm_hz = float(self.inverter_switching_frequency.value())
        if requested_pwm_hz <= 0.0:
            requested_pwm_hz = _as_float(
                SIMULATION_PARAMS.get("pwm_frequency_hz", 20000.0), 20000.0
            )
        pwm_period_s = 1.0 / requested_pwm_hz

        # Create motor
        params = MotorParameters(
            nominal_voltage=self.param_voltage.value(),
            phase_resistance=self.param_resistance.value(),
            phase_inductance=self.param_inductance.value(),
            back_emf_constant=self.param_emf.value(),
            torque_constant=self.param_kt.value(),
            rotor_inertia=self.param_inertia.value(),
            friction_coefficient=self.param_friction.value(),
            num_poles=int(self.param_poles.value()),
            ld=self.param_ld.value(),
            lq=self.param_lq.value(),
            poles_pairs=int(self.param_poles.value() / 2),
            model_type=self.param_model_type.currentText(),
            emf_shape=self.param_emf_shape.currentText(),
        )

        self.motor = BLDCMotor(params, dt=pwm_period_s)
        # If Ld/Lq provided in UI, update params
        try:
            params.ld = float(self.param_ld.value())
            params.lq = float(self.param_lq.value())
        except Exception as exc:
            logger.warning("Unable to read Ld/Lq values from UI widgets: %s", exc)

        # Create load profile
        load_type = self.load_type.currentText()
        load: ConstantLoad | RampLoad
        if load_type == "Constant":
            load = ConstantLoad(torque=self.load_constant_torque.value())
        elif load_type == "Ramp":
            load = RampLoad(
                initial=self.load_initial_torque.value(),
                final=self.load_final_torque.value(),
                duration=self.load_ramp_duration.value(),
            )
        else:
            load = ConstantLoad(0.0)

        # Create supply profile based on UI selection
        from src.core.power_model import ConstantSupply, RampSupply, SupplyProfile

        supply: SupplyProfile
        supply_type = getattr(self, "supply_type", None)
        if supply_type and supply_type.currentText() == "Ramp":
            supply = RampSupply(
                initial=self.supply_ramp_initial.value(),
                final=self.supply_ramp_final.value(),
                duration=self.supply_ramp_duration.value(),
            )
        else:
            supply = ConstantSupply(voltage=self.supply_constant_voltage.value())

        hardware_interface = None
        if (
            hasattr(self, "hardware_enable_backend")
            and self.hardware_enable_backend.isChecked()
            and getattr(self, "hardware_backend_type", None)
            and self.hardware_backend_type.currentText() == "Mock DAQ"
        ):
            hardware_interface = MockDAQHardware(
                noise_std=self.hardware_noise_std.value(),
                seed=int(self.hardware_seed.value()),
            )

        current_sense = self._build_current_sense_model()

        # Create simulation engine
        self.engine = SimulationEngine(
            self.motor,
            load,
            dt=pwm_period_s,
            supply_profile=supply,
            hardware_interface=hardware_interface,
            current_sense=current_sense,
        )
        self.engine.set_pwm_frequency(requested_pwm_hz)
        self.engine.configure_hardware_interface(
            enabled=bool(
                hasattr(self, "hardware_enable_backend")
                and self.hardware_enable_backend.isChecked()
            )
        )
        self.engine.configure_power_factor_control(
            enabled=self.pfc_mode.currentText() == "Enabled",
            target_pf=self.pfc_target_pf.value(),
            kp=self.pfc_kp.value(),
            ki=self.pfc_ki.value(),
            max_compensation_var=self.pfc_max_var.value(),
            window_samples=int(self.pfc_window_samples.value()),
        )

        # Create SVM generator (cartesian capable if needed later)
        self.svm = SVMGenerator(dc_voltage=_as_float(SIMULATION_PARAMS["dc_voltage"], 48.0))
        self.svm.set_sample_time(self.engine.dt)
        self.svm.set_nonidealities(
            device_drop_v=self.inverter_device_drop.value(),
            dead_time_fraction=self.inverter_dead_time_fraction.value(),
            conduction_resistance_ohm=self.inverter_conduction_resistance.value(),
            switching_frequency_hz=requested_pwm_hz,
            switching_loss_coeff_v_per_a_khz=self.inverter_switching_loss_coeff.value(),
            enable_device_drop=self.inverter_enable_device_drop.isChecked(),
            enable_dead_time=self.inverter_enable_dead_time.isChecked(),
            enable_conduction_drop=self.inverter_enable_conduction.isChecked(),
            enable_switching_loss=self.inverter_enable_switching.isChecked(),
            enable_diode_freewheel=self.inverter_enable_diode.isChecked(),
            diode_drop_v=self.inverter_diode_drop.value(),
            diode_resistance_ohm=self.inverter_diode_resistance.value(),
            enable_min_pulse=self.inverter_enable_min_pulse.isChecked(),
            min_pulse_fraction=self.inverter_min_pulse_fraction.value(),
            enable_bus_ripple=self.inverter_enable_bus_ripple.isChecked(),
            dc_link_capacitance_f=self.inverter_dc_link_capacitance.value(),
            dc_link_source_resistance_ohm=self.inverter_dc_link_source_resistance.value(),
            dc_link_esr_ohm=self.inverter_dc_link_esr.value(),
            enable_thermal_coupling=self.inverter_enable_thermal.isChecked(),
            thermal_resistance_k_per_w=self.inverter_thermal_resistance.value(),
            thermal_capacitance_j_per_k=self.inverter_thermal_capacitance.value(),
            ambient_temperature_c=self.inverter_ambient_temp.value(),
            temp_coeff_resistance_per_c=self.inverter_temp_coeff_resistance.value(),
            temp_coeff_drop_per_c=self.inverter_temp_coeff_drop.value(),
            enable_phase_asymmetry=self.inverter_enable_phase_asymmetry.isChecked(),
            phase_voltage_scale_a=self.inverter_phase_voltage_scale_a.value(),
            phase_voltage_scale_b=self.inverter_phase_voltage_scale_b.value(),
            phase_voltage_scale_c=self.inverter_phase_voltage_scale_c.value(),
            phase_drop_scale_a=self.inverter_phase_drop_scale_a.value(),
            phase_drop_scale_b=self.inverter_phase_drop_scale_b.value(),
            phase_drop_scale_c=self.inverter_phase_drop_scale_c.value(),
        )

        # Determine controller type
        controller: BaseController
        if self.ctrl_mode.currentText() == "V/f":
            controller = VFController(
                v_nominal=self.vf_v_nominal.value(),
                f_nominal=self.vf_f_nominal.value(),
                dc_voltage=_as_float(SIMULATION_PARAMS["dc_voltage"], 48.0),
                v_startup=self.vf_startup_voltage.value(),
                ramp_rate=10.0,
            )
            controller.set_frequency_slew_rate(self.vf_freq_slew.value())
            controller.set_speed_reference(self.vf_speed_ref.value())
            controller.set_startup_sequence(
                enable=self.vf_startup_sequence_mode.currentText() == "Enabled",
                align_duration_s=self.vf_align_time.value(),
                align_voltage_v=self.vf_align_voltage.value(),
                align_angle_deg=self.vf_align_angle.value(),
                ramp_initial_frequency_hz=self.vf_ramp_initial_frequency.value(),
            )
        else:
            # FOC controller configuration
            use_conc = self.foc_transform.currentText() == "Concordia"
            out_cart = self.foc_output_mode.currentText() == "Cartesian"
            speed_loop_enabled = self.foc_speed_loop_mode.currentText() == "Cascaded PI"
            controller = FOCController(
                motor=self.motor,
                use_concordia=use_conc,
                output_cartesian=out_cart,
                enable_speed_loop=speed_loop_enabled,
            )
            controller.set_current_feedback_mode(
                self.foc_current_feedback_source.currentText() == "Reconstructed (Shunt)"
            )
            # set references
            controller.set_current_references(
                id_ref=self.foc_id_ref.value(),
                iq_ref=self.foc_iq_ref.value(),
            )
            controller.set_speed_reference(self.foc_speed_ref.value())
            controller.set_field_weakening(
                enabled=self.foc_field_weakening_mode.currentText() == "Enabled",
                start_speed_rpm=self.foc_field_weakening_start_speed.value(),
                gain=self.foc_field_weakening_gain.value(),
                max_negative_id_a=self.foc_field_weakening_max_id.value(),
                headroom_target_v=self.foc_field_weakening_headroom_target.value(),
            )
            controller.set_cascaded_speed_loop(
                enabled=speed_loop_enabled,
                iq_limit_a=self.foc_iq_limit.value(),
            )
            controller.set_speed_pi_gains(
                kp=self.foc_speed_kp.value(),
                ki=self.foc_speed_ki.value(),
            )
            controller.set_current_pi_gains(
                d_kp=self.foc_d_kp.value(),
                d_ki=self.foc_d_ki.value(),
                q_kp=self.foc_q_kp.value(),
                q_ki=self.foc_q_ki.value(),
            )
            controller.set_decoupling(
                enable_d=self.foc_decouple_d_mode.currentText() == "Enabled",
                enable_q=self.foc_decouple_q_mode.currentText() == "Enabled",
            )
            _obs_mode = self.foc_angle_observer_mode.currentText()
            # ── Guard: resolve unresolved "Auto" before entering simulation ───
            # If the user selected "Auto (recommend from motor)" but did NOT run
            # Auto Calibrate first, the combo is never resolved to a concrete
            # observer.  Fall back to "Measured" (safe sensored mode) and warn.
            if _obs_mode == "Auto (recommend from motor)":
                logger.warning(
                    "Observer mode 'Auto' was not resolved by Auto-Calibrate. "
                    "Falling back to 'Measured' (sensored) for this simulation run. "
                    "Click 'Auto Calibrate' first to let the pipeline choose the "
                    "optimal observer for the loaded motor profile."
                )
                _obs_mode = "Measured"
                # Do NOT write back to the combo — user may still want to run
                # Auto Calibrate later, so keep "Auto" visible in the dropdown.
            # ── Observer-specific activation ─────────────────────────────────
            if _obs_mode in ("STSMO", "ActiveFlux", "PLL", "SMO"):
                # All sensorless modes need EMF reconstruction
                controller.enable_sensorless_emf_reconstruction()
                # EEMF model — saliency compensation for IPM motors (Chen 2003)
                if (
                    hasattr(self, "foc_smo_eemf_model")
                    and self.foc_smo_eemf_model.currentText() == "Enabled"
                ):
                    lq_val = float(self.param_lq.value())
                    controller.enable_eemf_model(Lq=lq_val)
                # SOGI bandpass filter — zero phase lag at ωe
                if (
                    hasattr(self, "foc_smo_sogi_filter")
                    and self.foc_smo_sogi_filter.currentText() == "Enabled"
                ):
                    controller.enable_sogi_filter(k=1.4142)
            if _obs_mode == "STSMO":
                # Apply backward-Euler STSMO; set_angle_observer("PLL") for the
                # outer angle/speed loop — STSMO provides the EMF feed.
                controller.calibrate_stsmo_gains_analytical(
                    rated_rpm=self.foc_stsmo_rated_rpm.value(),
                )
                # Override individual gains if user has manually tuned them
                controller.stsmo["k1"] = self.foc_stsmo_k1.value()
                controller.stsmo["k2_min"] = self.foc_stsmo_k2_min.value()
                controller.stsmo["k2_factor"] = self.foc_stsmo_k2_factor.value()
                controller.set_angle_observer("PLL")  # outer loop stays PLL
            elif _obs_mode == "ActiveFlux":
                controller.enable_active_flux_observer(
                    dc_cutoff_hz=self.foc_af_dc_cutoff.value(),
                )
            else:
                controller.set_angle_observer(_obs_mode)

            controller.set_pll_gains(
                kp=self.foc_pll_kp.value(),
                ki=self.foc_pll_ki.value(),
            )
            controller.set_smo_gains(
                k_slide=self.foc_smo_k_slide.value(),
                lpf_alpha=self.foc_smo_lpf_alpha.value(),
                boundary=self.foc_smo_boundary.value(),
            )
            controller.set_startup_sequence(
                enabled=self.foc_startup_sequence_mode.currentText() == "Enabled",
                align_duration_s=self.foc_align_time.value(),
                align_current_a=self.foc_align_current.value(),
                align_angle_deg=self.foc_align_angle.value(),
                open_loop_initial_speed_rpm=self.foc_open_loop_initial_speed.value(),
                open_loop_target_speed_rpm=self.foc_open_loop_target_speed.value(),
                open_loop_ramp_time_s=self.foc_open_loop_ramp_time.value(),
                open_loop_id_ref_a=self.foc_open_loop_id_ref.value(),
                open_loop_iq_ref_a=self.foc_open_loop_iq_ref.value(),
            )
            controller.set_startup_transition(
                enabled=self.foc_startup_transition_mode.currentText() == "Enabled",
                initial_mode=self.foc_startup_initial_observer.currentText(),
                min_speed_rpm=self.foc_startup_min_speed.value(),
                min_elapsed_s=self.foc_startup_min_time.value(),
                min_emf_v=self.foc_startup_min_emf.value(),
                min_confidence=self.foc_startup_min_confidence.value(),
                confidence_hold_s=self.foc_startup_confidence_hold.value(),
                confidence_hysteresis=self.foc_startup_confidence_hysteresis.value(),
                fallback_enabled=self.foc_startup_fallback_mode.currentText() == "Enabled",
                fallback_hold_s=self.foc_startup_fallback_hold.value(),
            )

        self.controller = controller

    def _is_any_task_running(self) -> bool:
        """Check if any long-running task (simulation or calibration) is currently active."""
        with self._task_lock:
            return self._running_task_name is not None

    def _get_running_task_name(self) -> str | None:
        """Return the name of the currently running task, or None."""
        with self._task_lock:
            return self._running_task_name

    def _mark_task_running(self, task_name: str) -> None:
        """Mark a task as running (e.g., 'simulation' or 'calibration')."""
        with self._task_lock:
            self._running_task_name = task_name

    def _mark_task_finished(self, task_name: str) -> None:
        """Mark a task as finished. Only clears if the given task is currently running."""
        with self._task_lock:
            if self._running_task_name == task_name:
                self._running_task_name = None

    def _terminate_process_gracefully(
        self,
        process,
        process_type: str = "subprocess",
        timeout_graceful_ms: int = 2000,
        timeout_kill_ms: int = 1000,
    ) -> bool:
        """
        Terminate a process (QThread or QProcess) gracefully with timeout protection.

        Args:
            process: QThread or QProcess to terminate
            process_type: String describing the process for logging ("thread" or "process")
            timeout_graceful_ms: Time (ms) to wait for graceful termination
            timeout_kill_ms: Time (ms) to wait for forceful kill

        Returns:
            True if terminated successfully, False if forceful kill was required
        """
        if process is None:
            return True

        # Check if it's a QThread or QProcess
        if hasattr(process, "stop_simulation"):  # SimulationThread
            process.stop_simulation()
            if process.wait(timeout_graceful_ms):
                return True
            print(
                f"Warning: {process_type} did not finish gracefully within {timeout_graceful_ms}ms"
            )
            return False

        if hasattr(process, "state"):  # QProcess
            if process.state() == QProcess.ProcessState.NotRunning:
                return True

            # Try graceful termination first
            process.terminate()
            if process.waitForFinished(timeout_graceful_ms):
                return True

            # If that fails, force kill
            print(f"Warning: {process_type} did not terminate gracefully, forcing kill")
            process.kill()
            return not process.waitForFinished(timeout_kill_ms)

        return True

    def _can_start_task(self, new_task_name: str) -> bool:
        """
        Check if a new task can start. If another task is running, show confirmation dialog.
        Returns True if OK to proceed, False if user cancels or task conflicts.
        """
        running = self._get_running_task_name()
        if running is None:
            return True

        # Prevent same task from running twice
        if running == new_task_name:
            speak(f"{new_task_name.capitalize()} is already running.")
            return False

        # Different task is running; ask user to confirm
        reply = QMessageBox.warning(
            self,
            "Another Process Running",
            f"{running.capitalize()} is currently running.\n\nStop it and start {new_task_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Stop the running task
            if running == "simulation" and self.is_running:
                self._stop_simulation()
            elif running == "calibration" and self.calib_process is not None:
                self._stop_calibration()
            elif running == "auto_calibration" and self.auto_calib_process is not None:
                self._stop_auto_calibrate_all()
            return True

        # User canceled
        speak(f"Cancelled; {running} is still running.")
        return False

    def _start_simulation(self):
        """Start simulation."""
        if self.is_running:
            QMessageBox.warning(self, "Warning", "Simulation already running!")
            return

        self._apply_to_simulation()

        self.is_running = True
        self._mark_task_running("simulation")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Update status bar
        duration = self.sim_duration.value()
        msg = f"Simulation started. Duration: {'∞ (infinite)' if duration == 0 else f'{duration}s'}"
        self.status_bar_state.setText("State: Running")
        self.status_bar_task.setText("Task: Simulation")
        self.status_bar_time_remaining.setText(
            f"Remaining: {duration:.1f}s" if duration > 0 else "Elapsed: 0.0s"
        )

        # Reset speed curve data
        self.speed_history_time = []
        self.speed_history_rpm = []

        # Notify oscilloscope: ghost previous run's traces, clear live buffers
        if self.oscilloscope is not None:
            self.oscilloscope.start_new_run()

        # Create and start simulation thread
        assert self.engine is not None
        assert self.svm is not None
        assert self.controller is not None
        self.sim_thread = SimulationThread()
        self.sim_thread.set_simulation(
            self.engine,
            self.svm,
            self.controller,
            max_duration=duration,
            pwm_frequency_hz=float(1.0 / self.engine.dt),
        )
        self.sim_thread.finished_signal.connect(self._on_simulation_finished)
        self.sim_thread.start_simulation()
        self.update_timer.start(int(self.sim_thread.update_interval * 1000.0))

        speak(msg)

    def _stop_simulation(self):
        """Stop simulation."""
        if not self.is_running:
            return
        speak("Simulation stopped.")

        self.is_running = False
        self._mark_task_finished("simulation")

        # Update status bar
        self.status_bar_state.setText("State: Stopped")
        self.status_bar_task.setText("Task: None")
        self.status_bar_time_remaining.setText("Remaining: -- s")
        self.status_bar_cpu_load.setText("CPU: -- %")

        if self.sim_thread:
            self.sim_thread.stop_simulation()
            # Wait with timeout to prevent indefinite hang
            if not self.sim_thread.wait(5000):
                # Timeout occurred - log and continue cleanup
                print("Warning: Simulation thread did not finish within 5s timeout")
        self.update_timer.stop()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        QMessageBox.information(self, "Info", "Simulation stopped.")

    def _reset_simulation(self):
        """Reset simulation."""
        if self._stop_simulation() is not None:
            return

        if self.engine:
            self.engine.reset()
            self._update_display()
            QMessageBox.information(self, "Info", "Simulation reset to initial state.")

    def _on_control_mode_changed(self, text: str) -> None:
        """Show/hide parameter groups depending on selected control mode."""
        if text == "V/f":
            self.vf_group.setVisible(True)
            self.foc_group.setVisible(False)
        else:
            self.vf_group.setVisible(False)
            self.foc_group.setVisible(True)

    def _on_stsmo_autocalib(self) -> None:
        """Analytically compute STSMO gains and populate the GUI spinboxes."""
        if self.motor is None:
            QMessageBox.warning(self, "Warning", "Motor parameters are not initialised.")
            return
        try:
            from src.control import FOCController as _FOC

            _tmp = _FOC(motor=self.motor)
            result = _tmp.calibrate_stsmo_gains_analytical(
                rated_rpm=self.foc_stsmo_rated_rpm.value(),
                apply=False,
            )
            self.foc_stsmo_k1.setValue(round(result["k1"], 2))
            # k2_min defaults to 500.0 per Levant — keep user value unless it's the factory
            # default (500.0) so we don't overwrite a manual entry.
            QMessageBox.information(
                self,
                "STSMO Gains Calibrated",
                (
                    f"k1 = {result['k1']:.3f}\n"
                    f"k2 (rated-speed ref) = {result['k2']:.1f} V/s\n"
                    f"E_max = {result['e_max_v']:.2f} V\n\n"
                    "k2_min and k2_factor remain unchanged.\n"
                    "The effective k2 at runtime is speed-adaptive:\n"
                    "  k2_eff = max(k2_min, k2_factor × ke × ωm × ωe)"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Calibration Error", str(exc))

    def _auto_tune_axis(self, axis: str) -> None:
        """Trigger FOC controller auto-tuning for a specific axis."""
        if not isinstance(self.controller, FOCController):
            QMessageBox.warning(self, "Warning", "FOC controller is not active.")
            return
        self.controller.auto_tune_pi(axis=axis)
        QMessageBox.information(self, "Info", f"Auto-tuned {axis}-axis PI parameters.")

    def _update_monitoring(self, state: dict):  # noqa: C901  # TODO: split into sub-updaters per panel (22)
        """Update monitoring display with real-time values and speed curve."""
        speed_val = state.get("speed_rpm", 0)
        time_val = state.get("time", 0)

        # Pull advanced controller telemetry when FOC is active.
        ctrl_state = {}
        pfc_state = state.get("pfc", {}) if isinstance(state.get("pfc", {}), dict) else {}
        efficiency_state = (
            state.get("efficiency_metrics", {})
            if isinstance(state.get("efficiency_metrics", {}), dict)
            else {}
        )
        inverter_state = (
            state.get("inverter", {}) if isinstance(state.get("inverter", {}), dict) else {}
        )
        control_timing_state = (
            state.get("control_timing", {})
            if isinstance(state.get("control_timing", {}), dict)
            else {}
        )
        hardware_state = (
            state.get("hardware", {}) if isinstance(state.get("hardware", {}), dict) else {}
        )
        if not pfc_state and self.engine is not None:
            pfc_state = self.engine.get_power_factor_control_state()
        if not efficiency_state and self.engine is not None:
            efficiency_state = self.engine.get_efficiency_state()
        if not inverter_state and self.engine is not None:
            inverter_state = self.engine.get_inverter_state()
        if not control_timing_state and self.engine is not None:
            control_timing_state = self.engine.get_control_timing_state()

        calc_duration_s = float(control_timing_state.get("calc_duration_s", 0.0))
        control_period_s = max(
            float(
                control_timing_state.get(
                    "control_period_s", self.engine.dt if self.engine else 1e-4
                )
            ),
            1e-12,
        )
        slowdown = max(float(self.mcu_perf_ratio.value()), 1e-9)
        ref_clock = max(float(self.mcu_reference_clock_mhz.value()), 1e-9)
        target_1 = max(float(self.mcu_target_clock_1_mhz.value()), 1e-9)
        target_2 = max(float(self.mcu_target_clock_2_mhz.value()), 1e-9)
        target_3 = max(float(self.mcu_target_clock_3_mhz.value()), 1e-9)

        def _mcu_load_pct(target_clock_mhz: float) -> float:
            scaled_duration = calc_duration_s * slowdown * (ref_clock / target_clock_mhz)
            return 100.0 * scaled_duration / control_period_s

        mcu_load_1 = _mcu_load_pct(target_1)
        mcu_load_2 = _mcu_load_pct(target_2)
        mcu_load_3 = _mcu_load_pct(target_3)
        if not hardware_state and self.engine is not None:
            hardware_state = self.engine.get_hardware_state()
        observer_mode_code = 0.0
        startup_phase_code = 0.0
        if isinstance(self.controller, FOCController):
            ctrl_state = self.controller.get_state()
            observer_mode = str(ctrl_state.get("angle_observer_mode", "Measured"))
            startup_phase_code = float(ctrl_state.get("startup_phase_code", 0.0))
            mode_to_code = {
                "Measured": 0.0,
                "PLL": 1.0,
                "SMO": 2.0,
                "Alignment": 3.0,
                "OpenLoop": 4.0,
            }
            observer_mode_code = mode_to_code.get(observer_mode, -1.0)
        elif self.controller is not None:
            ctrl_state = self.controller.get_state()
            startup_phase_code = float(ctrl_state.get("startup_phase_code", 0.0))

        backend_name = str(hardware_state.get("backend", "none")).lower()
        backend_code_map = {"none": 0.0, "mock-daq": 1.0}
        hardware_backend_code = backend_code_map.get(backend_name, 2.0)
        hardware_io_error_flag = 1.0 if str(hardware_state.get("last_io_error", "")) else 0.0

        # Update accessible text blocks with new values
        if hasattr(self, "status_blocks"):
            self.status_blocks["speed_rpm"].update_value(speed_val)
            self.status_blocks["omega"].update_value(state.get("omega", 0))
            self.status_blocks["theta"].update_value(state.get("theta", 0))
            self.status_blocks["currents_a"].update_value(state.get("currents_a", 0))
            self.status_blocks["currents_b"].update_value(state.get("currents_b", 0))
            self.status_blocks["currents_c"].update_value(state.get("currents_c", 0))
            self.status_blocks["torque"].update_value(state.get("torque", 0))
            self.status_blocks["back_emf_a"].update_value(state.get("back_emf_a", 0))
            self.status_blocks["back_emf_b"].update_value(state.get("back_emf_b", 0))
            self.status_blocks["back_emf_c"].update_value(state.get("back_emf_c", 0))
            self.status_blocks["id_ref"].update_value(
                ctrl_state.get("id_ref_command", ctrl_state.get("id_ref", 0.0))
            )
            self.status_blocks["iq_ref"].update_value(
                ctrl_state.get("iq_ref_command", ctrl_state.get("iq_ref", 0.0))
            )
            self.status_blocks["speed_error"].update_value(ctrl_state.get("speed_error", 0.0))
            self.status_blocks["v_d_ff"].update_value(ctrl_state.get("v_d_ff", 0.0))
            self.status_blocks["v_q_ff"].update_value(ctrl_state.get("v_q_ff", 0.0))
            self.status_blocks["speed_loop_enabled"].update_value(
                1.0 if ctrl_state.get("speed_loop_enabled", False) else 0.0
            )
            self.status_blocks["decouple_d_enabled"].update_value(
                1.0 if ctrl_state.get("decouple_d_enabled", False) else 0.0
            )
            self.status_blocks["decouple_q_enabled"].update_value(
                1.0 if ctrl_state.get("decouple_q_enabled", False) else 0.0
            )
            self.status_blocks["observer_mode_code"].update_value(observer_mode_code)
            self.status_blocks["theta_electrical"].update_value(
                ctrl_state.get("theta_electrical", 0.0)
            )
            self.status_blocks["theta_meas_emf"].update_value(ctrl_state.get("theta_meas_emf", 0.0))
            self.status_blocks["theta_error_pll"].update_value(
                ctrl_state.get("theta_error_pll", 0.0)
            )
            self.status_blocks["theta_error_smo"].update_value(
                ctrl_state.get("theta_error_smo", 0.0)
            )
            self.status_blocks["smo_omega_est"].update_value(
                ctrl_state.get("smo", {}).get("omega_est", 0.0)
            )
            self.status_blocks["observer_confidence"].update_value(
                ctrl_state.get("observer_confidence", 0.0)
            )
            self.status_blocks["observer_confidence_emf"].update_value(
                ctrl_state.get("observer_confidence_emf", 0.0)
            )
            self.status_blocks["observer_confidence_speed"].update_value(
                ctrl_state.get("observer_confidence_speed", 0.0)
            )
            self.status_blocks["observer_confidence_coherence"].update_value(
                ctrl_state.get("observer_confidence_coherence", 0.0)
            )
            self.status_blocks["observer_confidence_ema"].update_value(
                ctrl_state.get("observer_confidence_ema", 0.0)
            )
            self.status_blocks["observer_confidence_trend"].update_value(
                ctrl_state.get("observer_confidence_trend", 0.0)
            )
            self.status_blocks["observer_confidence_above_threshold_time_s"].update_value(
                ctrl_state.get("observer_confidence_above_threshold_time_s", 0.0)
            )
            self.status_blocks["observer_confidence_below_threshold_time_s"].update_value(
                ctrl_state.get("observer_confidence_below_threshold_time_s", 0.0)
            )
            self.status_blocks["observer_confidence_crossings_up"].update_value(
                ctrl_state.get("observer_confidence_crossings_up", 0.0)
            )
            self.status_blocks["observer_confidence_crossings_down"].update_value(
                ctrl_state.get("observer_confidence_crossings_down", 0.0)
            )
            self.status_blocks["startup_sequence_enabled"].update_value(
                1.0 if ctrl_state.get("startup_sequence_enabled", False) else 0.0
            )
            self.status_blocks["startup_phase_code"].update_value(startup_phase_code)
            self.status_blocks["startup_sequence_elapsed_s"].update_value(
                ctrl_state.get("startup_sequence_elapsed_s", 0.0)
            )
            self.status_blocks["startup_phase_elapsed_s"].update_value(
                ctrl_state.get("startup_phase_elapsed_s", 0.0)
            )
            self.status_blocks["startup_handoff_count"].update_value(
                ctrl_state.get("startup_handoff_count", 0.0)
            )
            self.status_blocks["startup_last_handoff_time_s"].update_value(
                ctrl_state.get("startup_last_handoff_time_s", 0.0)
            )
            self.status_blocks["startup_last_handoff_confidence"].update_value(
                ctrl_state.get("startup_last_handoff_confidence", 0.0)
            )
            self.status_blocks["startup_handoff_confidence_peak"].update_value(
                ctrl_state.get("startup_handoff_confidence_peak", 0.0)
            )
            self.status_blocks["startup_handoff_quality"].update_value(
                ctrl_state.get("startup_handoff_quality", 0.0)
            )
            self.status_blocks["startup_handoff_stability_ratio"].update_value(
                ctrl_state.get("startup_handoff_stability_ratio", 1.0)
            )
            self.status_blocks["pfc_enabled"].update_value(
                1.0 if pfc_state.get("enabled", False) else 0.0
            )
            self.status_blocks["pfc_target_pf"].update_value(float(pfc_state.get("target_pf", 0.0)))
            self.status_blocks["pfc_power_factor"].update_value(
                float(pfc_state.get("power_factor", 0.0))
            )
            self.status_blocks["pfc_active_power_w"].update_value(
                float(pfc_state.get("active_power_w", 0.0))
            )
            self.status_blocks["pfc_reactive_power_var"].update_value(
                float(pfc_state.get("reactive_power_var", 0.0))
            )
            self.status_blocks["pfc_command_var"].update_value(
                float(pfc_state.get("compensation_command_var", 0.0))
            )
            self.status_blocks["efficiency"].update_value(
                float(efficiency_state.get("efficiency", 0.0))
            )
            self.status_blocks["mechanical_output_power_w"].update_value(
                float(efficiency_state.get("mechanical_output_power_w", 0.0))
            )
            self.status_blocks["total_loss_power_w"].update_value(
                float(efficiency_state.get("total_loss_power_w", 0.0))
            )
            self.status_blocks["effective_dc_voltage"].update_value(
                float(inverter_state.get("effective_dc_voltage", 0.0))
            )
            self.status_blocks["dc_link_ripple_v"].update_value(
                float(inverter_state.get("dc_link_ripple_v", 0.0))
            )
            self.status_blocks["dc_link_bus_current_a"].update_value(
                float(inverter_state.get("dc_link_bus_current_a", 0.0))
            )
            self.status_blocks["inverter_total_loss_power_w"].update_value(
                float(inverter_state.get("total_inverter_loss_power_w", 0.0))
            )
            self.status_blocks["junction_temperature_c"].update_value(
                float(inverter_state.get("junction_temperature_c", 0.0))
            )
            self.status_blocks["common_mode_voltage"].update_value(
                float(inverter_state.get("common_mode_voltage", 0.0))
            )
            self.status_blocks["control_calc_duration_us"].update_value(
                1e6 * float(control_timing_state.get("calc_duration_s", 0.0))
            )
            self.status_blocks["control_cpu_load_pct"].update_value(
                float(control_timing_state.get("cpu_load_pct", 0.0))
            )
            self.status_blocks["control_cpu_load_avg_pct"].update_value(
                float(control_timing_state.get("cpu_load_avg_pct", 0.0))
            )
            self.status_blocks["mcu_load_target_1_pct"].update_value(mcu_load_1)
            self.status_blocks["mcu_load_target_2_pct"].update_value(mcu_load_2)
            self.status_blocks["mcu_load_target_3_pct"].update_value(mcu_load_3)
            self.status_blocks["hardware_enabled"].update_value(
                1.0 if hardware_state.get("enabled", False) else 0.0
            )
            self.status_blocks["hardware_connected"].update_value(
                1.0 if hardware_state.get("connected", False) else 0.0
            )
            self.status_blocks["hardware_backend_code"].update_value(hardware_backend_code)
            self.status_blocks["hardware_write_count"].update_value(
                float(hardware_state.get("write_count", 0.0))
            )
            self.status_blocks["hardware_read_count"].update_value(
                float(hardware_state.get("read_count", 0.0))
            )
            self.status_blocks["hardware_io_error_flag"].update_value(hardware_io_error_flag)
            self.status_blocks["time"].update_value(time_val)
        else:
            # Fallback for backward compatibility with old label-based system
            self.status_labels["speed_rpm"].setText(f"⚡ Rotor Speed: {speed_val:.2f} RPM")
            self.status_labels["omega"].setText(
                f"Angular Velocity: {state.get('omega', 0):.4f} rad/s"
            )
            self.status_labels["theta"].setText(f"Rotor Position: {state.get('theta', 0):.4f} rad")
            self.status_labels["currents_a"].setText(
                f"Phase A Current: {state.get('currents_a', 0):.3f} A"
            )
            self.status_labels["currents_b"].setText(
                f"Phase B Current: {state.get('currents_b', 0):.3f} A"
            )
            self.status_labels["currents_c"].setText(
                f"Phase C Current: {state.get('currents_c', 0):.3f} A"
            )
            self.status_labels["torque"].setText(
                f"Electromagnetic Torque: {state.get('torque', 0):.4f} N·m"
            )
            self.status_labels["back_emf_a"].setText(
                f"Back-EMF Phase A: {state.get('back_emf_a', 0):.3f} V"
            )
            self.status_labels["back_emf_b"].setText(
                f"Back-EMF Phase B: {state.get('back_emf_b', 0):.3f} V"
            )
            self.status_labels["back_emf_c"].setText(
                f"Back-EMF Phase C: {state.get('back_emf_c', 0):.3f} V"
            )
            self.status_labels["id_ref"].setText(
                f"d-axis Current Ref: {ctrl_state.get('id_ref', 0.0):.3f} A"
            )
            self.status_labels["iq_ref"].setText(
                f"q-axis Current Ref: {ctrl_state.get('iq_ref', 0.0):.3f} A"
            )
            self.status_labels["speed_error"].setText(
                f"Speed Error: {ctrl_state.get('speed_error', 0.0):.4f} rad/s"
            )
            self.status_labels["v_d_ff"].setText(
                f"d-axis Feedforward: {ctrl_state.get('v_d_ff', 0.0):.4f} V"
            )
            self.status_labels["v_q_ff"].setText(
                f"q-axis Feedforward: {ctrl_state.get('v_q_ff', 0.0):.4f} V"
            )
            self.status_labels["speed_loop_enabled"].setText(
                f"Speed Loop Enabled: {1 if ctrl_state.get('speed_loop_enabled', False) else 0}"
            )
            self.status_labels["decouple_d_enabled"].setText(
                f"D-axis Decoupling: {1 if ctrl_state.get('decouple_d_enabled', False) else 0}"
            )
            self.status_labels["decouple_q_enabled"].setText(
                f"Q-axis Decoupling: {1 if ctrl_state.get('decouple_q_enabled', False) else 0}"
            )
            self.status_labels["observer_mode_code"].setText(
                f"Observer Mode Code: {observer_mode_code:.0f}"
            )
            self.status_labels["theta_electrical"].setText(
                f"Estimated Electrical Angle: {ctrl_state.get('theta_electrical', 0.0):.4f} rad"
            )
            self.status_labels["theta_meas_emf"].setText(
                f"Back-EMF Angle: {ctrl_state.get('theta_meas_emf', 0.0):.4f} rad"
            )
            self.status_labels["theta_error_pll"].setText(
                f"PLL Angle Error: {ctrl_state.get('theta_error_pll', 0.0):.4f} rad"
            )
            self.status_labels["theta_error_smo"].setText(
                f"SMO Angle Error: {ctrl_state.get('theta_error_smo', 0.0):.4f} rad"
            )
            self.status_labels["smo_omega_est"].setText(
                f"SMO Estimated Speed: {ctrl_state.get('smo', {}).get('omega_est', 0.0):.4f} rad/s"
            )
            self.status_labels["observer_confidence"].setText(
                f"Observer Confidence: {ctrl_state.get('observer_confidence', 0.0):.3f}"
            )
            self.status_labels["observer_confidence_emf"].setText(
                f"Confidence from EMF: {ctrl_state.get('observer_confidence_emf', 0.0):.3f}"
            )
            self.status_labels["observer_confidence_speed"].setText(
                f"Confidence from Speed: {ctrl_state.get('observer_confidence_speed', 0.0):.3f}"
            )
            self.status_labels["observer_confidence_coherence"].setText(
                f"Confidence from Coherence: {ctrl_state.get('observer_confidence_coherence', 0.0):.3f}"  # noqa: E501
            )
            self.status_labels["observer_confidence_ema"].setText(
                f"Confidence EMA: {ctrl_state.get('observer_confidence_ema', 0.0):.3f}"
            )
            self.status_labels["observer_confidence_trend"].setText(
                f"Confidence Trend: {ctrl_state.get('observer_confidence_trend', 0.0):.4f}"
            )
            self.status_labels["observer_confidence_above_threshold_time_s"].setText(
                f"Confidence Above Threshold Time: {ctrl_state.get('observer_confidence_above_threshold_time_s', 0.0):.3f} s"  # noqa: E501
            )
            self.status_labels["observer_confidence_below_threshold_time_s"].setText(
                f"Confidence Below Threshold Time: {ctrl_state.get('observer_confidence_below_threshold_time_s', 0.0):.3f} s"  # noqa: E501
            )
            self.status_labels["observer_confidence_crossings_up"].setText(
                f"Confidence Crossings Up: {int(ctrl_state.get('observer_confidence_crossings_up', 0))}"  # noqa: E501
            )
            self.status_labels["observer_confidence_crossings_down"].setText(
                f"Confidence Crossings Down: {int(ctrl_state.get('observer_confidence_crossings_down', 0))}"  # noqa: E501
            )
            self.status_labels["startup_handoff_count"].setText(
                f"Startup Handoff Count: {int(ctrl_state.get('startup_handoff_count', 0))}"
            )
            self.status_labels["startup_last_handoff_time_s"].setText(
                f"Last Handoff Time: {ctrl_state.get('startup_last_handoff_time_s', 0.0):.3f} s"
            )
            self.status_labels["startup_last_handoff_confidence"].setText(
                f"Last Handoff Confidence: {ctrl_state.get('startup_last_handoff_confidence', 0.0):.3f}"  # noqa: E501
            )
            self.status_labels["startup_handoff_confidence_peak"].setText(
                f"Handoff Confidence Peak: {ctrl_state.get('startup_handoff_confidence_peak', 0.0):.3f}"  # noqa: E501
            )
            self.status_labels["startup_handoff_quality"].setText(
                f"Handoff Quality KPI: {ctrl_state.get('startup_handoff_quality', 0.0):.3f}"
            )
            self.status_labels["startup_handoff_stability_ratio"].setText(
                f"Handoff Stability Ratio: {ctrl_state.get('startup_handoff_stability_ratio', 1.0):.3f}"  # noqa: E501
            )
            self.status_labels["efficiency"].setText(
                f"System Efficiency: {efficiency_state.get('efficiency', 0.0):.3f}"
            )
            self.status_labels["mechanical_output_power_w"].setText(
                f"Mechanical Output Power: {efficiency_state.get('mechanical_output_power_w', 0.0):.3f} W"  # noqa: E501
            )
            self.status_labels["total_loss_power_w"].setText(
                f"Estimated Total Loss: {efficiency_state.get('total_loss_power_w', 0.0):.3f} W"
            )
            self.status_labels["effective_dc_voltage"].setText(
                f"Effective DC-Link Voltage: {inverter_state.get('effective_dc_voltage', 0.0):.3f} V"  # noqa: E501
            )
            self.status_labels["dc_link_ripple_v"].setText(
                f"DC-Link Ripple: {inverter_state.get('dc_link_ripple_v', 0.0):.3f} V"
            )
            self.status_labels["dc_link_bus_current_a"].setText(
                f"DC-Link Bus Current: {inverter_state.get('dc_link_bus_current_a', 0.0):.3f} A"
            )
            self.status_labels["inverter_total_loss_power_w"].setText(
                f"Inverter Total Loss: {inverter_state.get('total_inverter_loss_power_w', 0.0):.3f} W"  # noqa: E501
            )
            self.status_labels["junction_temperature_c"].setText(
                f"Inverter Junction Temp: {inverter_state.get('junction_temperature_c', 0.0):.3f} C"
            )
            self.status_labels["common_mode_voltage"].setText(
                f"Common-Mode Voltage: {inverter_state.get('common_mode_voltage', 0.0):.3f} V"
            )
            self.status_labels["control_calc_duration_us"].setText(
                f"Control Calc Duration: {1e6 * float(control_timing_state.get('calc_duration_s', 0.0)):.3f} us"  # noqa: E501
            )
            self.status_labels["control_cpu_load_pct"].setText(
                f"Control CPU Load: {float(control_timing_state.get('cpu_load_pct', 0.0)):.3f} %"
            )
            self.status_labels["control_cpu_load_avg_pct"].setText(
                f"Control CPU Load Avg: {float(control_timing_state.get('cpu_load_avg_pct', 0.0)):.3f} %"  # noqa: E501
            )
            self.status_labels["mcu_load_target_1_pct"].setText(
                f"MCU Load @ Target 1: {mcu_load_1:.3f} %"
            )
            self.status_labels["mcu_load_target_2_pct"].setText(
                f"MCU Load @ Target 2: {mcu_load_2:.3f} %"
            )
            self.status_labels["mcu_load_target_3_pct"].setText(
                f"MCU Load @ Target 3: {mcu_load_3:.3f} %"
            )
            self.status_labels["hardware_enabled"].setText(
                f"Communication Enabled: {1 if hardware_state.get('enabled', False) else 0}"
            )
            self.status_labels["hardware_connected"].setText(
                f"Communication Connected: {1 if hardware_state.get('connected', False) else 0}"
            )
            self.status_labels["hardware_backend_code"].setText(
                f"Communication Backend Code: {hardware_backend_code:.0f}"
            )
            self.status_labels["hardware_write_count"].setText(
                f"Communication Write Count: {int(hardware_state.get('write_count', 0))}"
            )
            self.status_labels["hardware_read_count"].setText(
                f"Communication Read Count: {int(hardware_state.get('read_count', 0))}"
            )
            self.status_labels["hardware_io_error_flag"].setText(
                f"Communication I/O Error Flag: {int(hardware_io_error_flag)}"
            )
            self.status_labels["time"].setText(f"Simulation Time: {time_val:.3f} s")

            # Update accessible descriptions
            for lbl in self.status_labels.values():
                try:
                    full_text = lbl.text()
                    lbl.setToolTip(full_text)
                    QAccessible.updateAccessibility(lbl, 0, QAccessible.Event.ValueChange)
                except Exception as exc:
                    logger.debug("Accessibility update skipped for status label: %s", exc)

        # Update permanent info panel values
        try:
            if self.engine:
                dt_val = self.engine.dt
                motor_params = self.engine.motor.params
            else:
                dt_val = _as_float(SIMULATION_PARAMS.get("dt", 0.0001), 0.0001)
                from src.core.motor_model import MotorParameters

                motor_params = MotorParameters()

            self.lbl_dt.setText(f"dt: {dt_val} s")
            # electrical & mechanical time constants
            try:
                tau_e = motor_params.phase_inductance / motor_params.phase_resistance
            except Exception:
                tau_e = None
            try:
                tau_m = motor_params.rotor_inertia / motor_params.friction_coefficient
            except Exception:
                tau_m = None
            self.lbl_tau_e.setText(
                f"Electrical time constant (L/R): {tau_e if tau_e is not None else '--'} s"
            )
            self.lbl_tau_m.setText(
                f"Mechanical time constant (J/b): {tau_m if tau_m is not None else '--'} s"
            )

            # update Ld/Lq display
            ld_val = self.param_ld.value()
            lq_val = self.param_lq.value()
            self.lbl_ld_lq.setText(f"Ld: {ld_val:.6f} H, Lq: {lq_val:.6f} H")

            # Update parameter units summary
            units_html = (
                "<b>Parameters (units):</b><br>"
                + "Nominal Voltage (V), Phase Resistance (Ω), Phase Inductance (H),<br>"
                + "Back-EMF (V·s/rad), Torque Constant (N·m/A), Inertia (kg·m²), Friction (N·m·s/rad),<br>"  # noqa: E501
                + "Poles (count), Ld (H), Lq (H)"
            )
            self.lbl_param_units.setText(units_html)

            # Update status bar with simulation parameters
            self.status_bar_dt.setText(f"dt: {dt_val} s")
            self.status_bar_tau_e.setText(f"τ_e: {tau_e if tau_e is not None else '--'} s")
            self.status_bar_tau_m.setText(f"τ_m: {tau_m if tau_m is not None else '--'} s")
            if self.engine:
                stability = self.engine.get_numerical_stability_advisory()
            else:
                pwm_hz = 1.0 / dt_val if dt_val > 0.0 else 0.0
                stability = self._build_dt_pwm_stability_advisory(dt_val, motor_params, pwm_hz)

            severity = str(stability.get("severity", "unknown"))
            margin = float(stability.get("stability_margin", float("inf")))
            dt_recommended_s = float(stability.get("dt_recommended_s", float("inf")))
            pwm_recommended_hz = float(stability.get("pwm_recommended_min_hz", 0.0))

            self.lbl_stability.setText(
                "RK4 stability advisory: "
                + str(stability.get("message", "n/a"))
                + " Recommended action: adjust dt or PWM frequency only."
            )
            if np.isfinite(dt_recommended_s):
                self.status_bar_stability.setText(
                    f"RK4: {severity} | margin={margin:.2f}x | dt<= {dt_recommended_s:.2e}s | PWM>= {pwm_recommended_hz:.1f}Hz"  # noqa: E501
                )
            else:
                self.status_bar_stability.setText(f"RK4: {severity}")

            # Keep advisory content discoverable by tooltips and screen readers.
            self.lbl_stability.setToolTip(self.lbl_stability.text())
            self.status_bar_stability.setToolTip(self.status_bar_stability.text())
            self.lbl_stability.setAccessibleDescription(self.lbl_stability.text())
            self.status_bar_stability.setAccessibleName(f"RK4 stability status: {severity}")
            self.status_bar_stability.setAccessibleDescription(self.status_bar_stability.text())
            QAccessible.updateAccessibility(self.lbl_stability, 0, QAccessible.Event.ValueChange)
            QAccessible.updateAccessibility(
                self.status_bar_stability, 0, QAccessible.Event.ValueChange
            )

            if severity == "unstable":
                severity_color = "#C62828"
            elif severity == "marginal":
                severity_color = "#EF6C00"
            elif severity == "stable":
                severity_color = "#2E7D32"
            else:
                severity_color = "#455A64"
            self.lbl_stability.setStyleSheet(f"color: {severity_color};")
            self.status_bar_stability.setStyleSheet(f"color: {severity_color};")
            self._announce_stability_advisory_if_needed(
                severity=severity,
                dt_recommended_s=dt_recommended_s,
                pwm_recommended_min_hz=pwm_recommended_hz,
                margin=margin,
            )
        except Exception as exc:
            logger.debug("Permanent info panel update skipped: %s", exc)

        # Keep speed history for CSV export / backward compatibility
        self.speed_history_time.append(float(time_val))
        self.speed_history_rpm.append(float(speed_val))

        # Push live sample to the oscilloscope panel
        if self.oscilloscope is not None:
            try:
                _osc_values: dict[str, float] = {
                    "speed_rpm": float(speed_val),
                    "speed": float(state.get("omega", 0.0)),
                    "torque": float(state.get("torque", 0.0)),
                    "load_torque": float(state.get("load_torque", 0.0)),
                    "currents_a": float(state.get("currents_a", 0.0)),
                    "currents_b": float(state.get("currents_b", 0.0)),
                    "currents_c": float(state.get("currents_c", 0.0)),
                    "voltages_a": float(state.get("voltages_a", 0.0)),
                    "voltages_b": float(state.get("voltages_b", 0.0)),
                    "voltages_c": float(state.get("voltages_c", 0.0)),
                    "emf_a": float(state.get("emf_a", 0.0)),
                    "emf_b": float(state.get("emf_b", 0.0)),
                    "emf_c": float(state.get("emf_c", 0.0)),
                    "theta": float(state.get("theta", 0.0)),
                    "theta_electrical": float(state.get("theta_electrical", 0.0)),
                    "observer_confidence": float(state.get("observer_confidence", 0.0)),
                    "efficiency": float(efficiency_state.get("efficiency", 0.0)),
                    "input_power": float(efficiency_state.get("input_power", 0.0)),
                    "mechanical_output_power": float(
                        efficiency_state.get("mechanical_output_power_w", 0.0)
                    ),
                    "effective_dc_voltage": float(inverter_state.get("effective_dc_voltage", 0.0)),
                    "dc_link_ripple_v": float(inverter_state.get("dc_link_ripple_v", 0.0)),
                }
                self.oscilloscope.push_sample(float(time_val), _osc_values)
            except Exception as _osc_exc:
                logger.debug("Oscilloscope push_sample skipped: %s", _osc_exc)

    def _update_display(self):
        """Update display (for manual updates)."""
        if self.engine:
            state = self.engine.get_current_state()
            info = self.engine.get_simulation_info()
            self._update_monitoring({**state, **info})

    def _poll_simulation_state(self):
        """Poll latest simulation snapshot without coupling GUI to control timing."""
        if self.sim_thread is None:
            return
        self._update_runtime_current_sense_actuals()
        snapshot = self.sim_thread.get_latest_state()
        if snapshot:
            self._update_monitoring(snapshot)
            self._update_status_bar(snapshot)
            self._update_current_sense_status(snapshot)
            self._update_bridge_visualization(snapshot)
            if self.current_fft_window is not None:
                self.current_fft_window.set_window_size(
                    int(self.current_sense_fft_window_samples.value())
                )
                self._apply_fft_display_settings()
                self.current_fft_window.push_snapshot(snapshot)

    def _apply_fft_display_settings(self, *_args) -> None:
        """Apply FFT graph display preferences to the detached FFT window."""
        if self.current_fft_window is None:
            return
        self.current_fft_window.apply_display_settings(
            grid_enabled=self.current_sense_fft_show_grid.isChecked(),
            mag_x_scale=self.current_sense_fft_mag_x_scale.currentText(),
            mag_y_scale=self.current_sense_fft_mag_y_scale.currentText(),
            phase_x_scale=self.current_sense_fft_phase_x_scale.currentText(),
            phase_y_scale=self.current_sense_fft_phase_y_scale.currentText(),
            amplitude_mode=self.current_sense_fft_amplitude_mode.currentText(),
            phase_unit=self.current_sense_fft_phase_unit.currentText(),
        )

    def _save_current_fft_csv(self) -> None:
        """Save FFT data from the detached FFT window."""
        if self.current_fft_window is None:
            self._open_current_fft_window()
        if self.current_fft_window is not None:
            self.current_fft_window._save_fft_csv()

    def _save_current_fft_image(self) -> None:
        """Save FFT image from the detached FFT window."""
        if self.current_fft_window is None:
            self._open_current_fft_window()
        if self.current_fft_window is not None:
            self.current_fft_window._save_fft_image()

    def _update_bridge_visualization(self, snapshot: dict) -> None:
        """Draw a live inverter bridge view and highlight selected shunt topology."""
        if not hasattr(self, "bridge_ax"):
            return
        from matplotlib.patches import Circle, Rectangle

        topology = self.current_sense_topology.currentText()
        meas = snapshot.get("current_measurement", {}) if isinstance(snapshot, dict) else {}
        if isinstance(meas, dict) and meas.get("enabled", False):
            topology = str(meas.get("topology", topology))

        va = float(snapshot.get("voltages_a", 0.0))
        vb = float(snapshot.get("voltages_b", 0.0))
        vc = float(snapshot.get("voltages_c", 0.0))
        upper_on = [va >= 0.0, vb >= 0.0, vc >= 0.0]

        ax = self.bridge_ax
        ax.clear()
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_axis_off()

        labels = ["A", "B", "C"]
        shunt_map = {
            "single": [True, False, False],
            "double": [True, True, False],
            "triple": [True, True, True],
        }.get(topology, [True, True, True])

        # Draw a shared low-side source bus used by all three legs.
        low_side_bus_y = 0.10
        ax.plot([0.2, 2.8], [low_side_bus_y, low_side_bus_y], color="#455A64", linewidth=2.0)

        for idx in range(3):
            x = idx + 0.2
            color_hi = "#2E7D32" if upper_on[idx] else "#CFD8DC"
            color_lo = "#CFD8DC" if upper_on[idx] else "#2E7D32"
            ax.add_patch(Rectangle((x, 0.58), 0.6, 0.28, color=color_hi, ec="#263238"))
            ax.add_patch(Rectangle((x, 0.14), 0.6, 0.28, color=color_lo, ec="#263238"))
            ax.text(x + 0.3, 0.90, f"{labels[idx]}U", ha="center", va="center", fontsize=8)
            ax.text(x + 0.3, 0.49, f"{labels[idx]}L", ha="center", va="center", fontsize=8)

            # Connect each low-side switch source to the shared return bus.
            ax.plot(
                [x + 0.3, x + 0.3],
                [0.14, low_side_bus_y],
                color="#455A64",
                linewidth=1.4,
            )

            if topology != "single" and shunt_map[idx]:
                ax.add_patch(Circle((x + 0.3, 0.05), 0.05, color="#F57C00", ec="#E65100"))
                ax.text(x + 0.3, 0.01, "shunt", ha="center", va="top", fontsize=7)

        if topology == "single":
            # Single-shunt hardware: one common shunt in the low-side return path to GND.
            shunt_x0 = 1.45
            shunt_w = 0.10
            shunt_y0 = 0.02
            shunt_h = 0.06
            ax.plot(
                [1.5, 1.5],
                [low_side_bus_y, shunt_y0 + shunt_h],
                color="#455A64",
                linewidth=1.6,
            )
            ax.add_patch(
                Rectangle(
                    (shunt_x0, shunt_y0),
                    shunt_w,
                    shunt_h,
                    color="#F57C00",
                    ec="#E65100",
                )
            )
            ax.text(1.5, shunt_y0 - 0.005, "shunt", ha="center", va="top", fontsize=8)

            # Ground connection and symbol.
            gnd_y = 0.0
            ax.plot([1.5, 1.5], [shunt_y0, gnd_y + 0.015], color="#455A64", linewidth=1.6)
            ax.plot(
                [1.42, 1.58],
                [gnd_y + 0.015, gnd_y + 0.015],
                color="#263238",
                linewidth=1.4,
            )
            ax.plot(
                [1.44, 1.56],
                [gnd_y + 0.008, gnd_y + 0.008],
                color="#263238",
                linewidth=1.2,
            )
            ax.plot(
                [1.46, 1.54],
                [gnd_y + 0.002, gnd_y + 0.002],
                color="#263238",
                linewidth=1.0,
            )

        ax.text(
            0.02,
            0.98,
            f"Topology: {topology} | Switch state inferred from phase voltage sign",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        if topology == "single":
            ax.text(
                0.02,
                0.90,
                "Single-shunt: shared low-side source bus -> one shunt -> ground",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#37474F",
            )
        self.bridge_figure.tight_layout()
        self.bridge_canvas.draw_idle()

    def _build_current_sense_model(self) -> InverterCurrentSense | None:
        """Create topology-aware current sense model from GUI settings."""
        if not hasattr(self, "current_sense_enable"):
            return None
        if not self.current_sense_enable.isChecked():
            return None

        topology_text = self.current_sense_topology.currentText().lower()
        if topology_text not in ("single", "double", "triple"):
            raise ValueError(f"Unsupported current-sense topology: {topology_text}")
        topology = cast(Literal["single", "double", "triple"], topology_text)
        expected_channels = {"single": 1, "double": 2, "triple": 3}[topology]
        actual_gains = [
            self.current_sense_actual_gain_a.value(),
            self.current_sense_actual_gain_b.value(),
            self.current_sense_actual_gain_c.value(),
        ]
        actual_offsets = [
            self.current_sense_actual_offset_a.value(),
            self.current_sense_actual_offset_b.value(),
            self.current_sense_actual_offset_c.value(),
        ]

        channels = []
        for idx in range(expected_channels):
            channels.append(
                ShuntAmplifierChannel(
                    r_shunt_ohm=self.current_sense_r_shunt.value(),
                    nominal_gain=self.current_sense_nominal_gain.value(),
                    nominal_offset_v=self.current_sense_nominal_offset.value(),
                    actual_gain=actual_gains[idx],
                    actual_offset_v=actual_offsets[idx],
                    cutoff_frequency_hz=self.current_sense_cutoff_hz.value(),
                    vcc=self.current_sense_vcc.value(),
                )
            )

        return InverterCurrentSense(topology=topology, channels=channels)

    def _update_runtime_current_sense_actuals(self) -> None:
        """Push runtime gain/offset edits into the active sense model during simulation."""
        if self.engine is None or getattr(self.engine, "current_sense", None) is None:
            return
        sense = self.engine.current_sense
        assert sense is not None
        actual_gains = [
            self.current_sense_actual_gain_a.value(),
            self.current_sense_actual_gain_b.value(),
            self.current_sense_actual_gain_c.value(),
        ]
        actual_offsets = [
            self.current_sense_actual_offset_a.value(),
            self.current_sense_actual_offset_b.value(),
            self.current_sense_actual_offset_c.value(),
        ]
        for idx in range(sense.n_shunts):
            sense.set_actual_channel(idx, gain=actual_gains[idx], offset_v=actual_offsets[idx])

    def _update_current_sense_status(self, snapshot: dict) -> None:
        """Update current sensing status and narrate measurement quality metrics."""
        meas = snapshot.get("current_measurement", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(meas, dict) or not meas.get("enabled", False):
            self.current_sense_status_label.setText(
                "Current sensing disabled. Enable to expose measured-vs-true current telemetry."
            )
            return

        topology = meas.get("topology", "unknown")
        sat = meas.get("adc_saturated", [])
        saturated_count = int(sum(1 for item in sat if bool(item))) if isinstance(sat, list) else 0
        drop = meas.get("phase_voltage_drop_v", [0.0, 0.0, 0.0])
        drop_arr = np.asarray(drop, dtype=np.float64) if isinstance(drop, list) else np.zeros(3)
        drop_rms = float(np.sqrt(np.mean(np.square(drop_arr)))) if drop_arr.size else 0.0

        # Compute per-phase instantaneous measurement error if true currents available
        err_text = ""
        err_speech = ""
        if self.engine is not None:
            hist = self.engine.get_history()
            has_true = (
                "currents_a_true" in hist
                and len(hist["currents_a_true"]) > 0
                and len(hist["currents_a"]) > 0
            )
            if has_true:
                errs_rms = []
                errs_peak = []
                for ph in ("a", "b", "c"):
                    m = np.asarray(hist[f"currents_{ph}"], dtype=np.float64)
                    t = np.asarray(hist[f"currents_{ph}_true"], dtype=np.float64)
                    diff = m - t
                    errs_rms.append(float(np.sqrt(np.mean(diff**2))))
                    errs_peak.append(float(np.max(np.abs(diff))))
                err_text = (
                    f" | Measurement error — "
                    f"RMS A:{errs_rms[0]:.4f} B:{errs_rms[1]:.4f} C:{errs_rms[2]:.4f} A; "
                    f"Peak A:{errs_peak[0]:.4f} B:{errs_peak[1]:.4f} C:{errs_peak[2]:.4f} A"
                )
                err_speech = (
                    f" Measurement RMS error: "
                    f"A {errs_rms[0]:.4f}, B {errs_rms[1]:.4f}, C {errs_rms[2]:.4f} amperes."
                )

        status_text = (
            f"Current sensing active. Topology: {topology}, "
            f"ADC saturation channels: {saturated_count}, "
            f"phase-drop RMS: {drop_rms:.4f} V" + err_text
        )
        self.current_sense_status_label.setText(status_text)

        # Periodic audio narration — every ~100 polling cycles to avoid speech spam.
        # Use a simple counter stored on self.
        poll_count = getattr(self, "_sense_status_poll_count", 0) + 1
        self._sense_status_poll_count = poll_count
        if poll_count % 100 == 1:  # narrate on first call and every 100 thereafter
            sat_msg = (
                f"{saturated_count} ADC channel{'s' if saturated_count != 1 else ''} saturated."
                if saturated_count > 0
                else "No ADC saturation."
            )
            speak(f"Current sensing active. Topology: {topology}. " + sat_msg + err_speech)

    def _open_current_fft_window(self) -> None:
        """Open or focus the asynchronous current FFT analysis window."""
        if self.current_fft_window is None:
            self.current_fft_window = CurrentSpectrumWindow(
                window_size_samples=int(self.current_sense_fft_window_samples.value()),
                parent=self,
            )
            self.current_fft_window.closed.connect(self._on_current_fft_window_closed)
        self._apply_fft_display_settings()
        self.current_fft_window.show()
        self.current_fft_window.raise_()
        self.current_fft_window.activateWindow()

    def _on_current_fft_window_closed(self) -> None:
        """Track FFT window closure to avoid stale references."""
        self.current_fft_window = None

    def _update_status_bar(self, snapshot: dict) -> None:
        """Update status bar with current simulation telemetry."""
        try:
            # Update simulation state
            state_text = "State: Running" if self.is_running else "State: Stopped"
            self.status_bar_state.setText(state_text)

            # Update task name
            task_name = self._get_running_task_name() or "None"
            self.status_bar_task.setText(f"Task: {task_name.capitalize()}")

            # Update remaining time estimate
            if self.is_running and self.sim_thread:
                current_time = snapshot.get("time", 0.0)
                duration = self.sim_duration.value()  # Get max duration from UI
                if duration > 0:
                    remaining = max(0.0, duration - current_time)
                    self.status_bar_time_remaining.setText(f"Remaining: {remaining:.1f}s")
                else:
                    # Infinite duration
                    self.status_bar_time_remaining.setText(f"Elapsed: {current_time:.1f}s")
            else:
                self.status_bar_time_remaining.setText("Remaining: -- s")

            # Update CPU load estimate
            cpu_load = snapshot.get("cpu_load_pct", 0.0)
            if cpu_load > 0:
                self.status_bar_cpu_load.setText(f"CPU: {cpu_load:.1f}%")
            else:
                self.status_bar_cpu_load.setText("CPU: -- %")

            # Update compute backend display
            backend_info = snapshot.get("compute_backend", {})
            if isinstance(backend_info, dict):
                selected = backend_info.get("selected", "--").upper()
                gpu_available = backend_info.get("gpu_available", False)
                if gpu_available and selected == "GPU":
                    self.status_bar_backend.setText(f"Backend: {selected}")
                elif gpu_available:
                    self.status_bar_backend.setText(f"Backend: {selected} (GPU avail)")
                else:
                    self.status_bar_backend.setText(f"Backend: {selected}")
            else:
                self.status_bar_backend.setText("Backend: --")

        except Exception as exc:
            # Keep GUI updates resilient, but do not swallow diagnostics.
            logger.debug("Monitoring panel update skipped due to runtime issue: %s", exc)

    def _on_simulation_finished(self):
        """Handle simulation thread completion."""
        self.is_running = False
        self._mark_task_finished("simulation")

        # Update status bar
        self.status_bar_state.setText("State: Stopped")
        self.status_bar_task.setText("Task: None")
        self.status_bar_time_remaining.setText("Remaining: -- s")
        self.status_bar_cpu_load.setText("CPU: -- %")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.update_timer.stop()

    def _on_load_type_changed(self):
        """Handle load type change."""
        load_type = self.load_type.currentText()
        self.load_constant_torque.setVisible(load_type == "Constant")
        self.load_initial_torque.setVisible(load_type == "Ramp")
        self.load_final_torque.setVisible(load_type == "Ramp")
        self.load_ramp_duration.setVisible(load_type == "Ramp")

    def _export_data(self):
        """Export simulation data to CSV."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No simulation data to export!")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Data", "", "CSV Files (*.csv)"
        )

        if filename:
            metadata = self._collect_simulation_configuration()

            try:
                self.logger.save_simulation_data(
                    self.engine.get_history(), metadata, filename, use_custom_path=True
                )
                QMessageBox.information(
                    self,
                    "Success",
                    f"Data saved to {filename}\nMetadata saved with _metadata suffix",
                )
                speak("Data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e!s}")
                logger.error(f"Export failed: {e!s}")

    def _plot_3phase(self):
        """Generate 3-phase plot."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_3phase_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_3phase = figure
        figure.show()
        speak("3-phase plot generated.")

    def _plot_currents(self):
        """Generate current plot."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_current_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_currents = figure
        figure.show()
        speak("Current plot generated.")

    def _plot_pfc_analysis(self):
        """Generate dedicated PFC telemetry plot."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_pfc_analysis_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_pfc = figure
        figure.show()
        speak("PFC analysis plot generated.")

    def _plot_efficiency_analysis(self):
        """Generate dedicated efficiency telemetry plot."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_efficiency_analysis_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_efficiency = figure
        figure.show()
        speak("Efficiency analysis plot generated.")

    def _plot_inverter_analysis(self):
        """Generate dedicated inverter-realism telemetry plot."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_inverter_analysis_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_inverter = figure
        figure.show()
        speak("Inverter analysis plot generated.")

    def _plot_measured_vs_true(self):
        """Generate measured-vs-true current overlay with RMS error panel."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return

        history = self.engine.get_history()
        has_true = "currents_a_true" in history and len(history["currents_a_true"]) > 0
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_measured_vs_true_current_plot(
            history,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_measured_vs_true = figure
        figure.show()
        if has_true:
            # Compute and narrate summary RMS errors
            errs = []
            for ph in ("a", "b", "c"):
                meas = np.asarray(history[f"currents_{ph}"], dtype=np.float64)
                true_v = np.asarray(history[f"currents_{ph}_true"], dtype=np.float64)
                rms = float(np.sqrt(np.mean((meas - true_v) ** 2)))
                errs.append(rms)
            speak(
                f"Measured vs true current plot generated. "
                f"RMS measurement error: phase A {errs[0]:.4f} A, "
                f"phase B {errs[1]:.4f} A, phase C {errs[2]:.4f} A."
            )
        else:
            speak(
                "Measured vs true current plot generated. "
                "No true-current history available; enable current sensing and re-run simulation."
            )

    def _show_efficiency_recommendations(self):
        """Show heuristic efficiency tuning suggestions for the current setup."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(
                self,
                "Warning",
                "Run a simulation first to generate efficiency recommendations.",
            )
            return

        efficiency_state = self.engine.get_efficiency_state()
        pfc_state = self.engine.get_power_factor_control_state()
        rec = recommend_efficiency_adjustments(
            efficiency=efficiency_state.get("efficiency", 0.0),
            power_factor=pfc_state.get("power_factor", 0.0),
            device_drop_v=self.inverter_device_drop.value(),
            dead_time_fraction=self.inverter_dead_time_fraction.value(),
            conduction_resistance_ohm=self.inverter_conduction_resistance.value(),
            switching_frequency_hz=self.inverter_switching_frequency.value(),
            switching_loss_coeff_v_per_a_khz=self.inverter_switching_loss_coeff.value(),
        )
        lines = [
            f"Efficiency: {rec['efficiency']:.3f}",
            f"Power factor: {rec['power_factor']:.3f}",
            "",
            "Suggestions:",
        ]
        lines.extend(f"- {item}" for item in rec["suggestions"])
        QMessageBox.information(self, "Efficiency Tuning Suggestions", "\n".join(lines))

    def _plot_custom(self):
        """Generate a custom multi-axis plot from selected variables."""
        if not self.engine or len(self.engine.get_history()["time"]) == 0:
            QMessageBox.warning(self, "Warning", "No data to plot!")
            return
        history = self.engine.get_history()
        # populate variable list if not done yet
        if self.plot_var_list.rowCount() == 0:
            keys = [k for k in history.keys() if k != "time"]
            self.plot_var_list.setRowCount(len(keys))
            for i, k in enumerate(keys):
                item = QTableWidgetItem(k)
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                )
                self.plot_var_list.setItem(i, 0, item)
        selected = [item.text() for item in self.plot_var_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "Warning", "No variables selected for plotting.")
            return
        grid_on = bool(
            getattr(self, "plot_grid_checkbox", None) and self.plot_grid_checkbox.isChecked()
        )
        grid_spacing = getattr(self, "plot_grid_spacing", None) and self.plot_grid_spacing.value()
        minor_grid = bool(
            getattr(self, "plot_minor_grid_checkbox", None)
            and self.plot_minor_grid_checkbox.isChecked()
        )
        grid_spacing_y = (
            getattr(self, "plot_grid_spacing_y", None) and self.plot_grid_spacing_y.value()
        )
        figure = SimulationPlotter.create_multi_axis_plot(
            history,
            selected,
            grid_on=grid_on,
            grid_spacing=grid_spacing,
            minor_grid=minor_grid,
            grid_spacing_y=grid_spacing_y,
        )
        self._last_fig_custom = figure
        figure.show()
        speak("Custom plot generated for variables: " + ", ".join(selected))
