"""
Configuration Module
====================

Central configuration for BLDC motor control application.

Contains default parameters, paths, and application settings.

:author: BLDC Control Team
:version: 0.8.0
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
PLOTS_DIR = DATA_DIR / "plots"
MOTOR_PROFILES_DIR = DATA_DIR / "motor_profiles"

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MOTOR_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Default motor parameters
DEFAULT_MOTOR_PARAMS = {
    "nominal_voltage": 48.0,  # V
    "phase_resistance": 2.5,  # Ohms
    "phase_inductance": 0.005,  # H
    "back_emf_constant": 0.1,  # V*s/rad
    "torque_constant": 0.1,  # N*m/A
    "rotor_inertia": 0.0005,  # kg*m^2
    "friction_coefficient": 0.001,  # N*m*s/rad
    "num_poles": 8,
    "poles_pairs": 4,
}

# Simulation parameters
DEFAULT_PWM_FREQUENCY_HZ = 20000.0

SIMULATION_PARAMS = {
    "pwm_frequency_hz": DEFAULT_PWM_FREQUENCY_HZ,
    "dt": 1.0 / DEFAULT_PWM_FREQUENCY_HZ,  # Simulation/control step [s]
    "compute_backend": "auto",  # auto -> gpu when available, else cpu
    "max_history": 100000,  # Maximum data points to store
    "dc_voltage": 48.0,  # DC bus voltage [V]
}

# V/f controller defaults
VF_CONTROLLER_PARAMS = {
    "v_nominal": 48.0,  # Nominal voltage [V]
    "f_nominal": 100.0,  # Nominal frequency [Hz]
    "v_startup": 1.0,  # Startup voltage [V]
    "ramp_rate": 10.0,  # Voltage ramp [V/s]
    "frequency_slew_rate": 50.0,  # Frequency slew [Hz/s]
    "startup_sequence_enabled": False,
    "startup_align_duration_s": 0.05,
    "startup_align_voltage_v": 1.5,
    "startup_align_angle_deg": 0.0,
    "startup_ramp_initial_frequency_hz": 2.0,
}

FOC_STARTUP_PARAMS = {
    "startup_sequence_enabled": False,
    "startup_align_duration_s": 0.05,
    "startup_align_current_a": 1.5,
    "startup_align_angle_deg": 0.0,
    "startup_open_loop_initial_speed_rpm": 30.0,
    "startup_open_loop_target_speed_rpm": 300.0,
    "startup_open_loop_ramp_time_s": 0.2,
    "startup_open_loop_id_ref_a": 0.0,
    "startup_open_loop_iq_ref_a": 2.0,
}

FOC_FIELD_WEAKENING_PARAMS = {
    "enabled": False,
    "start_speed_rpm": 12000.0,
    "gain": 1.0,
    "max_negative_id_a": 8.0,
    "headroom_target_v": 1.2,
}

# SOGI adaptive EMF filter defaults (Solution 1 in FOCController)
#
# * ``enabled`` toggles the SOGI bandpass on the reconstructed back-EMF.
# * ``k`` is the damping coefficient (sqrt(2) gives a critically-damped
#   resonant response with zero phase lag at the fundamental).
# * ``discretization`` selects the integration scheme used to step the
#   SOGI biquad:
#       - ``"euler"``  : legacy forward-Euler (cheap, drifts at high
#                        omega_e * Ts).
#       - ``"tustin"`` : prewarped bilinear transform (extra biquad
#                        state, exact resonance at every sample).
#   See the v5 addendum of the sensorless observers report for a
#   dual-motor comparison of the two schemes.
FOC_SOGI_PARAMS = {
    "enabled": False,
    "k": 1.4142135623730951,
    "discretization": "euler",
}

# Load profile defaults
DEFAULT_LOAD_PROFILE = {
    "type": "constant",  # 'constant', 'ramp', 'variable', 'cyclic'
    "torque": 0.5,  # N*m (for constant)
    "initial_torque": 0.0,  # N*m (for ramp)
    "final_torque": 2.0,  # N*m (for ramp)
    "ramp_duration": 1.0,  # seconds (for ramp)
}

# GUI parameters
GUI_PARAMS = {
    "update_interval": 100,  # GUI update interval [ms]
    "plot_buffer_size": 1000,  # Number of points to show in plots
    "theme": "light",  # 'light' or 'dark'
    "font_size": 10,
}

# Data logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Visualization defaults
PLOT_STYLE = {
    "current_color": "#1f77b4",  # Blue
    "voltage_color": "#ff7f0e",  # Orange
    "speed_color": "#2ca02c",  # Green
    "torque_color": "#d62728",  # Red
    "live_speed_color": "#FF6B9D",  # Hot Pink for live plot
}
