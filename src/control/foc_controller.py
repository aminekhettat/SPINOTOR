"""
Field-Oriented Control (FOC) Module
===================================

Implements basic FOC algorithm with PI current regulators and optional
auto-tuning functionality. Supports choosing Clarke or Concordia
transforms and running in either polar or Cartesian output modes.

:author: BLDC Control Team
:version: 0.8.0

.. versionadded:: 0.8.0
    Initial FOC implementation with PI regulators and auto-tune stub
"""

import logging

import numpy as np

from src.core.motor_model import BLDCMotor

from .base_controller import BaseController
from .transforms import (
    clarke_transform,
    concordia_transform,
    inverse_park,
    park_transform,
)

logger = logging.getLogger(__name__)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _blend_angles(theta_a: float, theta_b: float, weight_b: float) -> float:
    """Blend two wrapped angles using vector interpolation."""
    w = float(np.clip(weight_b, 0.0, 1.0))
    s = (1.0 - w) * np.sin(theta_a) + w * np.sin(theta_b)
    c = (1.0 - w) * np.cos(theta_a) + w * np.cos(theta_b)
    return float(np.arctan2(s, c) % (2 * np.pi))


def _pi_update(state: dict, error: float, dt: float) -> float:
    """Update simple PI controller with state dict containing kp, ki, integral."""
    kp = state["kp"]
    ki = state["ki"]
    state["integral"] += error * dt
    return float(kp * error + ki * state["integral"])


def _pi_update_anti_windup(
    state: dict[str, float],
    error: float,
    dt: float,
    limit: float | None = None,
) -> float:
    """PI update with optional back-calculation anti-windup and output limiting."""
    kp = state["kp"]
    ki = state["ki"]
    kaw = state.get("kaw", 0.0)

    integral = state["integral"]
    u_unsat = kp * error + ki * integral

    if limit is not None:
        u_sat = float(np.clip(u_unsat, -limit, limit))
    else:
        u_sat = u_unsat

    # Back-calculation anti-windup term uses (u_sat - u_unsat).
    state["integral"] += (error + kaw * (u_sat - u_unsat)) * dt
    return u_sat


class FOCController(BaseController):
    """
    Basic Field-Oriented Controller supporting current loops and speed
    reference.  Designed for integration with SVM generator, returning either
    (magnitude, angle) or Cartesian (v_alpha, v_beta) depending on mode.

    **Features:**
      - d/q current PI regulators
      - speed reference (via iq setpoint)
      - optional auto-tuning stub
      - transform choice (clarke or concordia)
      - output in polar or Cartesian
    """

    def __init__(
        self,
        motor: BLDCMotor,
        use_concordia: bool = False,
        output_cartesian: bool = False,
        enable_speed_loop: bool = False,
    ):
        """Initialize FOC controller.

        :param motor: BLDCMotor instance for state feedback
        :param use_concordia: Choose concordia transform instead of Clarke
        :param output_cartesian: If True, return (v_alpha, v_beta) instead of
                                 magnitude/angle
        """
        super().__init__()
        self.motor = motor
        self.use_concordia = use_concordia
        self.output_cartesian = output_cartesian
        self.enable_speed_loop = enable_speed_loop

        # PI states for d and q axes
        self.pi_d = {"kp": 1.0, "ki": 0.1, "kaw": 0.5, "integral": 0.0}
        self.pi_q = {"kp": 1.0, "ki": 0.1, "kaw": 0.5, "integral": 0.0}
        self.pi_speed = {"kp": 0.02, "ki": 1.0, "kaw": 0.3, "integral": 0.0}

        # References
        self.id_ref = 0.0
        self.iq_ref = 0.0
        self.speed_ref = 0.0
        self.speed_error = 0.0

        # Cascaded-loop limits.
        self.iq_limit_a = 30.0
        self.vdq_limit = self.motor.params.nominal_voltage / np.sqrt(3.0)
        self.voltage_saturation_mode = "proportional"
        self.coupled_voltage_antiwindup_enabled = False
        self.coupled_voltage_antiwindup_gain = 0.15

        # Optional decoupling feed-forward terms for current loops.
        self.enable_decouple_d = False
        self.enable_decouple_q = False
        self.last_v_d_ff = 0.0
        self.last_v_q_ff = 0.0

        # Internal angle state
        self.theta = 0.0
        self.last_v_d = 0.0
        self.last_v_q = 0.0

        # Angle observer configuration and states.
        self.observer_target_mode = "Measured"
        self.angle_observer_mode = "Measured"  # Measured | PLL | SMO
        self.pll = {"kp": 80.0, "ki": 2000.0, "integral": 0.0}
        self.smo = {
            "k_slide": 600.0,
            "lpf_alpha": 0.08,
            "boundary": 0.06,
            "omega_est": 0.0,
        }
        self.theta_est_pll = 0.0
        self.theta_est_smo = 0.0
        self.theta_meas_emf = 0.0
        self.theta_error_pll = 0.0
        self.theta_error_smo = 0.0
        self.emf_observer_mag = 0.0

        # ── Sensorless EMF reconstructor ──────────────────────────────────────
        # When enabled, back-EMF is reconstructed from applied voltages and
        # measured currents: e_αβ = v_αβ − R·i_αβ − L·Δi_αβ/dt
        # This replaces the simulation shortcut of reading motor.back_emf and
        # motor.omega directly, making the observers truly sensorless.
        self._sensorless_emf_enabled = False  # enabled via enable_sensorless_emf_reconstruction()
        self._emf_recon_R = float(motor.params.phase_resistance)
        self._emf_recon_L = float(
            motor.params.ld if motor.params.ld is not None else motor.params.phase_inductance
        )
        _tau_e = self._emf_recon_L / max(self._emf_recon_R, 1e-9)
        self._emf_recon_lpf_tau = 3.0 * _tau_e  # LPF time constant (default 3×τ_e)
        self._e_alpha_obs = 0.0  # reconstructed EMF state, α axis
        self._e_beta_obs = 0.0  # reconstructed EMF state, β axis
        self._i_alpha_prev = 0.0  # previous-step α current for di/dt
        self._i_beta_prev = 0.0  # previous-step β current for di/dt
        self._omega_elec_est = 0.0  # estimated ω_e from |E| = Ke·ω_mech
        self.emf_reconstructed_mag = 0.0  # diagnostic
        self._use_estimated_speed_ff = True  # use ω_est instead of motor.omega as FF
        self._v_alpha_prev = 0.0  # applied α-voltage from previous step (for EMF recon)
        self._v_beta_prev = 0.0  # applied β-voltage from previous step (for EMF recon)

        # ── Solution 1: SOGI adaptive EMF filter ─────────────────────────────
        # Replaces the fixed-frequency LPF with a speed-adaptive resonant
        # bandpass (Second Order Generalized Integrator).  Zero phase lag at
        # the fundamental electrical frequency → eliminates +9° error at 3000 RPM.
        self._use_sogi_filter = False
        self._sogi_discretization = "euler"  # "euler" (forward) or "tustin" (bilinear, prewarped)
        self._sogi_k = float(np.sqrt(2.0))  # SOGI damping (√2 ≈ 1.414)
        self._sogi_e_alpha = 0.0  # in-phase SOGI output, α axis
        self._sogi_e_alpha_90 = 0.0  # quadrature SOGI output, α axis
        self._sogi_e_beta = 0.0  # in-phase SOGI output, β axis
        self._sogi_e_beta_90 = 0.0  # quadrature SOGI output, β axis
        # Tustin SOGI biquad memory (per axis): u[n-1], u[n-2], y[n-1], y[n-2]
        self._sogi_ua1 = 0.0
        self._sogi_ua2 = 0.0
        self._sogi_ya1 = 0.0
        self._sogi_ya2 = 0.0
        self._sogi_ub1 = 0.0
        self._sogi_ub2 = 0.0
        self._sogi_yb1 = 0.0
        self._sogi_yb2 = 0.0

        # ── Solution 2: Active Flux observer (Boldea 2009) ────────────────────
        # ψa = ψs − Ld·is always points along the d-axis → saliency-immune
        # position estimation decoupled from FW flux reduction.
        self._use_active_flux = False
        self._psi_s_alpha = 0.0  # stator flux estimate, α axis [Wb·rad_e]
        self._psi_s_beta = 0.0  # stator flux estimate, β axis
        self._psi_af_prev_a = 0.0  # previous ψa_α for speed estimate
        self._psi_af_prev_b = 0.0  # previous ψa_β for speed estimate
        self._theta_est_af = 0.0  # Active Flux angle estimate [rad]
        self._omega_est_af = 0.0  # Active Flux speed estimate [rad_e/s]
        self._psi_af_mag = 0.0  # |ψa| diagnostic [Wb]
        self._af_omega_c = 2.0 * np.pi * 0.5  # drift corrector cutoff [rad/s]
        _ke_raw = getattr(motor.params, "back_emf_constant", None)
        self._emf_recon_ke = float(_ke_raw) if _ke_raw else 0.028

        # ── Solution 3: EEMF model (saliency Ld ≠ Lq) ───────────────────────
        # Uses Lq (not Ld) for the L·di/dt term, absorbing saliency into the
        # EEMF vector which remains proportional to [−sin θe; cos θe].
        self._use_eemf_model = False
        _lq_raw = getattr(motor.params, "lq", None)
        self._emf_recon_Lq = float(_lq_raw) if _lq_raw else self._emf_recon_L

        # ── Solution 4: Super-Twisting SMO (ST-SMO) ──────────────────────────
        # Replaces standard sign()+LPF with ST algorithm that converges to
        # back-EMF in finite time without requiring a subsequent LPF.
        # The integral state z→ê at steady state → zero phase lag.
        #
        # ⚠️  MOTOR COMPATIBILITY: SPM only (Ld ≈ Lq).
        # The current model uses a single inductance L = Ld.  On a salient
        # IPMSM (Lq ≠ Ld), the missing term (Lq−Ld)·id·ωe biases σ
        # unboundedly in field-weakening → z1 drift → divergence.
        # For salient IPMSM use ActiveFlux or EEMF-STSMO (Wang 2022).
        self._use_stsmo = False
        self.stsmo = {"k1": 100.0, "k2": 2000.0}
        self._stsmo_i_alpha = 0.0  # ST-SMO estimated current, α [A]
        self._stsmo_i_beta = 0.0  # ST-SMO estimated current, β [A]
        self._stsmo_z1_alpha = 0.0  # ST integrator state, α [V]
        self._stsmo_z1_beta = 0.0  # ST integrator state, β [V]
        self._stsmo_e_alpha = 0.0  # ST-SMO back-EMF estimate, α [V]
        self._stsmo_e_beta = 0.0  # ST-SMO back-EMF estimate, β [V]

        # ── Solution 5: Speed feedforward from vq voltage model ───────────────
        # ωe ≈ (vq − R·iq) / λeff  — bypasses the |E|/λ circular dependency.
        self._use_vq_speed_ff = False
        self._omega_vq_est = 0.0  # vq-model speed estimate [rad_m/s]
        self._omega_vq_alpha = 0.05  # blend LPF toward vq estimate

        # ── Solution 6: MRAS resistance adaptation ────────────────────────────
        # Online R estimation: dR̂/dt = −γR·(ê_α·i_α + ê_β·i_β).
        # Compensates +20–40 % thermal R drift in FW zone.
        self._use_mras_resistance = False
        self._mras_R = float(motor.params.phase_resistance)
        self._mras_gamma_r = 1.0  # adaptation gain [Ω/(V·A·s)]
        self._mras_R_min = 0.5 * float(motor.params.phase_resistance)
        self._mras_R_max = 4.0 * float(motor.params.phase_resistance)

        # ── Solution 7: Adaptive PLL bandwidth ────────────────────────────────
        # kp = 2ζ·ωn(ω̂e)  ki = ωn(ω̂e)²  where ωn ∝ |ω̂e|.
        # Widens BW at high speed → faster FW transient tracking.
        self._use_adaptive_pll_bw = False
        self._pll_zeta = 0.9
        self._pll_omega_n_factor = 0.10  # ωn = factor × |ω̂e|
        self._pll_omega_n_floor = 50.0  # minimum ωn [rad/s]
        self._pll_omega_n_ceil = 600.0  # maximum ωn [rad/s]

        # Sensorless startup transition (initial observer -> target observer).
        self.startup_transition_enabled = False
        self.startup_initial_mode = "Measured"
        self.startup_min_speed_rpm = 300.0
        self.startup_min_elapsed_s = 0.05
        self.startup_min_emf_v = 0.5
        self.startup_min_confidence = 0.6
        self.startup_confidence_hold_s = 0.02
        self.startup_confidence_hysteresis = 0.1
        self.startup_fallback_enabled = True
        self.startup_fallback_hold_s = 0.03
        self.startup_elapsed_s = 0.0
        self.startup_confidence_elapsed_s = 0.0
        self.startup_fallback_elapsed_s = 0.0
        self.startup_transition_done = True
        self.startup_ready_to_switch = False
        self.startup_fallback_event_count = 0
        self.observer_confidence = 0.0
        self.observer_confidence_emf = 0.0
        self.observer_confidence_speed = 0.0
        self.observer_confidence_coherence = 0.0
        self.observer_confidence_ema = 0.0
        self.observer_confidence_trend = 0.0
        self.observer_confidence_prev_above = False
        self.observer_confidence_above_threshold_time_s = 0.0
        self.observer_confidence_below_threshold_time_s = 0.0
        self.observer_confidence_crossings_up = 0
        self.observer_confidence_crossings_down = 0
        self.startup_handoff_count = 0
        self.startup_last_handoff_time_s = 0.0
        self.startup_last_handoff_confidence = 0.0
        self.startup_handoff_confidence_peak = 0.0
        self.startup_handoff_quality = 0.0
        self.startup_handoff_stability_ratio = 1.0

        # Standard startup sequence: align -> open-loop ramp -> closed-loop.
        self.startup_sequence_enabled = False
        self.startup_phase = "closed_loop"
        self.startup_sequence_elapsed_s = 0.0
        self.startup_phase_elapsed_s = 0.0
        self.startup_align_duration_s = 0.05
        self.startup_align_current_a = 1.5
        self.startup_align_angle = 0.0
        self.startup_open_loop_initial_speed_rpm = 30.0
        self.startup_open_loop_target_speed_rpm = 300.0
        self.startup_open_loop_ramp_time_s = 0.2
        self.startup_open_loop_id_ref_a = 0.0
        self.startup_open_loop_iq_ref_a = 2.0
        self.startup_open_loop_speed_rpm = 0.0
        self.startup_open_loop_angle = 0.0
        self.id_ref_command = 0.0
        self.iq_ref_command = 0.0

        # ── Multi-rate speed loop ──────────────────────────────────────────────
        # On a real single-shunt FOC MCU (e.g. Fpwm = 20 kHz):
        #   • Clarke/Park + current PI + observer : every PWM period  (50 µs)
        #   • Speed PI                            : every N periods  (~10 ms → N≈200)
        #
        # speed_loop_divider = 1 → legacy behaviour: speed PI runs every step.
        # speed_loop_divider = N → speed PI runs every N current-loop steps;
        #   effective dt for the speed integrator = N × current dt.
        self.speed_loop_divider: int = 1
        self._speed_loop_counter: int = 0
        self._last_iq_speed_ref: float = 0.0
        # FW integrator update divider — defaults to speed_loop_divider.
        # In a real MCU the FW block runs in the same slow ISR as the speed PI
        # (both at ~100 Hz).  Use configure_mcu_timing() to set both together.
        self.fw_loop_divider: int = 1
        self._fw_loop_counter: int = 0
        # ──────────────────────────────────────────────────────────────────────

        # Optional field-weakening scheduler (independent feature toggle).
        self.field_weakening_enabled = False
        self.field_weakening_start_speed_rpm = 0.0
        self.field_weakening_gain = 1.0
        self.field_weakening_max_negative_id_a = 0.0
        self.field_weakening_headroom_target_v = 0.08 * self.vdq_limit
        self.field_weakening_headroom_v = self.vdq_limit
        self.field_weakening_voltage_error_v = 0.0
        self.field_weakening_id_injection_a = 0.0

        # Sensorless maturity: blend measured and observer angle at low confidence
        # to improve low-speed robustness and reduce abrupt phase behavior.
        self.sensorless_blend_enabled = True
        self.sensorless_blend_min_speed_rpm = 250.0
        self.sensorless_blend_min_confidence = 0.65
        self.sensorless_blend_weight = 0.0
        self.theta_sensorless_raw = 0.0

        # Current-feedback source selection.
        self.use_external_current_feedback = False
        self._external_phase_currents = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _normalize_observer_mode(mode: str) -> str:
        normalized = str(mode).strip().upper()
        mapping = {
            "MEASURED": "Measured",
            "PLL": "PLL",
            "SMO": "SMO",
            "SLIDING": "SMO",
            "SLIDING MODE": "SMO",
        }
        if normalized not in mapping:
            raise ValueError("Unsupported angle observer mode")
        return mapping[normalized]

    def set_angle_observer(self, mode: str = "Measured") -> None:
        """Set electrical angle observer mode: Measured, PLL or SMO."""
        self.observer_target_mode = self._normalize_observer_mode(mode)
        if not self.startup_transition_enabled or self.startup_transition_done:
            self.angle_observer_mode = self.observer_target_mode

    def set_startup_transition(
        self,
        enabled: bool,
        initial_mode: str = "Measured",
        min_speed_rpm: float = 300.0,
        min_elapsed_s: float = 0.05,
        min_emf_v: float = 0.5,
        min_confidence: float = 0.6,
        confidence_hold_s: float = 0.02,
        confidence_hysteresis: float = 0.1,
        fallback_enabled: bool = True,
        fallback_hold_s: float = 0.03,
    ) -> None:
        """Configure observer startup handoff from an initial to target mode."""
        if min_speed_rpm < 0.0:
            raise ValueError("min_speed_rpm must be non-negative")
        if min_elapsed_s < 0.0:
            raise ValueError("min_elapsed_s must be non-negative")
        if min_emf_v < 0.0:
            raise ValueError("min_emf_v must be non-negative")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if confidence_hold_s < 0.0:
            raise ValueError("confidence_hold_s must be non-negative")
        if confidence_hysteresis < 0.0 or confidence_hysteresis > 1.0:
            raise ValueError("confidence_hysteresis must be in [0, 1]")
        if fallback_hold_s < 0.0:
            raise ValueError("fallback_hold_s must be non-negative")

        self.startup_transition_enabled = bool(enabled)
        self.startup_initial_mode = self._normalize_observer_mode(initial_mode)
        self.startup_min_speed_rpm = float(min_speed_rpm)
        self.startup_min_elapsed_s = float(min_elapsed_s)
        self.startup_min_emf_v = float(min_emf_v)
        self.startup_min_confidence = float(min_confidence)
        self.startup_confidence_hold_s = float(confidence_hold_s)
        self.startup_confidence_hysteresis = float(confidence_hysteresis)
        self.startup_fallback_enabled = bool(fallback_enabled)
        self.startup_fallback_hold_s = float(fallback_hold_s)

        self.startup_elapsed_s = 0.0
        self.startup_confidence_elapsed_s = 0.0
        self.startup_fallback_elapsed_s = 0.0
        self.startup_ready_to_switch = False
        self.startup_transition_done = not self.startup_transition_enabled
        self.startup_fallback_event_count = 0
        self.startup_handoff_count = 0
        self.startup_last_handoff_time_s = 0.0
        self.startup_last_handoff_confidence = 0.0
        self.startup_handoff_confidence_peak = 0.0
        self.startup_handoff_quality = 0.0
        self.startup_handoff_stability_ratio = 1.0
        self.observer_confidence_prev_above = False
        self.observer_confidence_above_threshold_time_s = 0.0
        self.observer_confidence_below_threshold_time_s = 0.0
        self.observer_confidence_crossings_up = 0
        self.observer_confidence_crossings_down = 0
        self.angle_observer_mode = (
            self.startup_initial_mode
            if self.startup_transition_enabled
            else self.observer_target_mode
        )

    def set_startup_sequence(
        self,
        enabled: bool,
        align_duration_s: float = 0.05,
        align_current_a: float = 1.5,
        align_angle_deg: float = 0.0,
        open_loop_initial_speed_rpm: float = 30.0,
        open_loop_target_speed_rpm: float = 300.0,
        open_loop_ramp_time_s: float = 0.2,
        open_loop_id_ref_a: float = 0.0,
        open_loop_iq_ref_a: float = 2.0,
    ) -> None:
        """Configure a standard sensored/sensorless startup sequence."""
        if align_duration_s < 0.0:
            raise ValueError("align_duration_s must be non-negative")
        if align_current_a < 0.0:
            raise ValueError("align_current_a must be non-negative")
        if open_loop_initial_speed_rpm < 0.0:
            raise ValueError("open_loop_initial_speed_rpm must be non-negative")
        if open_loop_target_speed_rpm < 0.0:
            raise ValueError("open_loop_target_speed_rpm must be non-negative")
        if open_loop_ramp_time_s < 0.0:
            raise ValueError("open_loop_ramp_time_s must be non-negative")

        self.startup_sequence_enabled = bool(enabled)
        self.startup_align_duration_s = float(align_duration_s)
        self.startup_align_current_a = float(align_current_a)
        self.startup_align_angle = float(np.deg2rad(align_angle_deg) % (2 * np.pi))
        self.startup_open_loop_initial_speed_rpm = float(open_loop_initial_speed_rpm)
        self.startup_open_loop_target_speed_rpm = float(open_loop_target_speed_rpm)
        self.startup_open_loop_ramp_time_s = float(open_loop_ramp_time_s)
        self.startup_open_loop_id_ref_a = float(open_loop_id_ref_a)
        self.startup_open_loop_iq_ref_a = float(open_loop_iq_ref_a)
        self._reset_startup_sequence_runtime()

    def _has_position_sensor(self) -> bool:
        """Treat direct measured angle mode as sensored operation."""
        return self.observer_target_mode == "Measured"

    def _enter_startup_phase(self, phase: str) -> None:
        """Switch startup phase and reset phase-local timers."""
        self.startup_phase = phase
        self.startup_phase_elapsed_s = 0.0
        if phase == "align":
            self.angle_observer_mode = "Alignment"
            self.startup_open_loop_angle = self.startup_align_angle
        elif phase == "open_loop":
            self.angle_observer_mode = "OpenLoop"
            measured_theta = (self.motor.theta * self.motor.params.poles_pairs) % (2 * np.pi)
            self.startup_open_loop_angle = measured_theta
            self.startup_open_loop_speed_rpm = self.startup_open_loop_initial_speed_rpm
            self.startup_elapsed_s = 0.0
            self.startup_confidence_elapsed_s = 0.0
            self.startup_fallback_elapsed_s = 0.0
            self.startup_ready_to_switch = False
            self.startup_transition_done = False
            # ── Reset current-PI integrals so the alignment→open-loop transition
            # does not cause a torque spike.  During alignment the d-PI integral
            # accumulates to hold id = align_current; if it is not reset it
            # creates a large v_q spike on the first open-loop step that kicks
            # the motor far ahead of the open-loop angle (≈180° slip), making
            # _v_alpha_prev point in the wrong direction and corrupting the EMF
            # reconstruction with a 180° phase error.
            self.pi_d["integral"] = 0.0
            self.pi_q["integral"] = 0.0
            self._v_alpha_prev = 0.0
            self._v_beta_prev = 0.0
        else:
            self.angle_observer_mode = self.observer_target_mode
            # ── Bumpless closed-loop transition ────────────────────────────
            # During open-loop the observers ran in proportional-only mode and
            # have been continuously tracking theta_meas_emf (the true motor
            # electrical angle reconstructed from actual terminal voltages and
            # measured currents).  By the time the transition fires, both
            # theta_est_pll and theta_est_smo should be close to the true
            # motor angle — DO NOT overwrite them with startup_open_loop_angle
            # (which lags the motor when it overshoots the open-loop ramp).
            #
            # "j'aligne les angles et je lance l'integrateur" means:
            #   • angles are already aligned by pre-convergence → no reset
            #   • launch the integrator from zero → pll["integral"] = 0
            #   • give SMO a sensible initial speed estimate
            if not self._has_position_sensor():
                # Keep theta_est_pll / theta_est_smo — already pre-converged.
                self.pll["integral"] = 0.0
                # Initialise SMO speed from the reconstructed electrical speed
                # so the first closed-loop integration step is smooth.
                self.smo["omega_est"] = (
                    self._omega_elec_est
                    if self._omega_elec_est > 0.0
                    else (
                        self.startup_open_loop_speed_rpm
                        / 60.0
                        * 2.0
                        * float(np.pi)
                        * float(self.motor.params.poles_pairs)
                    )
                )

    def _reset_startup_sequence_runtime(self) -> None:
        """Reset phase runtime for the standard startup sequence."""
        self.startup_sequence_elapsed_s = 0.0
        self.startup_phase_elapsed_s = 0.0
        self.startup_open_loop_speed_rpm = self.startup_open_loop_initial_speed_rpm
        self.startup_open_loop_angle = self.startup_align_angle
        # Reset SOGI states
        self._sogi_e_alpha = 0.0
        self._sogi_e_alpha_90 = 0.0
        self._sogi_e_beta = 0.0
        self._sogi_e_beta_90 = 0.0
        self._sogi_ua1 = 0.0
        self._sogi_ua2 = 0.0
        self._sogi_ya1 = 0.0
        self._sogi_ya2 = 0.0
        self._sogi_ub1 = 0.0
        self._sogi_ub2 = 0.0
        self._sogi_yb1 = 0.0
        self._sogi_yb2 = 0.0
        # Reset Active Flux states
        self._psi_s_alpha = 0.0
        self._psi_s_beta = 0.0
        self._psi_af_prev_a = 0.0
        self._psi_af_prev_b = 0.0
        self._theta_est_af = 0.0
        self._omega_est_af = 0.0
        # Reset ST-SMO states
        self._stsmo_i_alpha = 0.0
        self._stsmo_i_beta = 0.0
        self._stsmo_z1_alpha = 0.0
        self._stsmo_z1_beta = 0.0
        self._stsmo_e_alpha = 0.0
        self._stsmo_e_beta = 0.0
        # Reset vq-FF and MRAS states
        self._omega_vq_est = 0.0
        self._mras_R = self._emf_recon_R
        if not self.startup_sequence_enabled:
            self.startup_phase = "closed_loop"
            return
        if self.startup_align_duration_s > 0.0:
            self._enter_startup_phase("align")
        elif self._has_position_sensor():
            self._enter_startup_phase("closed_loop")
        else:
            self._enter_startup_phase("open_loop")

    def _get_observer_emf_components(self, dt: float) -> tuple[float, float, float]:
        """Return observer EMF components in the selected stationary frame."""
        if self._sensorless_emf_enabled:
            if self._use_stsmo:
                return self._update_stsmo_emf(dt)
            return self._reconstruct_emf_sensorless(dt)

        emf_a, emf_b, emf_c = self.motor.back_emf
        if self.use_concordia:
            emf_alpha, emf_beta = concordia_transform(emf_a, emf_b, emf_c)
        else:
            emf_alpha, emf_beta = clarke_transform(emf_a, emf_b, emf_c)
        emf_mag = float(np.hypot(emf_alpha, emf_beta))
        return emf_alpha, emf_beta, emf_mag

    def _update_measured_observer_theta(self, emf_alpha: float, emf_beta: float) -> None:
        """Update measured electrical angle from EMF components when available."""
        if abs(emf_alpha) + abs(emf_beta) <= 1e-12:
            return

        if self._sensorless_emf_enabled:
            # For the dq SPMSM model, EMF reconstruction from v_αβ and i_αβ gives:
            #   e_α = v_α − R·iα − L·diα/dt  =  −Ke·ω·sin(θe)   (negative sign)
            #   e_β = v_β − R·iβ − L·diβ/dt  =  +Ke·ω·cos(θe)   (positive sign)
            # ∴ arctan2(−e_α, e_β) = arctan2(sin θe, cos θe) = θe  ✓
            # (Using arctan2(e_α, −e_β) would give θe + π — 180° wrong.)
            self.theta_meas_emf = float(np.arctan2(-emf_alpha, emf_beta)) % (2 * np.pi)
            return

        self.theta_meas_emf = float(np.arctan2(emf_beta, emf_alpha)) % (2 * np.pi)

    def _get_observer_speed_feedforward(self) -> float:
        """Return electrical-speed feedforward for observer and decoupling paths."""
        if self._sensorless_emf_enabled and self._use_estimated_speed_ff:
            return self._omega_elec_est
        return self.motor.omega * self.motor.params.poles_pairs

    def _update_target_observer_mode(
        self,
        theta_measured: float,
        omega_ff: float,
        dt: float,
    ) -> float:
        """Advance the configured target observer and return its angle."""
        open_loop_freeze = self.startup_phase == "open_loop"

        if self.observer_target_mode == "PLL":
            err = _wrap_angle(self.theta_meas_emf - self.theta_est_pll)
            self.theta_error_pll = err
            self.theta_error_smo = 0.0
            # Solution 7: adaptive PLL bandwidth — ωn ∝ |ω̂e|
            if self._use_adaptive_pll_bw and abs(self._omega_elec_est) > 10.0:
                omega_n = float(
                    np.clip(
                        self._pll_omega_n_factor * abs(self._omega_elec_est),
                        self._pll_omega_n_floor,
                        self._pll_omega_n_ceil,
                    )
                )
                kp = 2.0 * self._pll_zeta * omega_n
                ki = omega_n**2
            else:
                kp = self.pll["kp"]
                ki = self.pll["ki"]
            if not open_loop_freeze:
                self.pll["integral"] += err * dt
            omega_correction = kp * err + ki * self.pll["integral"]
            self.theta_est_pll = (self.theta_est_pll + (omega_ff + omega_correction) * dt) % (
                2 * np.pi
            )
            return self.theta_est_pll

        if self.observer_target_mode == "SMO":
            err = _wrap_angle(self.theta_meas_emf - self.theta_est_smo)
            self.theta_error_smo = err
            self.theta_error_pll = 0.0
            boundary = self.smo["boundary"]
            slide = float(np.tanh(err / boundary))
            omega_target = omega_ff + self.smo["k_slide"] * slide
            alpha = self.smo["lpf_alpha"]
            self.smo["omega_est"] = (1.0 - alpha) * self.smo["omega_est"] + alpha * omega_target
            self.theta_est_smo = (self.theta_est_smo + self.smo["omega_est"] * dt) % (2 * np.pi)
            return self.theta_est_smo

        if self.observer_target_mode == "ActiveFlux":
            # Solution 2: Active Flux observer — run the integrator and return θ̂
            ia, ib, ic = self.motor.currents
            if self.use_concordia:
                i_alpha, i_beta = concordia_transform(ia, ib, ic)
            else:
                i_alpha, i_beta = clarke_transform(ia, ib, ic)
            theta_af = self._update_active_flux(dt, i_alpha, i_beta)
            self.theta_error_pll = float(_wrap_angle(self.theta_meas_emf - theta_af))
            self.theta_error_smo = 0.0
            return theta_af

        self.theta_error_pll = 0.0
        self.theta_error_smo = 0.0
        return theta_measured

    def _get_observer_speed_mag_rpm(self) -> float:
        """Return observer speed magnitude in RPM for confidence metrics."""
        if self._sensorless_emf_enabled and self._use_estimated_speed_ff:
            pp = float(self.motor.params.poles_pairs)
            return abs(self._omega_elec_est) / max(pp, 1.0) * 30.0 / np.pi
        return abs(self.motor.speed_rpm)

    def _update_target_observer_theta(self, dt: float) -> tuple[float, float, float]:
        """Update target observer states and return measured/target angles."""
        theta_measured = (self.motor.theta * self.motor.params.poles_pairs) % (2 * np.pi)

        emf_alpha, emf_beta, emf_mag = self._get_observer_emf_components(dt)
        self.emf_observer_mag = emf_mag

        self._update_measured_observer_theta(emf_alpha, emf_beta)
        omega_ff = self._get_observer_speed_feedforward()
        theta_target = self._update_target_observer_mode(theta_measured, omega_ff, dt)
        speed_mag_rpm = self._get_observer_speed_mag_rpm()
        speed_norm = min(speed_mag_rpm / max(self.startup_min_speed_rpm, 1e-9), 1.0)
        emf_norm = min(emf_mag / max(self.startup_min_emf_v, 1e-9), 1.0)
        phase_err = abs(_wrap_angle(self.theta_meas_emf - theta_target))
        coherence = 1.0 - min(phase_err / np.pi, 1.0)

        self.observer_confidence_emf = float(np.clip(emf_norm, 0.0, 1.0))
        self.observer_confidence_speed = float(np.clip(speed_norm, 0.0, 1.0))
        self.observer_confidence_coherence = float(np.clip(coherence, 0.0, 1.0))
        self.observer_confidence = float(
            np.clip(
                0.45 * self.observer_confidence_emf
                + 0.35 * self.observer_confidence_speed
                + 0.20 * self.observer_confidence_coherence,
                0.0,
                1.0,
            )
        )

        conf_prev = self.observer_confidence_ema
        conf_alpha = 0.15
        self.observer_confidence_ema = float(
            (1.0 - conf_alpha) * conf_prev + conf_alpha * self.observer_confidence
        )
        self.observer_confidence_trend = float(
            np.clip(self.observer_confidence_ema - conf_prev, -1.0, 1.0)
        )

        conf_above = self.observer_confidence >= self.startup_min_confidence
        if conf_above:
            self.observer_confidence_above_threshold_time_s += dt
        else:
            self.observer_confidence_below_threshold_time_s += dt
        if conf_above and not self.observer_confidence_prev_above:
            self.observer_confidence_crossings_up += 1
        elif (not conf_above) and self.observer_confidence_prev_above:
            self.observer_confidence_crossings_down += 1
        self.observer_confidence_prev_above = conf_above
        self.theta_sensorless_raw = theta_target
        return theta_measured, theta_target, emf_mag

    def _apply_sensorless_blend(self, theta_measured: float, theta_target: float) -> float:
        """Blend measured and observer angles for low-speed robustness."""
        if self.observer_target_mode == "Measured":
            self.sensorless_blend_weight = 0.0
            return theta_measured
        if not self.sensorless_blend_enabled:
            self.sensorless_blend_weight = 1.0
            return theta_target

        w_speed = np.clip(
            abs(self.motor.speed_rpm) / max(self.sensorless_blend_min_speed_rpm, 1e-9),
            0.0,
            1.0,
        )
        w_conf = np.clip(
            self.observer_confidence / max(self.sensorless_blend_min_confidence, 1e-9),
            0.0,
            1.0,
        )
        self.sensorless_blend_weight = float(w_speed * w_conf)
        return _blend_angles(theta_measured, theta_target, self.sensorless_blend_weight)

    def _evaluate_sensorless_handoff(self, dt: float, emf_mag: float, confidence: float) -> bool:
        """Evaluate when it is safe to leave forced open-loop startup."""
        self.startup_elapsed_s += dt

        conf_hi = self.startup_min_confidence
        speed_mag = abs(self.motor.speed_rpm)
        speed_ready = speed_mag >= self.startup_min_speed_rpm
        time_ready = self.startup_elapsed_s >= self.startup_min_elapsed_s
        emf_ready = emf_mag >= self.startup_min_emf_v
        static_ready = bool(speed_ready and time_ready and emf_ready)

        if confidence >= conf_hi:
            self.startup_confidence_elapsed_s += dt
        else:
            self.startup_confidence_elapsed_s = 0.0

        adaptive_ready = self.startup_confidence_elapsed_s >= self.startup_confidence_hold_s
        self.startup_ready_to_switch = bool(static_ready and adaptive_ready)
        self.startup_handoff_confidence_peak = max(self.startup_handoff_confidence_peak, confidence)
        return self.startup_transition_enabled and self.startup_ready_to_switch

    def _record_sensorless_handoff(self, confidence: float, emf_mag: float) -> None:
        """Store open-loop to closed-loop handoff diagnostics."""
        speed_mag = abs(self.motor.speed_rpm)
        speed_score = min(speed_mag / max(self.startup_min_speed_rpm, 1e-9), 1.0)
        emf_score = min(emf_mag / max(self.startup_min_emf_v, 1e-9), 1.0)
        time_score = min(self.startup_elapsed_s / max(self.startup_min_elapsed_s, 1e-9), 1.0)
        self.startup_handoff_quality = float(
            np.clip(
                0.5 * confidence + 0.2 * speed_score + 0.2 * emf_score + 0.1 * time_score,
                0.0,
                1.0,
            )
        )
        self.startup_handoff_count += 1
        self.startup_last_handoff_time_s = self.startup_sequence_elapsed_s
        self.startup_last_handoff_confidence = float(confidence)
        self.startup_transition_done = True
        self.startup_ready_to_switch = True
        self.startup_fallback_elapsed_s = 0.0

    def _maybe_apply_sensorless_fallback(
        self, dt: float, emf_mag: float, confidence: float
    ) -> bool:
        """Return True when closed-loop should fall back to forced open-loop."""
        if not self.startup_fallback_enabled:
            return False

        conf_lo = max(0.0, self.startup_min_confidence - self.startup_confidence_hysteresis)
        speed_release = self.startup_min_speed_rpm * 0.9
        emf_release = self.startup_min_emf_v * 0.9
        degraded = (
            confidence <= conf_lo
            or abs(self.motor.speed_rpm) < speed_release
            or emf_mag < emf_release
        )
        if degraded:
            self.startup_fallback_elapsed_s += dt
        else:
            self.startup_fallback_elapsed_s = 0.0

        if self.startup_fallback_elapsed_s < self.startup_fallback_hold_s:
            return False

        self.startup_fallback_elapsed_s = 0.0
        self.startup_transition_done = False
        self.startup_ready_to_switch = False
        self.startup_elapsed_s = 0.0
        self.startup_confidence_elapsed_s = 0.0
        self.startup_fallback_event_count += 1
        total_transitions = self.startup_handoff_count + self.startup_fallback_event_count
        self.startup_handoff_stability_ratio = (
            self.startup_handoff_count / total_transitions if total_transitions > 0 else 1.0
        )
        return True

    def _estimate_theta_with_startup_sequence(self, dt: float) -> float:
        """Resolve control angle using the standard startup phase machine."""
        theta_measured, theta_target, emf_mag = self._update_target_observer_theta(dt)
        self.startup_sequence_elapsed_s += dt
        self.startup_phase_elapsed_s += dt

        if self.startup_phase == "align":
            self.angle_observer_mode = "Alignment"
            self.sensorless_blend_weight = 0.0
            if self.startup_phase_elapsed_s >= self.startup_align_duration_s:
                if self._has_position_sensor():
                    self.startup_transition_done = True
                    self.startup_ready_to_switch = True
                    self._enter_startup_phase("closed_loop")
                else:
                    self.startup_transition_done = False
                    self._enter_startup_phase("open_loop")
            return self.startup_align_angle

        if self.startup_phase == "open_loop":
            self.angle_observer_mode = "OpenLoop"
            self.sensorless_blend_weight = 0.0
            sign = -1.0 if self.speed_ref < 0.0 else 1.0
            if self.startup_open_loop_ramp_time_s <= 0.0:
                ramp_ratio = 1.0
            else:
                ramp_ratio = min(
                    self.startup_phase_elapsed_s / self.startup_open_loop_ramp_time_s,
                    1.0,
                )
            self.startup_open_loop_speed_rpm = (
                self.startup_open_loop_initial_speed_rpm
                + (
                    self.startup_open_loop_target_speed_rpm
                    - self.startup_open_loop_initial_speed_rpm
                )
                * ramp_ratio
            )
            omega_elec = (
                sign
                * (self.startup_open_loop_speed_rpm / 60.0)
                * (2 * np.pi)
                * self.motor.params.poles_pairs
            )
            self.startup_open_loop_angle = (self.startup_open_loop_angle + omega_elec * dt) % (
                2 * np.pi
            )

            if self._evaluate_sensorless_handoff(dt, emf_mag, self.observer_confidence):
                self._record_sensorless_handoff(self.observer_confidence, emf_mag)
                self._enter_startup_phase("closed_loop")
                return self._apply_sensorless_blend(theta_measured, theta_target)
            return self.startup_open_loop_angle

        self.startup_transition_done = True
        self.startup_ready_to_switch = True
        self.angle_observer_mode = self.observer_target_mode
        if not self._has_position_sensor() and self._maybe_apply_sensorless_fallback(
            dt, emf_mag, self.observer_confidence
        ):
            self._enter_startup_phase("open_loop")
            return self.startup_open_loop_angle
        return self._apply_sensorless_blend(theta_measured, theta_target)

    def _get_active_references(self, dt: float) -> tuple[float, float]:
        """Return d/q references after applying startup phase overrides.

        Multi-rate speed loop
        ~~~~~~~~~~~~~~~~~~~~~
        The speed PI is executed only every ``speed_loop_divider`` calls,
        matching the real MCU timing model where:

        * Current PI + Clarke/Park + observer → every PWM period (e.g. 50 µs)
        * Speed PI → every N periods (e.g. N=200 → 10 ms at 20 kHz)

        Between speed-PI executions the last Iq reference is held, exactly as a
        real interrupt-driven implementation would do with a slow-loop flag.
        The effective integration step for the speed PI is ``N × dt`` so gains
        tuned with ``speed_loop_divider=1`` remain valid; only the update
        cadence changes.
        """
        id_ref = self.id_ref + self.field_weakening_id_injection_a

        if self.enable_speed_loop:
            speed_ref_rad_s = (self.speed_ref / 60.0) * (2 * np.pi)
            self.speed_error = speed_ref_rad_s - self.motor.omega

            if self.speed_loop_divider <= 1:
                # Legacy / high-fidelity mode: speed PI every current step.
                self._last_iq_speed_ref = _pi_update_anti_windup(
                    self.pi_speed, self.speed_error, dt, limit=self.iq_limit_a
                )
            else:
                # Multi-rate: only integrate the speed PI every N steps.
                # Hold _last_iq_speed_ref in between (ZOH, same as real MCU).
                self._speed_loop_counter += 1
                if self._speed_loop_counter >= self.speed_loop_divider:
                    self._speed_loop_counter = 0
                    dt_speed = dt * float(self.speed_loop_divider)
                    self._last_iq_speed_ref = _pi_update_anti_windup(
                        self.pi_speed,
                        self.speed_error,
                        dt_speed,
                        limit=self.iq_limit_a,
                    )
            iq_ref = self._last_iq_speed_ref
        else:
            iq_ref = (self.speed_ref / 60.0) * (2 * np.pi) * self.pi_q["kp"]
            self.speed_error = 0.0

        if not self.startup_sequence_enabled:
            return id_ref, iq_ref

        if self.startup_phase == "align":
            return self.startup_align_current_a, 0.0
        if self.startup_phase == "open_loop":
            return self.startup_open_loop_id_ref_a, self.startup_open_loop_iq_ref_a
        return id_ref, iq_ref

    def _resolve_observer_mode(self, dt: float, emf_mag: float, confidence: float) -> str:
        """Resolve active observer mode considering startup transition logic."""
        if not self.startup_transition_enabled:
            self.startup_transition_done = True
            self.startup_ready_to_switch = False
            return self.observer_target_mode

        self.startup_elapsed_s += dt
        speed_mag = abs(self.motor.speed_rpm)
        conf_hi = self.startup_min_confidence
        conf_lo = max(0.0, conf_hi - self.startup_confidence_hysteresis)

        speed_ready = abs(self.motor.speed_rpm) >= self.startup_min_speed_rpm
        time_ready = self.startup_elapsed_s >= self.startup_min_elapsed_s
        emf_ready = emf_mag >= self.startup_min_emf_v
        static_ready = bool(speed_ready and time_ready and emf_ready)

        if confidence >= conf_hi:
            self.startup_confidence_elapsed_s += dt
        else:
            self.startup_confidence_elapsed_s = 0.0

        adaptive_ready = self.startup_confidence_elapsed_s >= self.startup_confidence_hold_s
        self.startup_ready_to_switch = bool(static_ready and adaptive_ready)

        if not self.startup_transition_done:
            self.startup_handoff_confidence_peak = max(
                self.startup_handoff_confidence_peak, confidence
            )

        if not self.startup_transition_done and self.startup_ready_to_switch:
            speed_score = min(speed_mag / max(self.startup_min_speed_rpm, 1e-9), 1.0)
            emf_score = min(emf_mag / max(self.startup_min_emf_v, 1e-9), 1.0)
            time_score = min(self.startup_elapsed_s / max(self.startup_min_elapsed_s, 1e-9), 1.0)
            self.startup_handoff_quality = float(
                np.clip(
                    0.5 * confidence + 0.2 * speed_score + 0.2 * emf_score + 0.1 * time_score,
                    0.0,
                    1.0,
                )
            )
            self.startup_handoff_count += 1
            self.startup_last_handoff_time_s = self.startup_elapsed_s
            self.startup_last_handoff_confidence = float(confidence)
            self.startup_transition_done = True
            self.startup_fallback_elapsed_s = 0.0

        if self.startup_transition_done and self.startup_fallback_enabled:
            speed_release = self.startup_min_speed_rpm * 0.9
            emf_release = self.startup_min_emf_v * 0.9
            degraded = confidence <= conf_lo or speed_mag < speed_release or emf_mag < emf_release
            if degraded:
                self.startup_fallback_elapsed_s += dt
            else:
                self.startup_fallback_elapsed_s = 0.0

            if self.startup_fallback_elapsed_s >= self.startup_fallback_hold_s:
                self.startup_transition_done = False
                self.startup_ready_to_switch = False
                self.startup_elapsed_s = 0.0
                self.startup_confidence_elapsed_s = 0.0
                self.startup_fallback_elapsed_s = 0.0
                self.startup_fallback_event_count += 1
                total_transitions = self.startup_handoff_count + self.startup_fallback_event_count
                self.startup_handoff_stability_ratio = (
                    self.startup_handoff_count / total_transitions if total_transitions > 0 else 1.0
                )
                return self.startup_initial_mode

        total_transitions = self.startup_handoff_count + self.startup_fallback_event_count
        self.startup_handoff_stability_ratio = (
            self.startup_handoff_count / total_transitions if total_transitions > 0 else 1.0
        )

        if self.startup_transition_done:
            return self.observer_target_mode
        return self.startup_initial_mode

    def set_pll_gains(self, kp: float, ki: float) -> None:
        """Set PLL observer gains."""
        self.pll["kp"] = float(kp)
        self.pll["ki"] = float(ki)

    def set_smo_gains(
        self,
        k_slide: float,
        lpf_alpha: float,
        boundary: float = 0.06,
    ) -> None:
        """Set SMO-like observer gains and smoothing constants."""
        if lpf_alpha <= 0.0 or lpf_alpha > 1.0:
            raise ValueError("lpf_alpha must be in (0, 1]")
        if boundary <= 0.0:
            raise ValueError("boundary must be positive")
        self.smo["k_slide"] = float(k_slide)
        self.smo["lpf_alpha"] = float(lpf_alpha)
        self.smo["boundary"] = float(boundary)

    # ── Sensorless EMF reconstructor public API ─────────────────────────────

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — 7 improvement solutions
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Solution 1 ────────────────────────────────────────────────────────────
    def enable_sogi_filter(self, k: float = 1.4142, discretization: str = "euler") -> None:
        """Replace the fixed-frequency LPF with an adaptive SOGI bandpass filter.

        The SOGI (Second Order Generalized Integrator) has zero phase lag at
        the fundamental electrical frequency and tracks it automatically via the
        current ``_omega_elec_est``.  This eliminates the +9° systematic angle
        error that the LPF introduces at 3 000 RPM.

        Parameters
        ----------
        k : float
            SOGI damping coefficient.  √2 ≈ 1.414 gives critical damping
            (no overshoot, fast settling).  Range [0.5, 2].
        discretization : {"euler", "tustin"}
            Numerical integration scheme.  ``"euler"`` is the legacy forward-
            Euler update (low cost, frequency-dependent warping at high ωe·Ts).
            ``"tustin"`` uses the bilinear (trapezoidal) transform with the
            standard prewarped SOGI biquad, eliminating the per-step phase
            drift introduced by Euler when ωe approaches Nyquist/8.
        """
        self._use_sogi_filter = True
        self._sogi_k = float(np.clip(k, 0.1, 5.0))
        mode = str(discretization).strip().lower()
        if mode not in ("euler", "tustin"):
            raise ValueError("discretization must be 'euler' or 'tustin'")
        self._sogi_discretization = mode
        # Reset biquad memory whenever the scheme is (re)selected to avoid a
        # transient from stale forward-Euler states when switching to Tustin.
        self._sogi_ua1 = 0.0
        self._sogi_ua2 = 0.0
        self._sogi_ya1 = 0.0
        self._sogi_ya2 = 0.0
        self._sogi_ub1 = 0.0
        self._sogi_ub2 = 0.0
        self._sogi_yb1 = 0.0
        self._sogi_yb2 = 0.0

    def configure_sogi_from_dict(self, config: dict) -> None:
        """Apply a SOGI configuration dictionary.

        The dictionary follows the schema of ``FOC_SOGI_PARAMS`` in
        ``src.utils.config``:

        * ``enabled`` (bool): if ``False``, the SOGI bandpass is disabled and
          the standard first-order LPF remains in use.
        * ``k`` (float): SOGI damping coefficient, clipped to the valid range.
        * ``discretization`` (str): ``"euler"`` (forward) or ``"tustin"``
          (prewarped bilinear).

        Raises
        ------
        TypeError
            If ``config`` is not a mapping.
        ValueError
            If ``discretization`` is unknown.
        """
        if not isinstance(config, dict):
            raise TypeError("SOGI config must be a dict")
        enabled = bool(config.get("enabled", False))
        if not enabled:
            self._use_sogi_filter = False
            return
        k = float(config.get("k", float(np.sqrt(2.0))))
        discretization = str(config.get("discretization", "euler"))
        # Delegates validation and biquad-memory reset to enable_sogi_filter.
        self.enable_sogi_filter(k=k, discretization=discretization)

    # ── Solution 2 ────────────────────────────────────────────────────────────
    def enable_active_flux_observer(self, dc_cutoff_hz: float = 0.5) -> None:
        """Enable the Active Flux observer (Boldea 2009) as the angle source.

        Defines the "active flux" vector  ψa = ψs − Ld·is  which is always
        aligned with the rotor d-axis regardless of FW id injection.  Position
        is extracted as θe = arctan2(ψa_β, ψa_α).

        The stator flux is integrated with a slow drift-correction pole at
        ``dc_cutoff_hz`` to suppress integrator offset.

        Parameters
        ----------
        dc_cutoff_hz : float
            Cutoff frequency [Hz] of the drift-correction pole.  Must be much
            lower than the minimum electrical frequency (typically < 1 Hz).
        """
        self._use_active_flux = True
        self._af_omega_c = 2.0 * float(np.pi) * float(dc_cutoff_hz)
        self.observer_target_mode = "ActiveFlux"

    def _update_active_flux(self, dt: float, i_alpha: float, i_beta: float) -> float:
        """Advance Active Flux observer one step; return estimated θe [rad]."""
        dt_s = max(dt, 1e-12)
        R = self._mras_R if self._use_mras_resistance else self._emf_recon_R
        Ld = self._emf_recon_L

        # ── Stator flux integration (drift-compensated) ───────────────────────
        # dψs/dt = v_s − R·i_s  → discrete forward Euler + leaky term
        # Leaky integrator: ψs[k+1] = (1 − ωc·dt)·ψs[k] + dt·(vs − R·is)
        leak = 1.0 - self._af_omega_c * dt_s
        self._psi_s_alpha = leak * self._psi_s_alpha + dt_s * (self._v_alpha_prev - R * i_alpha)
        self._psi_s_beta = leak * self._psi_s_beta + dt_s * (self._v_beta_prev - R * i_beta)

        # ── Active flux ψa = ψs − Ld·is ──────────────────────────────────────
        psi_a_alpha = self._psi_s_alpha - Ld * i_alpha
        psi_a_beta = self._psi_s_beta - Ld * i_beta
        self._psi_af_mag = float(np.hypot(psi_a_alpha, psi_a_beta))

        if self._psi_af_mag > 1e-6:
            self._theta_est_af = float(np.arctan2(psi_a_beta, psi_a_alpha)) % (2.0 * float(np.pi))

        # ── Speed estimate from ψa cross-product ──────────────────────────────
        # ωe = (dψa_β·ψa_α − dψa_α·ψa_β) / |ψa|²
        dpa = psi_a_alpha - self._psi_af_prev_a
        dpb = psi_a_beta - self._psi_af_prev_b
        if self._psi_af_mag > 1e-6:
            omega_raw = (dpb * psi_a_alpha - dpa * psi_a_beta) / (self._psi_af_mag**2 * dt_s)
            # Low-pass blend
            self._omega_est_af = 0.9 * self._omega_est_af + 0.1 * omega_raw
            self._omega_elec_est = self._omega_est_af

        self._psi_af_prev_a = psi_a_alpha
        self._psi_af_prev_b = psi_a_beta
        return self._theta_est_af

    # ── Solution 3 ────────────────────────────────────────────────────────────
    def enable_eemf_model(self, Lq: float | None = None) -> None:
        """Use the EEMF model (Chen 2003) for saliency-robust reconstruction.

        Replaces the Ld·di/dt term with Lq·di/dt in the raw EMF formula.
        The resulting EEMF vector remains proportional to [−sin θe; cos θe]
        even when Ld ≠ Lq, making arctan2 extraction valid for IPMSMs.

        Parameters
        ----------
        Lq : float, optional
            q-axis inductance [H].  Defaults to ``motor.params.lq``.
        """
        self._use_eemf_model = True
        if Lq is not None:
            self._emf_recon_Lq = float(Lq)

    # ── Solution 4 ────────────────────────────────────────────────────────────
    def calibrate_stsmo_gains_analytical(
        self,
        rated_rpm: float | None = None,
        convergence_factor: float = 3.0,
        k2_min: float | None = None,
        apply: bool = True,
    ) -> dict:
        """Compute Super-Twisting SMO gains analytically.

        Based on the Levant (1993) sufficient conditions for finite-time
        convergence:  k1 = λ·√(E_max),  k2 = λ²·E_max / 2

        where  E_max = Ke·ωm_max  is the expected maximum back-EMF amplitude
        and  λ = convergence_factor  controls speed of convergence.

        Note: with the backward-Euler implicit integrator the forward-Euler
        stability constraint  k1 < L/(√2·dt) ≈ 1.06  is fully lifted.
        The gains are therefore chosen solely for convergence speed (λ ≥ 1)
        without any upper stability bound.

        The stored k2 value is now used as a **reference** only.  At runtime,
        `_update_stsmo_emf` overrides k2 with a speed-adaptive formula:

            k2_eff = max(k2_min, k2_factor · ke · ωm_est · ωe_est)

        where ``k2_factor = λ²`` is stored in ``self.stsmo["k2_factor"]``.
        This ensures z1 can track the rotating back-EMF at any speed, while
        a SOGI post-filter suppresses the resulting high-frequency chattering.

        Parameters
        ----------
        rated_rpm : float, optional
            Rated speed [RPM].  Defaults to motor rated speed.
        convergence_factor : float
            λ — convergence multiplier (>1 ensures robustness).  Use 3.0 for
            isotropic SPM motors and 4.0 for salient IPM motors.
        k2_min : float, optional
            Floor for the speed-adaptive k2 [V/s].  Guards STSMO tracking
            during the open-loop startup ramp when ωm ≈ 0.  If None, computed
            automatically as ``max(50, 1.5 · Ke · ωe_max)``, which scales with
            the motor's back-EMF constant and rated electrical speed.
        apply : bool
            If True, immediately loads the gains into ``self.stsmo``.

        Returns
        -------
        dict with keys ``k1``, ``k2``, ``e_max_v``, ``k2_min``.
        """
        if rated_rpm is None:
            rated_rpm = float(getattr(self.motor.params, "rated_speed_rpm", None) or 3500.0)
        ke = float(self.motor.params.back_emf_constant)
        pp = float(self.motor.params.poles_pairs)
        # ke is in V·s/rad_mech; use mechanical angular speed for correct EMF
        ωm_max = rated_rpm / 60.0 * 2.0 * float(np.pi)
        ωe_max = ωm_max * pp
        e_max = ke * ωm_max  # max back-EMF amplitude [V]  (≈8.8 V Nanotec)
        lam = float(convergence_factor)
        k1 = lam * float(np.sqrt(e_max))
        k2 = (lam**2) * e_max / 2.0
        # Motor-aware k2_min: proportional to Ke × ωe_max so the floor covers
        # the EMF rate-of-change during a typical open-loop startup acceleration.
        # Formula: 1.5 × Ke × ωe_max gives a physical bound that scales with
        # the motor's EMF constant and rated electrical speed.
        if k2_min is None:
            k2_min_val = float(np.clip(1.5 * ke * ωe_max, 50.0, 20000.0))
        else:
            k2_min_val = float(np.clip(k2_min, 50.0, 20000.0))
        if apply:
            self.stsmo["k1"] = k1
            self.stsmo["k2"] = k2  # rated-speed reference value (informational)
            # k2_factor = 1.0 is the Levant theoretical minimum for the
            # rotating-EMF tracking condition k2 ≥ ke·ωm·ωe.  Larger values
            # increase chattering amplitude; smaller values cause z1 to lag.
            # The SOGI post-filter suppresses chattering, so 1.0 is optimal.
            self.stsmo["k2_factor"] = 1.0
            self.stsmo["k2_min"] = k2_min_val  # motor-aware floor [V/s]
            self._use_stsmo = True
        return {"k1": k1, "k2": k2, "e_max_v": e_max, "k2_min": k2_min_val}

    def _update_stsmo_emf(self, dt: float) -> tuple[float, float, float]:  # noqa: C901
        """Advance Super-Twisting SMO one step — backward-Euler integration.

        Uses a fully implicit (backward-Euler) discretisation of the current
        estimator so the sliding-surface update is unconditionally stable for
        any k1 > 0.  The implicit equation

            σ[k+1] + B·k1·|σ[k+1]|^0.5·sign(σ[k+1]) = c

        (where A = L/(L+R·dt), B = dt/(L+R·dt),
         c = A·î[k] + B·(v[k]−z1[k]) − i_meas)

        is solved analytically in closed form: the function
        f(σ) = σ + B·k1·|σ|^0.5·sign(σ) is strictly monotone, so the solution
        is unique and reduces to a quadratic with substitution u = |σ|^0.5.

        This removes the forward-Euler stability constraint
        k1 < L/(√2·dt) that was incompatible with convergence at 10 kHz /
        L = 0.15 mH.  The integral state z1 still converges to the back-EMF in
        finite time provided k2 > 0.

        Returns (e_alpha, e_beta, |E|).
        """
        ia, ib, ic = self.motor.currents
        if self.use_concordia:
            i_alpha, i_beta = concordia_transform(ia, ib, ic)
        else:
            i_alpha, i_beta = clarke_transform(ia, ib, ic)

        R = self._mras_R if self._use_mras_resistance else self._emf_recon_R
        L = self._emf_recon_L
        dt_s = max(dt, 1e-12)
        k1 = self.stsmo["k1"]

        # ── Speed-adaptive k2 ────────────────────────────────────────────────
        # Levant condition: k2 ≥ λ²·|ė_max| where |ė| = ke·ωm·ωe (rotating EMF).
        # A fixed k2 based on amplitude (old formula) is ~408× too small at
        # rated speed.  Adaptive scaling ensures z1 can track the rotating
        # back-EMF at any operating point without excessive chattering.
        _ke_a = float(self.motor.params.back_emf_constant)
        _pp_a = float(self.motor.params.poles_pairs)
        # Use best available speed estimate for gain scheduling.
        # Priority: (1) open-loop reference speed during startup (known exact),
        # (2) omega_elec_est from PLL when above threshold, (3) true motor.omega
        # bootstrap when estimate is unreliable.  This prevents the chicken-and-egg
        # where a wrong omega_elec_est underestimates k2 → z1 can't track EMF →
        # emf_mag ≈ 0 → omega_elec_est never corrects.
        _oe_a = abs(self._omega_elec_est)
        # During open-loop startup, the reference speed is known — use it as a
        # reliable floor so k2 is always sufficient even before PLL convergence.
        if self.startup_phase == "open_loop":
            _ol_rpm = abs(self.startup_open_loop_speed_rpm)
            _oe_ol = _ol_rpm / 60.0 * 2.0 * float(np.pi) * _pp_a
            _oe_a = max(_oe_a, _oe_ol)
        # Always floor with motor.omega*Pp so k2 is sufficient as motor accelerates.
        _oe_motor_k2 = abs(float(self.motor.omega)) * _pp_a
        _oe_a = max(_oe_a, _oe_motor_k2, 10.0)
        _om_a = _oe_a / max(_pp_a, 1.0)
        _k2_fac = float(self.stsmo.get("k2_factor", 1.0))
        _k2_min = float(self.stsmo.get("k2_min", 500.0))
        k2 = max(_k2_min, _k2_fac * _ke_a * _om_a * _oe_a)

        # ── First-call EMF warm-start ─────────────────────────────────────────
        # Backward-Euler is unconditionally stable so z1=0 is safe, but
        # pre-loading z1 with a rough EMF estimate accelerates convergence:
        # without it z1 needs ~5–10 ms to climb from 0 to the true back-EMF
        # value.  On the first call after reset, initialise from omega_elec_est
        # (or fall back to motor.omega for the bootstrap).
        # omega_elec_est may still be 0 when the STSMO starts (the standard
        # path that sets it was bypassed), so fall back to motor.omega for
        # the one-time bootstrap only.
        _z_fresh = abs(self._stsmo_z1_alpha) < 1e-6 and abs(self._stsmo_z1_beta) < 1e-6
        if _z_fresh:
            _ke_v = float(self.motor.params.back_emf_constant)
            _pp_v = float(self.motor.params.poles_pairs)
            # Prefer the accumulated omega estimate; use open-loop reference during
            # startup (exact knowledge of the commanded frequency); fall back to
            # motor.omega truth for one-time bootstrap.
            _oe = abs(self._omega_elec_est)
            if self.startup_phase == "open_loop":
                _ol_rpm_v = abs(self.startup_open_loop_speed_rpm)
                _oe_ol_v = _ol_rpm_v / 60.0 * 2.0 * float(np.pi) * _pp_v
                _oe = max(_oe, _oe_ol_v)
            if _oe < 10.0:
                _oe = abs(float(self.motor.omega)) * _pp_v  # one-time bootstrap
            if _oe > 10.0:
                _emf0 = _ke_v * _oe / max(_pp_v, 1.0)
                _th0 = float(self.theta_meas_emf)
                self._stsmo_z1_alpha = float(-_emf0 * np.sin(_th0))
                self._stsmo_z1_beta = float(_emf0 * np.cos(_th0))
                self._omega_elec_est = _oe  # seed so subsequent steps stay valid
            self._stsmo_i_alpha = i_alpha  # start current estimate from measurement
            self._stsmo_i_beta = i_beta

        # Safety guard: snap if estimate diverged (should not happen after warm-start)
        if abs(self._stsmo_i_alpha - i_alpha) > 3.0 or abs(self._stsmo_i_beta - i_beta) > 3.0:
            self._stsmo_i_alpha = i_alpha
            self._stsmo_i_beta = i_beta

        # ── Backward-Euler implicit solve ────────────────────────────────────
        # Discretise  L·dî/dt = v − R·î − λ(σ)  with full implicit scheme:
        #   L·(î[k+1]−î[k])/dt = v[k] − R·î[k+1] − k1·|σ[k+1]|^0.5·sign(σ[k+1]) − z1[k]
        #
        # Let  A = L/(L+R·dt),  B = dt/(L+R·dt),  σ'= î[k+1]−i_meas.
        # Then:  σ' + B·k1·|σ'|^0.5·sign(σ') = c
        #        c = A·î[k] + B·(v[k]−z1[k]) − i_meas
        #
        # Closed-form solution — f(σ) strictly monotone ⇒ unique root:
        #   c ≥ 0:  u² + Bk1·u − c = 0,  u = √σ' ≥ 0  →  σ' = +u²
        #   c < 0:  u² + Bk1·u + c = 0,  u = √(−σ') ≥ 0 → σ' = −u²
        #   (discriminant = Bk1² + 4|c| > 0 always)
        #
        # Stability: regardless of k1, |σ[k+1]| ≤ |c|/(1+…) < |σ[k]| near
        # origin ⇒ unconditionally stable; forward-Euler constraint lifted.
        _den = L + R * dt_s  # effective impedance [Ω]  (≈ 1.62 Ω @ 10 kHz)
        _A_be = L / _den  # memory coefficient  (< 1, always stable)
        _B_be = dt_s / _den  # excitation coefficient
        _Bk1 = _B_be * k1  # combined super-twisting gain [A^0.5]

        def _be_sigma(c: float) -> float:
            """Solve σ + _Bk1·|σ|^0.5·sign(σ) = c analytically."""
            _disc = _Bk1 * _Bk1 + 4.0 * abs(c)  # always > 0
            _u = 0.5 * (-_Bk1 + float(np.sqrt(_disc)))
            return (_u * _u) if c >= 0.0 else -(_u * _u)

        _c_a = (
            _A_be * self._stsmo_i_alpha
            + _B_be * (self._v_alpha_prev - self._stsmo_z1_alpha)
            - i_alpha
        )  # noqa: E501
        _c_b = (
            _A_be * self._stsmo_i_beta + _B_be * (self._v_beta_prev - self._stsmo_z1_beta) - i_beta
        )  # noqa: E501

        sig_a = _be_sigma(_c_a)
        sig_b = _be_sigma(_c_b)

        # Recover new current estimate from implicit solve
        self._stsmo_i_alpha = i_alpha + sig_a
        self._stsmo_i_beta = i_beta + sig_b

        # Integrate z1 with sign(σ[k+1]) — consistent with backward-Euler
        sign_a = float(np.sign(sig_a))
        sign_b = float(np.sign(sig_b))
        self._stsmo_z1_alpha += dt_s * k2 * sign_a
        self._stsmo_z1_beta += dt_s * k2 * sign_b

        # ── SOGI filtering of z1 ─────────────────────────────────────────────
        # The sign-relay z1 update causes high-frequency chattering near the
        # Nyquist frequency (~fs/2).  Applying a SOGI resonator at ωe_est
        # extracts the fundamental back-EMF component with zero phase lag and
        # attenuates all chattering harmonics.  The SOGI states are shared with
        # the standard reconstruction path — safe because the standard path is
        # never called while STSMO is active.
        # Use the best available omega estimate for SOGI tuning:
        # • During open-loop, the reference speed is exact.
        # • During closed-loop, also floor with motor.omega*Pp so the SOGI stays
        #   tuned as the motor accelerates before the PLL integral builds up.
        #   This prevents the positive-feedback loop: SOGI detuned → emf_mag→0
        #   → omega_est drops further → SOGI detunes more.
        _sogi_omega = abs(self._omega_elec_est)
        _oe_motor_sogi = abs(float(self.motor.omega)) * float(self.motor.params.poles_pairs)
        if self.startup_phase == "open_loop":
            _ol_rpm_s = abs(self.startup_open_loop_speed_rpm)
            _oe_ol_s = _ol_rpm_s / 60.0 * 2.0 * float(np.pi) * float(self.motor.params.poles_pairs)  # noqa: E501
            _sogi_omega = max(_sogi_omega, _oe_ol_s)
        _sogi_omega = max(_sogi_omega, _oe_motor_sogi)
        if _sogi_omega > 5.0:
            _w_s = _sogi_omega
            _ks = self._sogi_k  # damping coefficient (√2 default)
            _ea = self._stsmo_z1_alpha - self._sogi_e_alpha
            self._sogi_e_alpha += dt_s * _w_s * (_ks * _ea - self._sogi_e_alpha_90)
            self._sogi_e_alpha_90 += dt_s * _w_s * self._sogi_e_alpha
            _eb = self._stsmo_z1_beta - self._sogi_e_beta
            self._sogi_e_beta += dt_s * _w_s * (_ks * _eb - self._sogi_e_beta_90)
            self._sogi_e_beta_90 += dt_s * _w_s * self._sogi_e_beta
            _e_alpha_filtered = self._sogi_e_alpha
            _e_beta_filtered = self._sogi_e_beta
        else:
            # Below minimum speed: use z1 directly (no significant chattering)
            _e_alpha_filtered = self._stsmo_z1_alpha
            _e_beta_filtered = self._stsmo_z1_beta

        # EMF estimate — SOGI-filtered z1, clipped to physical bounds
        # Floor e_max with nominal voltage so the clip never zeros the output
        # during the startup convergence phase when omega_elec_est ≈ 0.
        e_max = max(
            float(self.motor.params.back_emf_constant) * abs(self._omega_elec_est) * 2.5 + 0.5,
            float(self.motor.params.nominal_voltage),
        )
        self._stsmo_e_alpha = float(np.clip(_e_alpha_filtered, -e_max, e_max))
        self._stsmo_e_beta = float(np.clip(_e_beta_filtered, -e_max, e_max))

        # Reuse existing speed + angle estimation path
        self._e_alpha_obs = self._stsmo_e_alpha
        self._e_beta_obs = self._stsmo_e_beta

        # Update current memory for consistency with standard path
        self._i_alpha_prev = i_alpha
        self._i_beta_prev = i_beta

        emf_mag = float(np.hypot(self._stsmo_e_alpha, self._stsmo_e_beta))
        self.emf_reconstructed_mag = emf_mag

        # Speed estimate (same as standard path)
        ke = float(self.motor.params.back_emf_constant)
        pp = float(self.motor.params.poles_pairs)
        lambda_pm = ke / max(pp, 1.0)
        Ld = self._emf_recon_L
        if emf_mag > 1e-3:
            cos_th = float(np.cos(self.theta_meas_emf))
            sin_th = float(np.sin(self.theta_meas_emf))
            id_est = float(i_alpha * cos_th + i_beta * sin_th)
        else:
            id_est = 0.0
        lambda_eff = max(lambda_pm + Ld * id_est, 0.3 * lambda_pm)
        # Only update speed estimate when STSMO has produced a meaningful EMF.
        # Guard prevents overwriting the good PLL-derived omega_elec_est with
        # near-zero during the first steps while z1 is converging.
        if emf_mag > 0.3:
            omega_mech = emf_mag / (lambda_eff * max(pp, 1.0))
            self._omega_elec_est = omega_mech * pp

        # MRAS-R update (if enabled)
        if self._use_mras_resistance and emf_mag > 5e-2:
            mras_signal = self._stsmo_e_alpha * i_alpha + self._stsmo_e_beta * i_beta
            self._mras_R -= self._mras_gamma_r * mras_signal * dt_s
            self._mras_R = float(np.clip(self._mras_R, self._mras_R_min, self._mras_R_max))

        return self._stsmo_e_alpha, self._stsmo_e_beta, emf_mag

    # ── Solution 5 ────────────────────────────────────────────────────────────
    def enable_vq_speed_feedforward(self, blend_alpha: float = 0.05) -> None:
        """Enable speed estimation from the vq voltage-balance equation.

        Computes  ωe ≈ (vq − R·iq) / (λeff·Pp)  in parallel with the |E|
        estimate and blends the two, removing the circular id→θ→λeff→ω
        dependency that causes large speed errors in the FW zone.

        Parameters
        ----------
        blend_alpha : float
            Low-pass coefficient for the vq estimate (0 < α ≤ 1).
            Smaller values give smoother but slower response.
        """
        self._use_vq_speed_ff = True
        self._omega_vq_alpha = float(np.clip(blend_alpha, 1e-4, 1.0))

    # ── Solution 6 ────────────────────────────────────────────────────────────
    def enable_mras_resistance(
        self,
        gamma_r: float = 1.0,
        r_min_factor: float = 0.5,
        r_max_factor: float = 4.0,
    ) -> None:
        """Enable online stator resistance estimation via MRAS.

        The MRAS update law  dR̂/dt = −γR·(ê·i)  converges R̂ toward the true
        R, compensating +20–40 % thermal drift in the FW zone.

        Parameters
        ----------
        gamma_r : float
            Adaptation gain [Ω/(V·A·s)].  Larger → faster but noisier.
        r_min_factor, r_max_factor : float
            Clamp R̂ to [r_min_factor × R_nom, r_max_factor × R_nom].
        """
        self._use_mras_resistance = True
        self._mras_gamma_r = float(gamma_r)
        self._mras_R_min = float(r_min_factor) * self._emf_recon_R
        self._mras_R_max = float(r_max_factor) * self._emf_recon_R
        self._mras_R = self._emf_recon_R  # start from nominal

    # ── Solution 7 ────────────────────────────────────────────────────────────
    def enable_adaptive_pll_bandwidth(
        self,
        zeta: float = 0.9,
        omega_n_factor: float = 0.10,
        omega_n_floor_rad: float = 50.0,
        omega_n_ceil_rad: float = 600.0,
    ) -> None:
        """Enable PLL natural frequency that scales with estimated rotor speed.

        ωn(t) = clip(factor × |ω̂e|, floor, ceil)
        kp = 2·ζ·ωn,  ki = ωn²

        This widens the PLL bandwidth proportionally to speed, giving faster
        FW-zone transient tracking while remaining stable at low speed.

        Parameters
        ----------
        zeta : float
            Damping ratio (0.7–1.0 recommended).
        omega_n_factor : float
            Fraction of |ω̂e| used as ωn.  0.10 = 10 %.
        omega_n_floor_rad : float
            Minimum ωn [rad/s] to avoid near-zero gains at startup.
        omega_n_ceil_rad : float
            Maximum ωn [rad/s] to cap bandwidth.
        """
        self._use_adaptive_pll_bw = True
        self._pll_zeta = float(zeta)
        self._pll_omega_n_factor = float(omega_n_factor)
        self._pll_omega_n_floor = float(omega_n_floor_rad)
        self._pll_omega_n_ceil = float(omega_n_ceil_rad)

    # ═══════════════════════════════════════════════════════════════════════════
    # END improvement solutions API
    # ═══════════════════════════════════════════════════════════════════════════

    def enable_sensorless_emf_reconstruction(
        self,
        R: float | None = None,
        L: float | None = None,
        lpf_tau_s: float | None = None,
        use_estimated_speed_ff: bool = True,
    ) -> None:
        """Enable truly sensorless EMF reconstruction for both observers.

        Once enabled, back-EMF is estimated from applied voltages and measured
        phase currents: ``e_αβ = v_αβ − R·i_αβ − L·Δi_αβ/dt``, with a
        first-order LPF.  The observers never read ``motor.back_emf`` or
        ``motor.omega`` when this flag is set.

        Parameters
        ----------
        R : float, optional
            Phase resistance [Ω].  Defaults to motor parameter.
        L : float, optional
            Phase inductance [H].  Defaults to Ld (or phase_inductance).
        lpf_tau_s : float, optional
            LPF time constant [s].  Defaults to 3×τ_e = 3·L/R.
        use_estimated_speed_ff : bool
            Use ``ω_est`` instead of ``motor.omega`` for PLL/SMO feedforward
            and decoupling.  Default True.
        """
        if R is not None:
            self._emf_recon_R = float(R)
        if L is not None:
            self._emf_recon_L = float(L)
        if lpf_tau_s is not None:
            self._emf_recon_lpf_tau = float(lpf_tau_s)
        else:
            tau_e = self._emf_recon_L / max(self._emf_recon_R, 1e-9)
            self._emf_recon_lpf_tau = 3.0 * tau_e
        self._use_estimated_speed_ff = bool(use_estimated_speed_ff)
        self._sensorless_emf_enabled = True

    def calibrate_pll_gains_analytical(
        self,
        rated_rpm: float | None = None,
        zeta: float = 0.9,
        apply: bool = True,
    ) -> dict:
        """Compute PLL gains analytically from motor bandwidth target.

        The PLL is a type-2 phase tracker:
        ``H(s) = (kp·s + ki) / (s² + kp·s + ki)``.
        With natural frequency ωn = √ki and damping ζ = kp/(2·√ki):

        * ``ωn`` is chosen so the PLL bandwidth (≈ωn) is ≤ 1/5 of the
          electrical bandwidth ``ω_e_max = rated_rpm × π/30 × Pp``.
        * ``ki = ωn²``, ``kp = 2·ζ·ωn``

        Parameters
        ----------
        rated_rpm : float, optional
            Rated mechanical speed [RPM].  If None, uses motor rated_speed_rpm
            or falls back to 3000 RPM.
        zeta : float
            Desired damping ratio (default 0.9 for slight overdamping).
        apply : bool
            If True, immediately calls ``set_pll_gains(kp, ki)`` with the result.

        Returns
        -------
        dict with keys ``kp``, ``ki``, ``omega_n_rad_s``.
        """
        if rated_rpm is None:
            rated_rpm = float(getattr(self.motor.params, "rated_speed_rpm", None) or 3000.0)
        pp = float(self.motor.params.poles_pairs)
        omega_e_max = rated_rpm * np.pi / 30.0 * pp  # max electrical rad/s
        # PLL bandwidth = ωn ≤ ω_e_max / 5 to avoid phase lag at rated speed
        omega_n = omega_e_max / 5.0
        ki = omega_n**2
        kp = 2.0 * float(zeta) * omega_n
        if apply:
            self.set_pll_gains(kp, ki)
        return {"kp": kp, "ki": ki, "omega_n_rad_s": omega_n}

    def calibrate_smo_gains_analytical(
        self,
        rated_rpm: float | None = None,
        dt: float | None = None,
        apply: bool = True,
    ) -> dict:
        """Compute SMO gains analytically from motor electrical time constant.

        * ``k_slide = 5 × ω_e_max`` — gain high enough to pull the phase
          trajectory to the sliding surface within a few electrical periods.
        * ``lpf_alpha = dt / (dt + 3·τ_e)`` — LPF matched to electrical
          bandwidth (3× electrical time constant).
        * ``boundary = 0.04`` rad — tanh boundary layer (avoids chattering).

        Parameters
        ----------
        rated_rpm : float, optional
            Rated mechanical speed [RPM].  If None, uses motor parameter.
        dt : float, optional
            Current-loop time step [s].  If None, uses 100 µs.
        apply : bool
            If True, immediately calls ``set_smo_gains(k_slide, lpf_alpha, boundary)``.

        Returns
        -------
        dict with keys ``k_slide``, ``lpf_alpha``, ``boundary``, ``tau_e_s``.
        """
        if rated_rpm is None:
            rated_rpm = float(getattr(self.motor.params, "rated_speed_rpm", None) or 3000.0)
        if dt is None:
            dt = 100e-6  # 100 µs default (10 kHz current loop)
        pp = float(self.motor.params.poles_pairs)
        R = float(self.motor.params.phase_resistance)
        L_val = float(
            self.motor.params.ld
            if self.motor.params.ld is not None
            else self.motor.params.phase_inductance
        )
        tau_e = L_val / max(R, 1e-9)
        omega_e_max = rated_rpm * np.pi / 30.0 * pp
        k_slide = 5.0 * omega_e_max
        # LPF alpha: matches electrical bandwidth (3·τ_e cut-off)
        lpf_alpha = float(dt) / (float(dt) + 3.0 * tau_e)
        lpf_alpha = float(np.clip(lpf_alpha, 1e-4, 0.5))  # safety clamp
        boundary = 0.04  # rad — tanh boundary layer
        if apply:
            self.set_smo_gains(k_slide, lpf_alpha, boundary)
        return {
            "k_slide": k_slide,
            "lpf_alpha": lpf_alpha,
            "boundary": boundary,
            "tau_e_s": tau_e,
        }

    def _reconstruct_emf_sensorless(self, dt: float) -> tuple[float, float, float]:
        """Reconstruct back-EMF from voltage commands and measured currents.

        Formula: ``e_αβ = v_αβ[n-1] − R·i_αβ[n] − L·(i_αβ[n]−i_αβ[n-1])/dt``

        Returns
        -------
        (e_alpha, e_beta, emf_mag) — filtered reconstructed EMF in α-β frame.
        """
        # Phase currents → Clarke/Concordia → α-β
        ia, ib, ic = self.motor.currents
        if self.use_concordia:
            i_alpha, i_beta = concordia_transform(ia, ib, ic)
        else:
            i_alpha, i_beta = clarke_transform(ia, ib, ic)

        dt_safe = max(dt, 1e-12)
        di_alpha = (i_alpha - self._i_alpha_prev) / dt_safe
        di_beta = (i_beta - self._i_beta_prev) / dt_safe

        # Solution 6: use adaptive R if MRAS enabled, else nominal
        R_val = self._mras_R if self._use_mras_resistance else self._emf_recon_R
        L_val = self._emf_recon_L
        # Solution 3: use Lq for di/dt term (EEMF model absorbs saliency)
        Lq_val = self._emf_recon_Lq if self._use_eemf_model else L_val

        # Raw EMF: e = v_prev − R·i − L·di/dt  (Lq used when EEMF enabled)
        e_alpha_raw = self._v_alpha_prev - R_val * i_alpha - Lq_val * di_alpha
        e_beta_raw = self._v_beta_prev - R_val * i_beta - Lq_val * di_beta

        # ── Solution 1: SOGI adaptive filter vs standard LPF ─────────────────
        if self._use_sogi_filter and abs(self._omega_elec_est) > 5.0:
            w = abs(self._omega_elec_est)
            k = self._sogi_k
            if self._sogi_discretization == "tustin":
                # Prewarped bilinear (Tustin) SOGI biquad — Yepes 2011:
                #   H_d(z) = (N1/A)·(1 - z⁻²) / (1 + (B/A)·z⁻¹ + (C/A)·z⁻²)
                # with c=2/Ts, A=c²+kωc+ω², B=−2c²+2ω², C=c²−kωc+ω², N1=kωc.
                # Coefficients are recomputed every step because ω is adaptive.
                c = 2.0 / dt_safe
                w2 = w * w
                kwc = k * w * c
                A = c * c + kwc + w2
                B = -2.0 * c * c + 2.0 * w2
                C = c * c - kwc + w2
                N1 = kwc
                inv_A = 1.0 / A
                # α-axis biquad
                y_a = (
                    N1 * (e_alpha_raw - self._sogi_ua2) - B * self._sogi_ya1 - C * self._sogi_ya2
                ) * inv_A
                self._sogi_ua2 = self._sogi_ua1
                self._sogi_ua1 = e_alpha_raw
                self._sogi_ya2 = self._sogi_ya1
                self._sogi_ya1 = y_a
                self._sogi_e_alpha = y_a
                # β-axis biquad
                y_b = (
                    N1 * (e_beta_raw - self._sogi_ub2) - B * self._sogi_yb1 - C * self._sogi_yb2
                ) * inv_A
                self._sogi_ub2 = self._sogi_ub1
                self._sogi_ub1 = e_beta_raw
                self._sogi_yb2 = self._sogi_yb1
                self._sogi_yb1 = y_b
                self._sogi_e_beta = y_b
            else:
                # Forward-Euler SOGI (legacy fast path):
                #   ê_α[k+1] = ê_α + dt·ωe·(k·(e_raw − ê_α) − ê_α⊥)
                #   ê_α⊥[k+1]= ê_α⊥ + dt·ωe·ê_α
                err_a = e_alpha_raw - self._sogi_e_alpha
                self._sogi_e_alpha += dt_safe * w * (k * err_a - self._sogi_e_alpha_90)
                self._sogi_e_alpha_90 += dt_safe * w * self._sogi_e_alpha
                err_b = e_beta_raw - self._sogi_e_beta
                self._sogi_e_beta += dt_safe * w * (k * err_b - self._sogi_e_beta_90)
                self._sogi_e_beta_90 += dt_safe * w * self._sogi_e_beta
            self._e_alpha_obs = self._sogi_e_alpha
            self._e_beta_obs = self._sogi_e_beta
        else:
            # Standard first-order LPF
            lpf_alpha = dt_safe / (dt_safe + self._emf_recon_lpf_tau)
            self._e_alpha_obs = (1.0 - lpf_alpha) * self._e_alpha_obs + lpf_alpha * e_alpha_raw
            self._e_beta_obs = (1.0 - lpf_alpha) * self._e_beta_obs + lpf_alpha * e_beta_raw

        # Update current memory for next step
        self._i_alpha_prev = i_alpha
        self._i_beta_prev = i_beta

        # Estimate electrical speed from back-EMF magnitude.
        # In field-weakening, negative id reduces the effective flux linkage:
        #   λ_eff = Ke/Pp + Ld·id   →   |E| = ω_e · λ_eff · Pp
        # Using only Ke (no id correction) would under-estimate ω in FW zone,
        # causing the feedforward to lag and inflating the observer angle error.
        emf_mag = float(np.hypot(self._e_alpha_obs, self._e_beta_obs))
        self.emf_reconstructed_mag = emf_mag
        ke = float(
            self.motor.params.back_emf_constant
            if hasattr(self.motor.params, "back_emf_constant")
            and self.motor.params.back_emf_constant
            else getattr(self.motor.params, "torque_constant", 0.028)
        )
        pp = float(self.motor.params.poles_pairs)
        lambda_pm = ke / max(pp, 1.0)  # nominal PM flux linkage [V·s/rad_e]
        ld_raw = getattr(self.motor.params, "ld", None)
        if ld_raw is None:
            ld_raw = getattr(self.motor.params, "phase_inductance", None)
        ld = float(ld_raw) if ld_raw is not None else lambda_pm

        # id estimate: project measured α-β currents onto the d-axis using the
        # last known EMF angle (theta_meas_emf).  Guarded against the startup
        # transient where theta_meas_emf may still be 0.
        if emf_mag > 1e-3:  # only when EMF signal is valid
            cos_th = float(np.cos(self.theta_meas_emf))
            sin_th = float(np.sin(self.theta_meas_emf))
            id_est = float(i_alpha * cos_th + i_beta * sin_th)
        else:
            id_est = 0.0

        # Solution 3: EEMF uses (Lq-Ld)·id correction instead of Ld·id
        if self._use_eemf_model:
            lq_ld = self._emf_recon_Lq - L_val  # Lq − Ld (≥ 0 for IPMSM)
            lambda_eff = lambda_pm + lq_ld * id_est
        else:
            lambda_eff = lambda_pm + ld * id_est
        # Floor so noise in id_est cannot drive λ_eff ≤ 0
        lambda_floor = 0.3 * lambda_pm
        lambda_eff = max(lambda_eff, lambda_floor)

        omega_mech = emf_mag / (lambda_eff * max(pp, 1.0))

        # ── Solution 5: vq-model speed feedforward ────────────────────────────
        # Direct estimate from voltage balance: ωe ≈ (vq − R·iq) / (λeff·Pp)
        # Bypasses the |E|→θ→id→λ circular dependency.
        if self._use_vq_speed_ff and emf_mag > 1e-2:
            cos_th = float(np.cos(self.theta_meas_emf))
            sin_th = float(np.sin(self.theta_meas_emf))
            vq_proj = -self._v_alpha_prev * sin_th + self._v_beta_prev * cos_th
            iq_proj = float(-i_alpha * sin_th + i_beta * cos_th)
            lam_pp = max(lambda_eff * max(pp, 1.0), 1e-6)
            omega_vq_raw = (vq_proj - R_val * iq_proj) / lam_pp
            # Low-pass blend so transients don't destabilise the estimate
            self._omega_vq_est = (
                1.0 - self._omega_vq_alpha
            ) * self._omega_vq_est + self._omega_vq_alpha * omega_vq_raw
            # Weighted blend: 60 % |E| estimate, 40 % vq estimate.
            # omega_vq_raw = (Vq−R·iq)/(λeff·pp) is already in rad/s_mech,
            # so omega_vq_est is also in rad/s_mech — no extra /pp needed.
            omega_mech_vq = abs(self._omega_vq_est)
            omega_mech = 0.6 * omega_mech + 0.4 * omega_mech_vq

        self._omega_elec_est = omega_mech * pp

        # ── Solution 6: MRAS resistance adaptation ────────────────────────────
        # Update law: ΔR = −γR·(ê_α·i_α + ê_β·i_β)·dt
        # Intuition: if R is overestimated, ê contains a bias in the same
        # direction as i → dot product > 0 → R is decreased.
        if self._use_mras_resistance and emf_mag > 5e-2:
            mras_signal = self._e_alpha_obs * i_alpha + self._e_beta_obs * i_beta
            self._mras_R -= self._mras_gamma_r * mras_signal * dt_safe
            self._mras_R = float(np.clip(self._mras_R, self._mras_R_min, self._mras_R_max))

        return self._e_alpha_obs, self._e_beta_obs, emf_mag

    # ────────────────────────────────────────────────────────────────────────

    def set_sensorless_blend(
        self,
        enabled: bool = True,
        min_speed_rpm: float = 250.0,
        min_confidence: float = 0.65,
    ) -> None:
        """Configure low-speed sensorless angle blending behavior."""
        if min_speed_rpm < 0.0:
            raise ValueError("min_speed_rpm must be non-negative")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.sensorless_blend_enabled = bool(enabled)
        self.sensorless_blend_min_speed_rpm = float(min_speed_rpm)
        self.sensorless_blend_min_confidence = float(min_confidence)

    def _estimate_theta_electrical(self, dt: float) -> float:  # noqa: C901  # TODO: extract observer-mode branches (11)
        """Estimate electrical angle from selected observer mode."""
        theta_measured = (self.motor.theta * self.motor.params.poles_pairs) % (2 * np.pi)

        # ── Sensorless or simulation-shortcut EMF source ──────────────────
        if self._sensorless_emf_enabled:
            emf_alpha, emf_beta, emf_mag = self._reconstruct_emf_sensorless(dt)
        else:
            emf_a, emf_b, emf_c = self.motor.back_emf
            if self.use_concordia:
                emf_alpha, emf_beta = concordia_transform(emf_a, emf_b, emf_c)
            else:
                emf_alpha, emf_beta = clarke_transform(emf_a, emf_b, emf_c)
            emf_mag = float(np.hypot(emf_alpha, emf_beta))
        # ─────────────────────────────────────────────────────────────────

        self.emf_observer_mag = emf_mag

        if abs(emf_alpha) + abs(emf_beta) > 1e-12:
            if self._sensorless_emf_enabled:
                # dq reconstruction: e_α = −Ke·ω·sin(θe), e_β = +Ke·ω·cos(θe)
                # → arctan2(−e_α, e_β) = arctan2(sin θe, cos θe) = θe  ✓
                self.theta_meas_emf = float(np.arctan2(-emf_alpha, emf_beta)) % (2 * np.pi)
            else:
                self.theta_meas_emf = float(np.arctan2(emf_beta, emf_alpha)) % (2 * np.pi)

        # Speed magnitude for confidence: use estimated speed when sensorless
        if self._sensorless_emf_enabled and self._use_estimated_speed_ff:
            pp = float(self.motor.params.poles_pairs)
            speed_mag_rpm = abs(self._omega_elec_est) / max(pp, 1.0) * 30.0 / np.pi
        else:
            speed_mag_rpm = abs(self.motor.speed_rpm)
        speed_norm = min(speed_mag_rpm / max(self.startup_min_speed_rpm, 1e-9), 1.0)
        emf_norm = min(emf_mag / max(self.startup_min_emf_v, 1e-9), 1.0)

        if self.observer_target_mode == "PLL":
            target_theta = self.theta_est_pll
        elif self.observer_target_mode == "SMO":
            target_theta = self.theta_est_smo
        else:
            target_theta = theta_measured

        phase_err = abs(_wrap_angle(self.theta_meas_emf - target_theta))
        coherence = 1.0 - min(phase_err / np.pi, 1.0)

        self.observer_confidence_emf = float(np.clip(emf_norm, 0.0, 1.0))
        self.observer_confidence_speed = float(np.clip(speed_norm, 0.0, 1.0))
        self.observer_confidence_coherence = float(np.clip(coherence, 0.0, 1.0))

        self.observer_confidence = float(
            np.clip(
                0.45 * self.observer_confidence_emf
                + 0.35 * self.observer_confidence_speed
                + 0.20 * self.observer_confidence_coherence,
                0.0,
                1.0,
            )
        )

        conf_prev = self.observer_confidence_ema
        conf_alpha = 0.15
        self.observer_confidence_ema = float(
            (1.0 - conf_alpha) * conf_prev + conf_alpha * self.observer_confidence
        )
        self.observer_confidence_trend = float(
            np.clip(self.observer_confidence_ema - conf_prev, -1.0, 1.0)
        )

        conf_above = self.observer_confidence >= self.startup_min_confidence
        if conf_above:
            self.observer_confidence_above_threshold_time_s += dt
        else:
            self.observer_confidence_below_threshold_time_s += dt
        if conf_above and not self.observer_confidence_prev_above:
            self.observer_confidence_crossings_up += 1
        elif (not conf_above) and self.observer_confidence_prev_above:
            self.observer_confidence_crossings_down += 1
        self.observer_confidence_prev_above = conf_above

        self.angle_observer_mode = self._resolve_observer_mode(
            dt, emf_mag, self.observer_confidence
        )

        # Speed feedforward: use estimated ω when sensorless, else true motor.omega
        if self._sensorless_emf_enabled and self._use_estimated_speed_ff:
            omega_ff = self._omega_elec_est
        else:
            omega_ff = self.motor.omega * self.motor.params.poles_pairs

        mode = self.angle_observer_mode
        if mode == "Measured":
            self.theta_error_pll = 0.0
            self.theta_error_smo = 0.0
            self.sensorless_blend_weight = 0.0
            self.theta_sensorless_raw = theta_measured
            return theta_measured

        # Freeze PLL integral during open-loop to prevent windup from noisy
        # low-speed EMF.  The observer tracks angle proportionally during
        # open-loop; the integral is enabled the moment the closed-loop
        # transition fires (pll["integral"] is set to 0 at that transition).
        _open_loop_freeze = self.startup_phase == "open_loop"

        if mode == "PLL":
            err = _wrap_angle(self.theta_meas_emf - self.theta_est_pll)
            self.theta_error_pll = err
            if not _open_loop_freeze:
                self.pll["integral"] += err * dt
            omega_correction = self.pll["kp"] * err + self.pll["ki"] * self.pll["integral"]
            self.theta_est_pll = (self.theta_est_pll + (omega_ff + omega_correction) * dt) % (
                2 * np.pi
            )
            theta_obs = self.theta_est_pll
        else:
            # Sliding-mode-inspired observer variant.
            err = _wrap_angle(self.theta_meas_emf - self.theta_est_smo)
            self.theta_error_smo = err
            boundary = self.smo["boundary"]
            slide = float(np.tanh(err / boundary))
            omega_target = omega_ff + self.smo["k_slide"] * slide
            alpha = self.smo["lpf_alpha"]
            self.smo["omega_est"] = (1.0 - alpha) * self.smo["omega_est"] + alpha * omega_target
            self.theta_est_smo = (self.theta_est_smo + self.smo["omega_est"] * dt) % (2 * np.pi)
            theta_obs = self.theta_est_smo

        self.theta_sensorless_raw = theta_obs
        if not self.sensorless_blend_enabled:
            self.sensorless_blend_weight = 1.0
            return theta_obs

        # Blend weight: use estimated speed when sensorless
        if self._sensorless_emf_enabled and self._use_estimated_speed_ff:
            pp = float(self.motor.params.poles_pairs)
            blend_speed_rpm = abs(self._omega_elec_est) / max(pp, 1.0) * 30.0 / np.pi
        else:
            blend_speed_rpm = abs(self.motor.speed_rpm)
        w_speed = np.clip(
            blend_speed_rpm / max(self.sensorless_blend_min_speed_rpm, 1e-9),
            0.0,
            1.0,
        )
        w_conf = np.clip(
            self.observer_confidence / max(self.sensorless_blend_min_confidence, 1e-9),
            0.0,
            1.0,
        )
        self.sensorless_blend_weight = float(w_speed * w_conf)
        return _blend_angles(theta_measured, theta_obs, self.sensorless_blend_weight)

    def set_cascaded_speed_loop(self, enabled: bool, iq_limit_a: float | None = None) -> None:
        """Enable/disable cascaded speed->iq loop and optionally set iq clamp."""
        self.enable_speed_loop = enabled
        if iq_limit_a is not None:
            self.iq_limit_a = float(abs(iq_limit_a))

    def set_speed_pi_gains(self, kp: float, ki: float, kaw: float = 0.3) -> None:
        """Set speed loop PI gains for cascaded control mode."""
        self.pi_speed["kp"] = float(kp)
        self.pi_speed["ki"] = float(ki)
        self.pi_speed["kaw"] = float(kaw)

    def set_speed_loop_divider(self, divider: int) -> None:
        """Configure multi-rate speed loop decimation.

        Matches real MCU timing where the current loop (Clarke/Park + d/q PI)
        runs every PWM interrupt while the speed PI runs at a lower rate:

        .. code-block:: text

            Real MCU example (Fpwm = 20 kHz, Ts = 50 µs):
              Current PI + observer : every Ts        →  50 µs
              Speed PI              : every 200 × Ts  →  10 ms

        Parameters
        ----------
        divider : int
            Number of current-loop steps between successive speed-PI updates.

            * ``1``   — legacy/high-rate mode: speed PI executes every step
                        (same rate as current PI). Default, backward-compatible.
            * ``200`` — speed PI at 1/200 of current-loop rate (10 ms at 20 kHz).
            * Any positive integer is accepted.

        Notes
        -----
        Between speed-PI firings the last Iq reference is held (zero-order hold),
        exactly as a real interrupt-driven implementation does.
        The effective integration step passed to the speed-PI integrator is
        ``divider × dt``, so d/q PI gains tuned at ``divider=1`` remain valid.
        """
        if divider < 1:
            raise ValueError("speed_loop_divider must be >= 1")
        self.speed_loop_divider = int(divider)
        self._speed_loop_counter = 0

    def set_fw_loop_divider(self, divider: int) -> None:
        """Configure multi-rate field-weakening integrator decimation.

        Sets how many current-loop steps elapse between successive executions
        of the FW headroom integrator.  On a real MCU the FW block runs inside
        the same slow interrupt as the speed PI (both at ~100 Hz), so the
        typical setting matches ``speed_loop_divider``.

        Parameters
        ----------
        divider : int
            * ``1``   — FW integrator updates every current-loop step (default,
                        backward-compatible).
            * ``N``   — FW integrator updates every N current steps; the
                        effective integration ``dt`` passed internally is
                        ``N × current_dt`` so gain values remain unchanged.

        See Also
        --------
        configure_mcu_timing : sets both speed and FW dividers together.
        """
        if divider < 1:
            raise ValueError("fw_loop_divider must be >= 1")
        self.fw_loop_divider = int(divider)
        self._fw_loop_counter = 0

    def configure_mcu_timing(
        self,
        pwm_freq_hz: float,
        speed_loop_hz: float = 100.0,
        fw_loop_hz: float | None = None,
    ) -> dict:
        """Configure all loop dividers to match a real MCU timing scheme.

        This is a single convenience call that replicates the two-rate ISR
        structure used in production single-shunt FOC firmware:

        .. code-block:: text

            ┌──────────────────────────────────────────────────────┐
            │  PWM interrupt  (every T_pwm = 1/Fpwm)              │
            │    • ADC trigger / current reconstruction            │
            │    • Clarke / Park transform                         │
            │    • Angle observer  (SMO or PLL)                   │
            │    • d-axis PI  →  v_d_sat                          │
            │    • q-axis PI  →  v_q_sat                          │
            │    • Voltage saturation (d_priority)                 │
            ├──────────────────────────────────────────────────────┤
            │  Slow ISR  (every T_slow = 1/F_speed = N × T_pwm)  │
            │    • Speed PI  →  Iq_ref  (ZOH between firings)    │
            │    • FW integrator  →  Id_fw  (ZOH between firings) │
            └──────────────────────────────────────────────────────┘

        Parameters
        ----------
        pwm_freq_hz : float
            PWM (current-loop) frequency in Hz.  Corresponds directly to
            ``1 / dt`` used when creating ``BLDCMotor`` / ``SimulationEngine``.
            Example: 20 000 Hz → dt = 50 µs.
        speed_loop_hz : float, optional
            Frequency of the slow ISR (speed PI + FW), default 100 Hz.
            Must satisfy ``speed_loop_hz ≤ pwm_freq_hz``.
        fw_loop_hz : float or None, optional
            Frequency of the FW integrator.  When *None* (default) the FW
            runs at the same rate as the speed PI (recommended — both in the
            same slow ISR).  Pass an explicit value only if a different rate
            is needed.

        Returns
        -------
        dict
            Resolved timing parameters: pwm_freq_hz, speed_loop_hz,
            fw_loop_hz, speed_divider, fw_divider.

        Raises
        ------
        ValueError
            If ``speed_loop_hz > pwm_freq_hz`` or any frequency is non-positive.

        Examples
        --------
        Match a typical STM32 single-shunt FOC setup::

            ctrl.configure_mcu_timing(pwm_freq_hz=20_000, speed_loop_hz=100)
            # → speed_divider = 200, fw_divider = 200

        Higher-bandwidth speed loop (500 Hz)::

            ctrl.configure_mcu_timing(pwm_freq_hz=20_000, speed_loop_hz=500)
            # → speed_divider = 40, fw_divider = 40
        """
        if pwm_freq_hz <= 0:
            raise ValueError("pwm_freq_hz must be positive")
        if speed_loop_hz <= 0 or speed_loop_hz > pwm_freq_hz:
            raise ValueError(f"speed_loop_hz must be in (0, pwm_freq_hz={pwm_freq_hz}]")
        if fw_loop_hz is None:
            fw_loop_hz = speed_loop_hz
        if fw_loop_hz <= 0 or fw_loop_hz > pwm_freq_hz:
            raise ValueError(f"fw_loop_hz must be in (0, pwm_freq_hz={pwm_freq_hz}]")

        speed_divider = max(1, round(pwm_freq_hz / speed_loop_hz))
        fw_divider = max(1, round(pwm_freq_hz / fw_loop_hz))

        self.set_speed_loop_divider(speed_divider)
        self.set_fw_loop_divider(fw_divider)

        return {
            "pwm_freq_hz": pwm_freq_hz,
            "speed_loop_hz": pwm_freq_hz / speed_divider,
            "fw_loop_hz": pwm_freq_hz / fw_divider,
            "speed_divider": speed_divider,
            "fw_divider": fw_divider,
        }

    def set_decoupling(self, enable_d: bool = False, enable_q: bool = False) -> None:
        """Enable/disable optional d/q decoupling feed-forward compensation."""
        self.enable_decouple_d = bool(enable_d)
        self.enable_decouple_q = bool(enable_q)

    def set_current_pi_gains(
        self,
        d_kp: float,
        d_ki: float,
        q_kp: float,
        q_ki: float,
        kaw: float = 0.5,
    ) -> None:
        """Set d/q current PI gains and anti-windup gain."""
        self.pi_d.update({"kp": float(d_kp), "ki": float(d_ki), "kaw": float(kaw)})
        self.pi_q.update({"kp": float(q_kp), "ki": float(q_ki), "kaw": float(kaw)})

    def set_voltage_saturation(
        self,
        mode: str = "proportional",
        coupled_antiwindup_enabled: bool = False,
        coupled_antiwindup_gain: float = 0.15,
    ) -> None:
        """Configure dq voltage saturation strategy and optional coupled anti-windup."""
        normalized = str(mode).strip().lower()
        if normalized not in {"proportional", "d_priority"}:
            raise ValueError("mode must be 'proportional' or 'd_priority'")
        if coupled_antiwindup_gain < 0.0:
            raise ValueError("coupled_antiwindup_gain must be non-negative")

        self.voltage_saturation_mode = normalized
        self.coupled_voltage_antiwindup_enabled = bool(coupled_antiwindup_enabled)
        self.coupled_voltage_antiwindup_gain = float(coupled_antiwindup_gain)

    def set_current_references(self, id_ref: float, iq_ref: float) -> None:
        """Set d/q current references."""
        self.id_ref = id_ref
        self.iq_ref = iq_ref

    def set_current_feedback_mode(self, use_external_feedback: bool) -> None:
        """Select whether update() uses externally provided phase currents."""
        self.use_external_current_feedback = bool(use_external_feedback)

    def set_external_phase_currents(self, currents_abc: np.ndarray) -> None:
        """Set external phase-current feedback (typically reconstructed shunt currents)."""
        currents = np.asarray(currents_abc, dtype=np.float64)
        if currents.shape != (3,):
            raise ValueError("currents_abc must have shape (3,)")
        self._external_phase_currents = currents.copy()

    def set_speed_reference(self, speed_rpm: float) -> None:
        """Set speed reference (rpm). qi reference will be derived from it."""
        self.speed_ref = speed_rpm

    def set_field_weakening(
        self,
        enabled: bool,
        start_speed_rpm: float = 0.0,
        gain: float = 1.0,
        max_negative_id_a: float = 0.0,
        headroom_target_v: float | None = None,
    ) -> None:
        """Configure voltage-headroom-based field-weakening d-axis current injection."""
        if start_speed_rpm < 0.0:
            raise ValueError("start_speed_rpm must be non-negative")
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        if max_negative_id_a < 0.0:
            raise ValueError("max_negative_id_a must be non-negative")
        if headroom_target_v is not None and headroom_target_v < 0.0:
            raise ValueError("headroom_target_v must be non-negative")

        self.field_weakening_enabled = bool(enabled)
        self.field_weakening_start_speed_rpm = float(start_speed_rpm)
        self.field_weakening_gain = float(gain)
        self.field_weakening_max_negative_id_a = float(max_negative_id_a)
        self.field_weakening_headroom_target_v = (
            float(headroom_target_v)
            if headroom_target_v is not None
            else 0.08 * float(self.vdq_limit)
        )
        self.field_weakening_headroom_v = float(self.vdq_limit)
        self.field_weakening_voltage_error_v = 0.0
        self.field_weakening_id_injection_a = 0.0

    def _update_field_weakening_from_voltage(
        self,
        v_d: float,
        v_q: float,
        dt: float,
    ) -> None:
        """Update FW Id injection to maintain configured dq voltage headroom.

        Design (Kim & Sul 1997, Bolognani 1999 — production-drive approach)
        ────────────────────────────────────────────────────────────────────
        The FW headroom is computed from the **saturated** (hard-limited) dq
        voltage command, NOT the unsaturated PI output.

        Why unsaturated PI commands fail
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        During acceleration and load transients the current PI integral winds
        up well beyond the physical voltage limit (``v_q_unsat`` can reach
        50–200 V).  Using the raw unsaturated output tricks the FW integrator
        into injecting maximum negative Id at speeds far below the FW region,
        reducing Kt_eff and creating a limit cycle (motor stuck at ~2720 RPM
        instead of reaching 4000 RPM).

        Why back-EMF estimation also fails
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Estimating |v| from motor speed + FW state gives the steady-state
        back-EMF, which exceeds ``vdq_limit`` any time the actual FW injection
        is slightly less than the equilibrium value.  This causes persistent
        negative headroom → FW over-injection → instability.

        Saturated voltage (this implementation)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        After the d_priority saturation step, |v_d, v_q| ≤ vdq_limit by
        construction.  Therefore:

          headroom = vdq_limit − √(v_d² + v_q²)  ≥ 0  always.

        Key properties:
          • Motor at voltage limit (FW region):  |v_sat| ≈ vdq_limit →
            headroom ≈ 0 → FW integrates at rate ``gain × headroom_target``.
          • Motor below voltage limit (sub-FW):   |v_sat| < vdq_limit →
            headroom > 0 → FW injection decays toward zero (correct).
          • PI windup (large |v_unsat|):  saturation clips to vdq_limit →
            headroom ≈ 0 (conservative), FW integrates slowly rather than
            exploding.  The anti-windup on the current PI simultaneously
            drains the integral, so the windup resolves naturally.

        This is the standard approach in production variable-frequency drives.

        Reference: Kim & Sul, "Torque-Maximizing Field-Weakening Control",
        IEEE Trans. Ind. Electron., vol. 44, pp. 93–103, 1997.
        Bolognani, Zigliotto, Zordan, "Extended-range PMSM drives with
        flux weakening", IEEE Trans. Ind. Appl., 1999.
        """
        v_mag = float(np.hypot(v_d, v_q))
        # v_mag ≤ vdq_limit (caller must pass saturated v_d, v_q)
        self.field_weakening_headroom_v = float(self.vdq_limit - v_mag)

        active = (
            self.field_weakening_enabled
            and self.field_weakening_max_negative_id_a > 0.0
            and self.field_weakening_gain > 0.0
            and abs(self.motor.speed_rpm) > self.field_weakening_start_speed_rpm
        )

        inj_mag = max(-float(self.field_weakening_id_injection_a), 0.0)
        target_headroom = max(float(self.field_weakening_headroom_target_v), 0.0)

        if not active:
            decay_rate = self.field_weakening_gain * max(target_headroom, 1.0)
            inj_mag = max(0.0, inj_mag - decay_rate * dt)
            self.field_weakening_voltage_error_v = 0.0
        else:
            headroom_error_v = target_headroom - self.field_weakening_headroom_v
            self.field_weakening_voltage_error_v = float(headroom_error_v)
            inj_mag += self.field_weakening_gain * headroom_error_v * dt
            inj_mag = float(np.clip(inj_mag, 0.0, self.field_weakening_max_negative_id_a))

        self.field_weakening_id_injection_a = -float(inj_mag)

    def _get_feedback_phase_currents(self) -> tuple[float, float, float]:
        """Return the active three-phase current feedback source."""
        if self.use_external_current_feedback:
            ia, ib, ic = self._external_phase_currents
            return float(ia), float(ib), float(ic)
        ia, ib, ic = self.motor.currents
        return float(ia), float(ib), float(ic)

    def _phase_currents_to_stationary(self, ia: float, ib: float, ic: float) -> tuple[float, float]:
        """Project phase currents into the selected stationary reference frame."""
        if self.use_concordia:
            return concordia_transform(ia, ib, ic)
        return clarke_transform(ia, ib, ic)

    def _get_effective_inductances(self) -> tuple[float, float]:
        """Return effective q- and d-axis inductances for decoupling."""
        lq_eff = (
            float(self.motor.params.lq)
            if self.motor.params.lq is not None
            else float(self.motor.params.phase_inductance)
        )
        ld_eff = (
            float(self.motor.params.ld)
            if self.motor.params.ld is not None
            else float(self.motor.params.phase_inductance)
        )
        return lq_eff, ld_eff

    def _compute_decoupling_feedforward(self, omega_elec: float) -> tuple[float, float]:
        """Return d/q decoupling feedforward voltages from active references."""
        lq_eff, ld_eff = self._get_effective_inductances()
        v_d_ff = (
            -omega_elec * lq_eff * float(self.iq_ref_command) if self.enable_decouple_d else 0.0
        )
        v_q_ff = omega_elec * ld_eff * float(self.id_ref_command) if self.enable_decouple_q else 0.0
        return v_d_ff, v_q_ff

    def _apply_voltage_saturation(
        self,
        v_d_unsat: float,
        v_q_unsat: float,
    ) -> tuple[float, float]:
        """Clamp d/q voltage commands to the configured inverter limit."""
        if self.voltage_saturation_mode == "d_priority":
            v_d = float(np.clip(v_d_unsat, -self.vdq_limit, self.vdq_limit))
            v_q_headroom = float(np.sqrt(max(self.vdq_limit * self.vdq_limit - v_d * v_d, 0.0)))
            v_q = float(np.clip(v_q_unsat, -v_q_headroom, v_q_headroom))
            return v_d, v_q

        v_d = float(v_d_unsat)
        v_q = float(v_q_unsat)
        vdq_mag = np.hypot(v_d, v_q)
        if vdq_mag > self.vdq_limit and vdq_mag > 1e-12:
            scale = self.vdq_limit / vdq_mag
            v_d *= scale
            v_q *= scale
        return v_d, v_q

    def auto_tune_pi(self, axis: str = "q", bandwidth: float = 50.0) -> None:
        """Auto-tune PI parameters for given axis ('d' or 'q').

        This is a placeholder using simple heuristics.  A real implementation
        would perform step responses to estimate Ku and Tu and apply
        Ziegler–Nichols or IMC rules.
        """
        state = self.pi_d if axis == "d" else self.pi_q
        # heuristic: make kp proportional to bandwidth and rotor inertia
        J = self.motor.params.rotor_inertia
        state["kp"] = bandwidth * J * 10.0
        state["ki"] = state["kp"] * 10.0  # arbitrary
        logger.info(f"Auto-tuned {axis}-axis PI: kp={state['kp']:.3f}, ki={state['ki']:.3f}")

    def update(self, dt: float) -> tuple[float, float]:
        """Update controller and return voltage command.

        :param dt: Time step [s]
        :return: Either (magnitude, angle) or (v_alpha, v_beta) depending on
                 `output_cartesian` setting.
        """
        # compute electrical angle from selected observer or startup sequencer
        if self.startup_sequence_enabled:
            self.theta = self._estimate_theta_with_startup_sequence(dt)
        else:
            self.theta = self._estimate_theta_electrical(dt)

        ia, ib, ic = self._get_feedback_phase_currents()
        v_alpha, v_beta = self._phase_currents_to_stationary(ia, ib, ic)

        # park transform to get d/q currents
        id_val, iq_val = park_transform(v_alpha, v_beta, self.theta)

        self.id_ref_command, self.iq_ref_command = self._get_active_references(dt)

        # compute errors
        error_d = self.id_ref_command - id_val
        error_q = self.iq_ref_command - iq_val

        # PI controller outputs Vd and Vq with anti-windup clamps.
        v_d_pi = _pi_update_anti_windup(self.pi_d, error_d, dt, limit=self.vdq_limit)
        v_q_pi = _pi_update_anti_windup(self.pi_q, error_q, dt, limit=self.vdq_limit)

        # Use estimated electrical speed for decoupling when sensorless mode active
        omega_elec = self._get_observer_speed_feedforward()
        # Decoupling feedforward: use *reference* currents, not measured.
        #
        # Using measured id_val / iq_val for feedforward causes large-signal
        # instability: during transient braking (negative Iq), the d-axis
        # feedforward v_d_ff = −ωe·Lq·iq_measured can exceed +vdq_limit,
        # saturating the d-axis completely and driving v_q to zero.  With
        # no q-axis voltage the motor loses all accelerating torque,
        # creating a violent limit-cycle even when the speed loop is gentle.
        #
        # Using the *reference* values (id_ref_command, iq_ref_command)
        # instead is the standard production approach (cf. Holtz, 2002;
        # Briz et al., 2000): references change smoothly and are bounded by
        # iq_limit_a, so the feedforward never exceeds vdq_limit in the
        # linear operating range.  In steady-state the reference ≈ the
        # measured value, so the decoupling quality is unaffected.
        # id_ref_command / iq_ref_command are set just above (line 1177).
        v_d_ff, v_q_ff = self._compute_decoupling_feedforward(omega_elec)
        v_d_unsat = v_d_pi + v_d_ff
        v_q_unsat = v_q_pi + v_q_ff

        # Enforce vector saturation to available voltage circle.
        # NOTE: saturation is applied BEFORE the FW headroom update so that
        # the FW integrator sees the *actual* applied voltage, not the
        # wind-up-inflated unsaturated command (which can be 50-200 V during
        # acceleration transients even though the inverter is limited to
        # vdq_limit).  Using saturated voltages guarantees headroom ≥ 0 and
        # makes the FW response immune to PI windup.  This follows the
        # standard production-drive implementation (e.g. Kim & Sul 1997,
        # Bolognani 1999): the voltage regulator feedback is the actual
        # commanded voltage after hard-limiting, not the desired PI output.
        v_d, v_q = self._apply_voltage_saturation(v_d_unsat, v_q_unsat)

        # FW headroom from SATURATED voltage: |v_sat| ≤ vdq_limit always,
        # so headroom = vdq_limit − |v_sat| ≥ 0 always.
        #
        # Multi-rate: FW integrator fires every fw_loop_divider current steps
        # (same cadence as the speed PI — both belong to the slow ISR on a
        # real MCU).  Between firings the last Id injection is held (ZOH).
        if self.fw_loop_divider <= 1:
            self._update_field_weakening_from_voltage(v_d, v_q, dt)
        else:
            self._fw_loop_counter += 1
            if self._fw_loop_counter >= self.fw_loop_divider:
                self._fw_loop_counter = 0
                self._update_field_weakening_from_voltage(
                    v_d, v_q, dt * float(self.fw_loop_divider)
                )

        if self.coupled_voltage_antiwindup_enabled:
            aw = self.coupled_voltage_antiwindup_gain
            self.pi_d["integral"] += aw * (v_d - v_d_unsat) * dt
            self.pi_q["integral"] += aw * (v_q - v_q_unsat) * dt

        self.last_v_d = v_d
        self.last_v_q = v_q
        self.last_v_d_ff = v_d_ff
        self.last_v_q_ff = v_q_ff

        # inverse park to alpha-beta voltages
        v_alpha_cmd, v_beta_cmd = inverse_park(v_d, v_q, self.theta)

        # Store applied voltage vector for sensorless EMF reconstruction (next step)
        self._v_alpha_prev = v_alpha_cmd
        self._v_beta_prev = v_beta_cmd

        if self.output_cartesian:
            return v_alpha_cmd, v_beta_cmd
        mag = np.hypot(v_alpha_cmd, v_beta_cmd)
        ang = np.arctan2(v_beta_cmd, v_alpha_cmd)
        return mag, ang

    def reset(self) -> None:
        """Reset controller state."""
        self.pi_d["integral"] = 0.0
        self.pi_q["integral"] = 0.0
        self.pi_speed["integral"] = 0.0
        self.id_ref = 0.0
        self.iq_ref = 0.0
        self.speed_ref = 0.0
        self.speed_error = 0.0
        self.theta = 0.0
        self.last_v_d = 0.0
        self.last_v_q = 0.0
        self.last_v_d_ff = 0.0
        self.last_v_q_ff = 0.0
        self.pll["integral"] = 0.0
        self.smo["omega_est"] = 0.0
        self.theta_est_pll = 0.0
        self.theta_est_smo = 0.0
        self.theta_meas_emf = 0.0
        self.theta_error_pll = 0.0
        self.theta_error_smo = 0.0
        self.emf_observer_mag = 0.0
        # Sensorless EMF reconstructor state
        self._e_alpha_obs = 0.0
        self._e_beta_obs = 0.0
        self._i_alpha_prev = 0.0
        self._i_beta_prev = 0.0
        self._omega_elec_est = 0.0
        self.emf_reconstructed_mag = 0.0
        self._v_alpha_prev = 0.0
        self._v_beta_prev = 0.0
        self.startup_elapsed_s = 0.0
        self.startup_confidence_elapsed_s = 0.0
        self.startup_fallback_elapsed_s = 0.0
        self.startup_ready_to_switch = False
        self.startup_transition_done = not self.startup_transition_enabled
        self.startup_fallback_event_count = 0
        self.observer_confidence = 0.0
        self.observer_confidence_emf = 0.0
        self.observer_confidence_speed = 0.0
        self.observer_confidence_coherence = 0.0
        self.observer_confidence_ema = 0.0
        self.observer_confidence_trend = 0.0
        self.observer_confidence_prev_above = False
        self.observer_confidence_above_threshold_time_s = 0.0
        self.observer_confidence_below_threshold_time_s = 0.0
        self.observer_confidence_crossings_up = 0
        self.observer_confidence_crossings_down = 0
        self.startup_handoff_count = 0
        self.startup_last_handoff_time_s = 0.0
        self.startup_last_handoff_confidence = 0.0
        self.startup_handoff_confidence_peak = 0.0
        self.startup_handoff_quality = 0.0
        self.startup_handoff_stability_ratio = 1.0
        self.startup_sequence_elapsed_s = 0.0
        self.startup_phase_elapsed_s = 0.0
        self.startup_open_loop_speed_rpm = self.startup_open_loop_initial_speed_rpm
        self.startup_open_loop_angle = self.startup_align_angle
        self.sensorless_blend_weight = 0.0
        self.theta_sensorless_raw = 0.0
        self.id_ref_command = 0.0
        self.iq_ref_command = 0.0
        self.use_external_current_feedback = False
        self._external_phase_currents = np.zeros(3, dtype=np.float64)
        self.field_weakening_headroom_v = self.vdq_limit
        self.field_weakening_voltage_error_v = 0.0
        self.field_weakening_id_injection_a = 0.0
        self.voltage_saturation_mode = "proportional"
        self.coupled_voltage_antiwindup_enabled = False
        self.coupled_voltage_antiwindup_gain = 0.15
        self.angle_observer_mode = (
            self.startup_initial_mode
            if self.startup_transition_enabled
            else self.observer_target_mode
        )
        self._reset_startup_sequence_runtime()

    def update_applied_voltage(self, va: float, vb: float, vc: float) -> None:
        """Store actual applied phase voltages for sensorless EMF reconstruction.

        Call this *after* ``svm.modulate()`` and *before* the next
        ``ctrl.update()`` so that :meth:`_reconstruct_emf_sensorless` uses the
        true motor terminal voltage rather than the FOC-commanded inverse-Park
        output.  The SVM output can deviate from the FOC command by ±15 %
        (sector-dependent gain) and using the real applied vector eliminates
        that systematic error from the EMF reconstruction.

        Parameters
        ----------
        va, vb, vc:
            Phase-to-neutral (or phase-to-virtual-neutral) voltages in volts,
            as actually applied to the motor terminals by the inverter.
        """
        self._v_alpha_prev, self._v_beta_prev = clarke_transform(va, vb, vc)

    def get_state(self) -> dict:
        """Return controller state variables."""
        return {
            "id_ref": self.id_ref,
            "iq_ref": self.iq_ref,
            "speed_ref": self.speed_ref,
            "speed_error": self.speed_error,
            "speed_loop_enabled": self.enable_speed_loop,
            "use_external_current_feedback": self.use_external_current_feedback,
            "decouple_d_enabled": self.enable_decouple_d,
            "decouple_q_enabled": self.enable_decouple_q,
            "angle_observer_mode": self.angle_observer_mode,
            "theta_electrical": self.theta,
            "theta_meas_emf": self.theta_meas_emf,
            "theta_error_pll": self.theta_error_pll,
            "theta_error_smo": self.theta_error_smo,
            "emf_observer_mag": self.emf_observer_mag,
            "emf_reconstructed_mag": self.emf_reconstructed_mag,
            "omega_elec_est_rad_s": self._omega_elec_est,
            "sensorless_emf_enabled": self._sensorless_emf_enabled,
            "observer_target_mode": self.observer_target_mode,
            "startup_transition_enabled": self.startup_transition_enabled,
            "startup_transition_done": self.startup_transition_done,
            "startup_ready_to_switch": self.startup_ready_to_switch,
            "startup_elapsed_s": self.startup_elapsed_s,
            "startup_confidence_elapsed_s": self.startup_confidence_elapsed_s,
            "startup_fallback_elapsed_s": self.startup_fallback_elapsed_s,
            "startup_initial_mode": self.startup_initial_mode,
            "startup_min_speed_rpm": self.startup_min_speed_rpm,
            "startup_min_elapsed_s": self.startup_min_elapsed_s,
            "startup_min_emf_v": self.startup_min_emf_v,
            "startup_min_confidence": self.startup_min_confidence,
            "startup_confidence_hold_s": self.startup_confidence_hold_s,
            "startup_confidence_hysteresis": self.startup_confidence_hysteresis,
            "startup_fallback_enabled": self.startup_fallback_enabled,
            "startup_fallback_hold_s": self.startup_fallback_hold_s,
            "startup_fallback_event_count": self.startup_fallback_event_count,
            "observer_confidence": self.observer_confidence,
            "observer_confidence_emf": self.observer_confidence_emf,
            "observer_confidence_speed": self.observer_confidence_speed,
            "observer_confidence_coherence": self.observer_confidence_coherence,
            "observer_confidence_ema": self.observer_confidence_ema,
            "observer_confidence_trend": self.observer_confidence_trend,
            "observer_confidence_above_threshold_time_s": self.observer_confidence_above_threshold_time_s,  # noqa: E501
            "observer_confidence_below_threshold_time_s": self.observer_confidence_below_threshold_time_s,  # noqa: E501
            "observer_confidence_crossings_up": self.observer_confidence_crossings_up,
            "observer_confidence_crossings_down": self.observer_confidence_crossings_down,
            "startup_handoff_count": self.startup_handoff_count,
            "startup_last_handoff_time_s": self.startup_last_handoff_time_s,
            "startup_last_handoff_confidence": self.startup_last_handoff_confidence,
            "startup_handoff_confidence_peak": self.startup_handoff_confidence_peak,
            "startup_handoff_quality": self.startup_handoff_quality,
            "startup_handoff_stability_ratio": self.startup_handoff_stability_ratio,
            "startup_sequence_enabled": self.startup_sequence_enabled,
            "startup_phase": self.startup_phase,
            "startup_phase_code": {
                "align": 1.0,
                "open_loop": 2.0,
                "closed_loop": 3.0,
            }.get(self.startup_phase, 0.0),
            "startup_sequence_elapsed_s": self.startup_sequence_elapsed_s,
            "startup_phase_elapsed_s": self.startup_phase_elapsed_s,
            "startup_has_position_sensor": self._has_position_sensor(),
            "startup_align_duration_s": self.startup_align_duration_s,
            "startup_align_current_a": self.startup_align_current_a,
            "startup_align_angle_rad": self.startup_align_angle,
            "startup_open_loop_initial_speed_rpm": self.startup_open_loop_initial_speed_rpm,
            "startup_open_loop_target_speed_rpm": self.startup_open_loop_target_speed_rpm,
            "startup_open_loop_ramp_time_s": self.startup_open_loop_ramp_time_s,
            "startup_open_loop_id_ref_a": self.startup_open_loop_id_ref_a,
            "startup_open_loop_iq_ref_a": self.startup_open_loop_iq_ref_a,
            "startup_open_loop_speed_rpm": self.startup_open_loop_speed_rpm,
            "sensorless_blend_enabled": self.sensorless_blend_enabled,
            "sensorless_blend_min_speed_rpm": self.sensorless_blend_min_speed_rpm,
            "sensorless_blend_min_confidence": self.sensorless_blend_min_confidence,
            "sensorless_blend_weight": self.sensorless_blend_weight,
            "theta_sensorless_raw": self.theta_sensorless_raw,
            "id_ref_command": self.id_ref_command,
            "iq_ref_command": self.iq_ref_command,
            "v_d_cmd": self.last_v_d,
            "v_q_cmd": self.last_v_q,
            "v_d_ff": self.last_v_d_ff,
            "v_q_ff": self.last_v_q_ff,
            "field_weakening_enabled": self.field_weakening_enabled,
            "field_weakening_start_speed_rpm": self.field_weakening_start_speed_rpm,
            "field_weakening_gain": self.field_weakening_gain,
            "field_weakening_max_negative_id_a": self.field_weakening_max_negative_id_a,
            "field_weakening_headroom_target_v": self.field_weakening_headroom_target_v,
            "field_weakening_headroom_v": self.field_weakening_headroom_v,
            "field_weakening_voltage_error_v": self.field_weakening_voltage_error_v,
            "field_weakening_id_injection_a": self.field_weakening_id_injection_a,
            "voltage_saturation_mode": self.voltage_saturation_mode,
            "coupled_voltage_antiwindup_enabled": self.coupled_voltage_antiwindup_enabled,
            "coupled_voltage_antiwindup_gain": self.coupled_voltage_antiwindup_gain,
            "speed_loop_divider": self.speed_loop_divider,
            "speed_loop_counter": self._speed_loop_counter,
            "fw_loop_divider": self.fw_loop_divider,
            "fw_loop_counter": self._fw_loop_counter,
            "pi_d": self.pi_d,
            "pi_q": self.pi_q,
            "speed_pi": self.pi_speed,
            "pll": self.pll,
            "smo": self.smo,
            # ── improvement solutions diagnostics ──────────────────────────
            "sogi_enabled": self._use_sogi_filter,
            "sogi_k": self._sogi_k,
            "active_flux_enabled": self._use_active_flux,
            "active_flux_theta": self._theta_est_af,
            "active_flux_omega": self._omega_est_af,
            "psi_af_mag": self._psi_af_mag,
            "eemf_enabled": self._use_eemf_model,
            "eemf_Lq": self._emf_recon_Lq,
            "stsmo_enabled": self._use_stsmo,
            "vq_ff_enabled": self._use_vq_speed_ff,
            "omega_vq_est_rad_s": self._omega_vq_est,
            "mras_R_enabled": self._use_mras_resistance,
            "mras_R_ohm": self._mras_R,
            "adaptive_pll_bw": self._use_adaptive_pll_bw,
        }
