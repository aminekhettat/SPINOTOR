"""
Atomic features tested in this module:
- FOC_SOGI_PARAMS default schema
- configure_sogi_from_dict disabled leaves filter off
- configure_sogi_from_dict enables Euler discretization
- configure_sogi_from_dict enables Tustin discretization
- configure_sogi_from_dict rejects non-dict input
- configure_sogi_from_dict rejects unknown discretization
- enable_sogi_filter rejects unknown discretization string
- enable_sogi_filter resets Tustin biquad memory on (re)entry
- Tustin SOGI biquad updates remain finite at high omega_e
- multi-motor analytical calibration converges with FW off
- multi-motor analytical calibration converges with FW on
- post-calibration short-horizon run remains finite (no NaN/inf)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.control import FOCController
from src.control.adaptive_tuning import AdaptiveFOCTuner
from src.core.motor_model import BLDCMotor, MotorParameters
from src.utils.config import FOC_SOGI_PARAMS


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_spm_motor() -> BLDCMotor:
    """Surface-PM Nanotec-like profile (12 V, 5 pole pairs)."""
    return BLDCMotor(
        MotorParameters(
            nominal_voltage=12.0,
            phase_resistance=0.12,
            phase_inductance=0.00015,
            back_emf_constant=0.028,
            torque_constant=0.028,
            rotor_inertia=5.0e-5,
            friction_coefficient=1.0e-4,
            num_poles=10,
            poles_pairs=5,
        )
    )


def _make_ipm_motor() -> BLDCMotor:
    """Salient IPM synthetic profile (48 V, Lq/Ld = 2)."""
    return BLDCMotor(
        MotorParameters(
            nominal_voltage=48.0,
            phase_resistance=0.08,
            phase_inductance=0.0001,
            back_emf_constant=0.082,
            torque_constant=0.082,
            rotor_inertia=2.0e-4,
            friction_coefficient=5.0e-4,
            num_poles=8,
            poles_pairs=4,
            ld=0.0001,
            lq=0.0002,
            model_type="dq",
        )
    )


def _seed_motor(motor: BLDCMotor, omega_mech: float, theta: float = 0.3) -> None:
    motor.state[0:3] = np.array([0.4, -0.2, -0.2])
    motor.state[3] = omega_mech
    motor.state[4] = theta
    motor._last_emf = motor._calculate_back_emf(motor.theta)


# ── configuration surface ────────────────────────────────────────────────────
def test_sogi_default_config_has_expected_schema():
    assert set(FOC_SOGI_PARAMS) == {"enabled", "k", "discretization"}
    assert FOC_SOGI_PARAMS["enabled"] is False
    assert FOC_SOGI_PARAMS["discretization"] in ("euler", "tustin")
    assert FOC_SOGI_PARAMS["k"] == pytest.approx(math.sqrt(2.0))


def test_configure_sogi_from_dict_disabled_leaves_filter_off():
    ctrl = FOCController(motor=_make_spm_motor())
    ctrl.enable_sogi_filter(k=1.4, discretization="tustin")
    assert ctrl._use_sogi_filter is True

    ctrl.configure_sogi_from_dict({"enabled": False})
    assert ctrl._use_sogi_filter is False


def test_configure_sogi_from_dict_enables_euler_with_custom_k():
    ctrl = FOCController(motor=_make_spm_motor())
    ctrl.configure_sogi_from_dict({"enabled": True, "k": 1.0, "discretization": "euler"})
    assert ctrl._use_sogi_filter is True
    assert ctrl._sogi_discretization == "euler"
    assert ctrl._sogi_k == pytest.approx(1.0)


def test_configure_sogi_from_dict_enables_tustin_and_resets_biquad():
    ctrl = FOCController(motor=_make_spm_motor())
    # Seed stale biquad memory
    ctrl._sogi_ua1 = 1.0
    ctrl._sogi_yb2 = -2.5
    ctrl.configure_sogi_from_dict(
        {"enabled": True, "k": math.sqrt(2.0), "discretization": "tustin"}
    )
    assert ctrl._sogi_discretization == "tustin"
    assert ctrl._sogi_ua1 == 0.0
    assert ctrl._sogi_yb2 == 0.0


def test_configure_sogi_from_dict_rejects_non_dict():
    ctrl = FOCController(motor=_make_spm_motor())
    with pytest.raises(TypeError):
        ctrl.configure_sogi_from_dict("euler")  # type: ignore[arg-type]


def test_configure_sogi_from_dict_rejects_unknown_discretization():
    ctrl = FOCController(motor=_make_spm_motor())
    with pytest.raises(ValueError):
        ctrl.configure_sogi_from_dict({"enabled": True, "discretization": "RK4"})


def test_enable_sogi_filter_rejects_unknown_discretization():
    ctrl = FOCController(motor=_make_spm_motor())
    with pytest.raises(ValueError):
        ctrl.enable_sogi_filter(k=1.4, discretization="rk45")


# ── runtime behaviour ────────────────────────────────────────────────────────
def test_tustin_sogi_emf_stays_finite_in_field_weakening_window():
    """Drive the SOGI biquad at high omega_e for many steps; outputs must
    remain finite (no NaN/inf) and bounded by the raw EMF amplitude scale."""
    motor = _make_spm_motor()
    ctrl = FOCController(motor=motor)
    ctrl.enable_sensorless_emf_reconstruction()
    ctrl.set_angle_observer("PLL")
    ctrl.enable_sogi_filter(k=math.sqrt(2.0), discretization="tustin")

    dt = 1.0e-4
    # ω_e ≈ ω_mech · Pp.  Push to 3000 RPM (≈1571 rad/s electrical) to
    # exercise the FW operating band of the Nanotec profile.
    _seed_motor(motor, omega_mech=314.0)  # 3000 RPM
    ctrl._omega_elec_est = 314.0 * 5.0

    for _ in range(2000):
        ctrl.update(dt)
        assert math.isfinite(ctrl._sogi_e_alpha)
        assert math.isfinite(ctrl._sogi_e_beta)
        assert abs(ctrl._sogi_e_alpha) < 1000.0
        assert abs(ctrl._sogi_e_beta) < 1000.0


def test_euler_and_tustin_sogi_both_drive_observer_to_finite_state():
    """Both discretizations must keep the observer state finite for the
    same operating point."""
    for mode in ("euler", "tustin"):
        motor = _make_spm_motor()
        ctrl = FOCController(motor=motor)
        ctrl.enable_sensorless_emf_reconstruction()
        ctrl.set_angle_observer("PLL")
        ctrl.enable_sogi_filter(k=math.sqrt(2.0), discretization=mode)
        _seed_motor(motor, omega_mech=80.0)
        ctrl._omega_elec_est = 80.0 * 5.0
        for _ in range(500):
            ctrl.update(1.0e-4)
        st = ctrl.get_state()
        assert math.isfinite(st["sogi_k"])
        assert st["sogi_enabled"] is True
        assert math.isfinite(ctrl._omega_elec_est)


# ── multi-motor analytical calibration + stability ───────────────────────────
@pytest.mark.parametrize(
    "motor_factory,rated_rpm",
    [
        (_make_spm_motor, 3500.0),
        (_make_ipm_motor, 3000.0),
    ],
    ids=["spm_nanotec_like", "ipm_salient_like"],
)
def test_analytical_calibration_produces_finite_gains(motor_factory, rated_rpm):
    """AdaptiveFOCTuner must produce finite, positive PI gains for both
    SPM and IPM profiles — basic prerequisite for FW operation."""
    motor = motor_factory()
    tuner = AdaptiveFOCTuner(motor.params)
    tuning, analytical = tuner.tune_analytical()
    for label, kp, ki in (
        ("speed", tuning.speed_kp, tuning.speed_ki),
        ("current", tuning.current_kp, tuning.current_ki),
    ):
        assert math.isfinite(kp), f"{label} Kp not finite"
        assert math.isfinite(ki), f"{label} Ki not finite"
        assert kp > 0.0, f"{label} Kp must be positive"
        assert ki > 0.0, f"{label} Ki must be positive"
    assert tuning.current_margin.phase_margin_deg > 0.0
    assert analytical is not None
    _ = rated_rpm  # explicit unused tag for readability


@pytest.mark.parametrize(
    "fw_enabled",
    [False, True],
    ids=["no_field_weakening", "with_field_weakening"],
)
@pytest.mark.parametrize(
    "motor_factory,rated_rpm,fw_start_rpm",
    [
        (_make_spm_motor, 3500.0, 2000.0),
        (_make_ipm_motor, 3000.0, 1800.0),
    ],
    ids=["spm", "ipm"],
)
def test_post_calibration_run_is_stable_across_motors_and_fw_modes(
    motor_factory, rated_rpm, fw_start_rpm, fw_enabled
):
    """End-to-end smoke test:

    1. Build motor and controller.
    2. Apply analytical PI tuning + STSMO gains from rated speed.
    3. Optionally enable field-weakening.
    4. Step the closed-loop simulator for 200 ms (2000 control steps).
    5. Assert motor and controller states stay finite — no NaN, no inf,
       no runaway above physical bounds.
    """
    motor = motor_factory()
    ctrl = FOCController(motor=motor)

    # Analytical observer & PI calibration.
    ctrl.calibrate_pll_gains_analytical(rated_rpm=rated_rpm)
    ctrl.calibrate_stsmo_gains_analytical(rated_rpm=rated_rpm)
    ctrl.auto_tune_pi(axis="q", bandwidth=120.0)
    ctrl.auto_tune_pi(axis="d", bandwidth=120.0)

    # SOGI from config dict, exercising the new public surface.
    ctrl.configure_sogi_from_dict(
        {"enabled": True, "k": math.sqrt(2.0), "discretization": "tustin"}
    )

    if fw_enabled:
        ctrl.set_field_weakening(
            enabled=True,
            start_speed_rpm=fw_start_rpm,
            gain=1.0,
            max_negative_id_a=5.0,
        )
    else:
        ctrl.set_field_weakening(enabled=False)

    ctrl.set_current_references(id_ref=0.0, iq_ref=1.2)
    _seed_motor(motor, omega_mech=60.0)

    dt = 1.0e-4
    for _ in range(2000):
        motor._last_emf = motor._calculate_back_emf(motor.theta)
        ctrl.update(dt)
        # state finiteness
        for s in motor.state:
            assert math.isfinite(float(s))
        assert math.isfinite(ctrl._omega_elec_est)
        # the rotor should not have flipped to an absurd electrical
        # speed (sanity bound: 10× rated)
        assert abs(motor.state[3]) < (rated_rpm * 2.0 * math.pi / 60.0) * 10.0
