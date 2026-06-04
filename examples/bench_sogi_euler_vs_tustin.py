"""SOGI discretization benchmark — Euler vs Tustin, two motors, pre-FW & FW.

For each motor profile (Nanotec SPM 12V, IPM Salient 48V) the same PLL
observer is exercised in three configurations:

* LPF             — legacy first-order EMF low-pass (baseline)
* SOGI(Euler)     — forward-Euler discretization (current SPINOTOR default)
* SOGI(Tustin)    — prewarped bilinear biquad (new in v0.10.2)

The speed profile climbs from 0 to a pre-FW plateau, then to a field-weakening
plateau.  Two analysis windows are reported:

* pre_fw : last 0.5 s before the FW ramp begins (steady, |id| ≈ 0)
* fw_ss  : last 1.0 s of the FW plateau (steady, id < 0)

Metrics per window: θ̂ vs θ_true RMS / peak (electrical degrees), |E| relative
error, observer confidence.  Results are written to
``sim_results_sogi_tustin.json`` and a Markdown comparison table is printed.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control import FOCController, SVMGenerator  # noqa: E402
from src.core import BLDCMotor, ConstantLoad, MotorParameters, SimulationEngine  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
DT = 1e-4  # 100 µs control + sim step (same as run_observer_validation.py)
T_SIM = 4.0
N_STEPS = int(T_SIM / DT)
T_VEC = np.arange(N_STEPS, dtype=np.float64) * DT

OUT_PATH = ROOT / "sim_results_sogi_tustin.json"


def speed_profile(pre_fw_rpm: float, fw_rpm: float) -> np.ndarray:
    """0→pre_fw (0–1 s), hold (1–2 s), ramp pre_fw→fw (2–3 s), hold (3–4 s)."""
    return np.where(
        T_VEC < 1.0,
        T_VEC * pre_fw_rpm,
        np.where(
            T_VEC < 2.0,
            pre_fw_rpm,
            np.where(
                T_VEC < 3.0,
                pre_fw_rpm + (T_VEC - 2.0) * (fw_rpm - pre_fw_rpm),
                fw_rpm,
            ),
        ),
    )


def load_profile(path: Path) -> dict[str, Any]:
    prof = json.loads(path.read_text())
    mp = prof["motor_params"]
    # Profiles use either nested "rated_info" or flat top-level rated_*.
    if "rated_info" in prof:
        ri = prof["rated_info"]
        rated_rpm = float(ri["rated_speed_rpm"])
        rated_i = float(ri["rated_current_a"])
    else:
        rated_rpm = float(prof["rated_speed_rpm"])
        rated_i = float(prof["rated_current_a"])
    return {
        "name": prof.get("profile_name", path.stem),
        "mp": mp,
        "rated_rpm": rated_rpm,
        "rated_i": rated_i,
    }


def build_sim(
    prof: dict[str, Any], sogi_mode: str, fw_id_max: float, fw_start_rpm: float, fw_gain: float
):
    mp = prof["mp"]
    R = float(mp["phase_resistance"])
    L_val = float(mp["phase_inductance"])
    Ke = float(mp["back_emf_constant"])
    V_NOM = float(mp["nominal_voltage"])
    Pp = int(mp.get("poles_pairs", int(mp["num_poles"]) // 2))

    params = MotorParameters(
        nominal_voltage=V_NOM,
        phase_resistance=R,
        phase_inductance=L_val,
        back_emf_constant=Ke,
        torque_constant=float(mp["torque_constant"]),
        rotor_inertia=float(mp["rotor_inertia"]),
        friction_coefficient=float(mp["friction_coefficient"]),
        num_poles=int(mp.get("num_poles", 2 * Pp)),
        poles_pairs=Pp,
        ld=float(mp["ld"]),
        lq=float(mp["lq"]),
        model_type=mp["model_type"],
        emf_shape=mp["emf_shape"],
        flux_weakening_id_coefficient=float(mp.get("flux_weakening_id_coefficient", 0.0)),
        flux_weakening_min_ratio=float(mp.get("flux_weakening_min_ratio", 0.2)),
    )
    motor = BLDCMotor(params, dt=DT)
    load = ConstantLoad(torque=0.0)
    engine = SimulationEngine(motor, load, dt=DT, max_history=200)
    svm = SVMGenerator(dc_voltage=V_NOM)
    ctrl = FOCController(motor=motor, enable_speed_loop=True)

    V_LIM = V_NOM / math.sqrt(3.0)

    # Current PI: analytical tuning, identical to the validation script.
    i_kp = R * 1.5
    i_ki = R / L_val * 0.3
    ctrl.set_current_pi_gains(d_kp=i_kp, d_ki=i_ki, q_kp=i_kp, q_ki=i_ki)
    ctrl.configure_mcu_timing(pwm_freq_hz=1.0 / DT, speed_loop_hz=100.0)
    ctrl.set_field_weakening(
        enabled=True,
        start_speed_rpm=fw_start_rpm,
        gain=fw_gain,
        max_negative_id_a=fw_id_max,
        headroom_target_v=0.92 * V_LIM,
    )

    # PLL observer, analytically tuned for this motor.
    omega_e_max = prof["rated_rpm"] * math.pi / 30.0 * Pp
    omega_n = omega_e_max / 10.0
    zeta = 0.9
    ctrl.set_pll_gains(kp=2.0 * zeta * omega_n, ki=omega_n**2)
    ctrl.observer_target_mode = "PLL"
    ctrl.angle_observer_mode = "PLL"

    # Sensorless EMF reconstruction (LPF baseline first, then optional SOGI).
    ctrl.enable_sensorless_emf_reconstruction(
        R=R,
        L=L_val,
        lpf_tau_s=DT,
        use_estimated_speed_ff=True,
    )
    ctrl.vdq_limit = V_LIM
    ctrl.set_speed_pi_gains(kp=0.045, ki=0.0025)

    if sogi_mode in ("euler", "tustin"):
        ctrl.enable_sogi_filter(k=math.sqrt(2.0), discretization=sogi_mode)

    # Standard sensorless startup: align → open-loop ramp → observer.
    ctrl.set_startup_sequence(
        enabled=True,
        align_duration_s=0.08,
        align_current_a=2.0,
        align_angle_deg=0.0,
        open_loop_initial_speed_rpm=30.0,
        open_loop_target_speed_rpm=800.0,
        open_loop_ramp_time_s=0.50,
        open_loop_id_ref_a=0.0,
        open_loop_iq_ref_a=0.50,
    )
    ctrl.set_startup_transition(
        enabled=True,
        initial_mode="Measured",
        min_speed_rpm=700.0,
        min_elapsed_s=0.15,
        min_emf_v=1.0,
        min_confidence=0.60,
        confidence_hold_s=0.04,
        fallback_enabled=True,
        fallback_hold_s=0.08,
    )
    return motor, engine, svm, ctrl, Pp, Ke, float(mp["ld"])


def wrap(arr: np.ndarray) -> np.ndarray:
    return (arr + math.pi) % (2 * math.pi) - math.pi


def run_one(
    prof: dict[str, Any], sogi_mode: str, pre_fw_rpm: float, fw_rpm: float
) -> dict[str, Any]:
    fw_id_max = 0.25 * prof["rated_i"]
    fw_start = pre_fw_rpm * 0.9
    fw_gain = 0.05  # mild integrator gain — sufficient for SS comparison

    motor, engine, svm, ctrl, Pp, Ke, ld = build_sim(
        prof,
        sogi_mode,
        fw_id_max,
        fw_start,
        fw_gain,
    )
    ref = speed_profile(pre_fw_rpm, fw_rpm)

    theta_true = np.empty(N_STEPS, np.float32)
    theta_obs = np.empty(N_STEPS, np.float32)
    speed_t = np.empty(N_STEPS, np.float32)
    emf_t = np.empty(N_STEPS, np.float32)
    emf_o = np.empty(N_STEPS, np.float32)
    conf = np.empty(N_STEPS, np.float32)

    lambda_pm = Ke / Pp
    t0 = time.perf_counter()
    for i in range(N_STEPS):
        ctrl.set_speed_reference(float(ref[i]))
        mag, ang = ctrl.update(DT)
        vph = svm.modulate(mag, ang)
        ctrl.update_applied_voltage(float(vph[0]), float(vph[1]), float(vph[2]))
        engine.step(vph, log_data=False)

        th_e = (motor.theta * Pp) % (2 * math.pi)
        theta_true[i] = th_e
        theta_obs[i] = ctrl.theta_est_pll
        speed_t[i] = motor.speed_rpm

        ia, ib, ic = motor.currents
        ialpha = ia
        ibeta = (ib - ic) / math.sqrt(3.0)
        cs = math.cos(th_e)
        sn = math.sin(th_e)
        idq_d = ialpha * cs + ibeta * sn
        idq_q = -ialpha * sn + ibeta * cs
        we = abs(motor.omega) * Pp
        leff = max(lambda_pm + ld * idq_d, 0.3 * lambda_pm)
        emf_t[i] = we * math.sqrt((ld * idq_q) ** 2 + leff**2)

        st = ctrl.get_state()
        emf_o[i] = float(st.get("emf_reconstructed_mag", 0.0))
        conf[i] = float(st.get("observer_confidence", 0.0))

    elapsed = time.perf_counter() - t0

    err_deg = np.degrees(wrap(theta_obs - theta_true))

    pre_lo, pre_hi = int(1.5 / DT), int(2.0 / DT)
    fw_lo, fw_hi = int(3.0 / DT), int(4.0 / DT)

    def metrics(lo: int, hi: int) -> dict[str, float]:
        e = err_deg[lo:hi]
        emf_rel = np.mean(np.abs(emf_o[lo:hi] - emf_t[lo:hi]) / np.maximum(emf_t[lo:hi], 1e-3))
        return {
            "angle_rms_deg": float(np.sqrt(np.mean(e * e))),
            "angle_peak_deg": float(np.max(np.abs(e))),
            "speed_err_rpm": float(np.mean(speed_t[lo:hi] - ref[lo:hi])),
            "speed_mean_rpm": float(np.mean(speed_t[lo:hi])),
            "emf_rel": float(emf_rel),
            "conf_mean": float(np.mean(conf[lo:hi])),
        }

    return {
        "sogi_mode": sogi_mode,
        "pre_fw_rpm": pre_fw_rpm,
        "fw_rpm": fw_rpm,
        "wallclock_s": elapsed,
        "pre_fw": metrics(pre_lo, pre_hi),
        "fw_ss": metrics(fw_lo, fw_hi),
    }


def main() -> None:
    motor_specs = [
        {
            "path": ROOT / "data/motor_profiles/nanotec_db57m012_12v.json",
            "pre_fw_rpm": 1500.0,
            "fw_rpm": 3000.0,
            "tag": "Nanotec_SPM_12V",
        },
        {
            "path": ROOT / "data/motor_profiles/ipm_salient_48v.json",
            "pre_fw_rpm": 2000.0,
            "fw_rpm": 4000.0,
            "tag": "IPM_Salient_48V",
        },
    ]
    schemes = ["lpf", "euler", "tustin"]

    all_results: dict[str, Any] = {"dt_s": DT, "t_sim_s": T_SIM, "motors": {}}
    for spec in motor_specs:
        prof = load_profile(spec["path"])
        print(f"\n=== {spec['tag']}  ({prof['name']}) ===")
        print(f"    pre-FW: {spec['pre_fw_rpm']:.0f} RPM   FW: {spec['fw_rpm']:.0f} RPM")
        per_motor: dict[str, Any] = {
            "profile": prof["name"],
            "pre_fw_rpm": spec["pre_fw_rpm"],
            "fw_rpm": spec["fw_rpm"],
        }
        for sogi_mode in schemes:
            print(f"  [{sogi_mode:>7}] running...", end="", flush=True)
            res = run_one(prof, sogi_mode, spec["pre_fw_rpm"], spec["fw_rpm"])
            per_motor[sogi_mode] = res
            print(
                f"  done in {res['wallclock_s']:.1f}s   "
                f"pre-FW theta_RMS={res['pre_fw']['angle_rms_deg']:.2f}deg   "
                f"FW theta_RMS={res['fw_ss']['angle_rms_deg']:.2f}deg"
            )
        all_results["motors"][spec["tag"]] = per_motor

    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {OUT_PATH.relative_to(ROOT)}")

    # Markdown summary
    print("\n## Summary -- theta_RMS (electrical degrees)")
    print("| Motor | Scheme | pre-FW theta_RMS | FW theta_RMS | pre-FW |E|_rel | FW |E|_rel |")
    print("|---|---|---:|---:|---:|---:|")
    for tag, pm in all_results["motors"].items():
        for s in schemes:
            r = pm[s]
            print(
                f"| {tag} | {s} | "
                f"{r['pre_fw']['angle_rms_deg']:.2f}deg | "
                f"{r['fw_ss']['angle_rms_deg']:.2f}deg | "
                f"{r['pre_fw']['emf_rel'] * 100:.1f}% | "
                f"{r['fw_ss']['emf_rel'] * 100:.1f}% |"
            )


if __name__ == "__main__":
    main()
