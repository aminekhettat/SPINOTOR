# SPINOTOR

> **Free, accessible BLDC/PMSM motor-control simulator with sensorless FOC and V/f control — designed for engineers, researchers, and users with visual disabilities.**

**Version 0.12.0** · Python 3.12 · PySide6 · [Documentation](docs/index.rst) · [Quickstart](QUICKSTART.md) · [Roadmap](Roadmap/ROADMAP.md)

---

## What Is SPINOTOR?

SPINOTOR is a desktop simulation platform for three-phase **BLDC and PMSM** motor drives. It lets you design, tune, and validate motor-control strategies — including full **sensorless operation** — entirely in software before touching real hardware.

The GUI runs on Windows, Linux, and macOS. It is built for keyboard-only navigation and screen reader compatibility, making it one of the only motor-control simulators designed with **visual accessibility as a first-class requirement**.

---

## Why This Project?

When this project was started, no existing free tool filled all of these gaps at once:

| Gap                                                                      | How SPINOTOR fills it                                                                   |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| No free BLDC/PMSM simulator with **sensorless control**                  | Five observer modes: PLL, SMO, STSMO, Active Flux, and sensored reference               |
| No simulator combining **FOC** (vector) and **V/f** (scalar) in one tool | Both control strategies selectable from the same GUI                                    |
| Existing tools are commercial or closed-source                           | Fully open, no license fee                                                              |
| Motor-control tools are inaccessible to visually impaired engineers      | Screen reader support (NVDA, JAWS, Orca), keyboard-only workflow, spoken status updates |

---

## Key Features

### Control Strategies

- **Field-Oriented Control (FOC)** — decoupled d/q current loops, PI auto-tuning, field weakening, cascaded speed loop
- **Voltage/Frequency Control (V/f)** — scalar open-loop speed control with configurable startup ramp
- **Space Vector Modulation (SVM)** — with optional PWM non-ideality effects

### Sensorless Angle Observers

All modes are selectable from the **Observer & Startup** tab with context-sensitive parameter widgets:

| Observer       | Principle                                                | Best for                                                   |
| -------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| **Measured**   | True simulation angle (reference)                        | Controller tuning and debug                                |
| **PLL**        | Back-EMF phase-locked loop                               | Simple bring-up, SPM motors                                |
| **SMO**        | First-order sliding-mode                                 | Disturbance-robust operation                               |
| **STSMO**      | Super-Twisting (2nd order), backward-Euler + SOGI filter | **SPM motors (Ld≈Lq)** — chattering-free, wide speed range |
| **ActiveFlux** | Active flux vector ψa = ψs − Ld·is (Boldea 2009)         | IPM motors, field-weakening                                |

→ Full mathematical derivations and tuning guide: [`docs/sensorless_observers.rst`](docs/sensorless_observers.rst)

### Motor & Drive Modeling

- Configurable motor parameters (R, Ld, Lq, Ke, Kt, poles, topology)
- Star (wye) and delta winding topologies
- Constant, ramp, and custom load profiles
- Realistic inverter model: device drop, dead-time, conduction/switching losses, DC-link ripple, thermal coupling, phase asymmetry
- 1-, 2-, and 3-shunt current reconstruction with injectable amplifier errors

### Analysis & Calibration

- Real-time monitoring dashboard (speed, currents, torque, power, efficiency)
- FFT spectrum analyzer with magnitude/phase, dB scaling, and CSV/image export
- Auto-calibration pipeline (analytic + physics-based, single-click)
- Regression baseline framework with KPI drift detection
- CSV data export with JSON metadata

### Accessibility

- PySide6 GUI (LGPL) with full accessible names and descriptions on every widget
- Keyboard shortcuts for all key actions (F5 start, F6 stop, F7 reset, Ctrl+S export)
- Spoken status updates and narrated workflow events
- Compatible with NVDA, JAWS, and Orca screen readers

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the GUI
python main.py

# 3. Run tests
pytest -q
```

For a step-by-step first simulation (5 minutes): see [QUICKSTART.md](QUICKSTART.md).

---

## GUI Tab Layout

| Tab                    | Contents                                                    |
| ---------------------- | ----------------------------------------------------------- |
| **Motor & Drive**      | Motor parameters · Load profile · Supply profile            |
| **Control**            | FOC / V/f mode · PI tuning · Inverter realism · Timing      |
| **Observer & Startup** | Observer selection · Observer parameters · Startup sequence |
| **Advanced Settings**  | Current sensing · MCU budget · Hardware backend             |
| **Analysis**           | Monitoring · Plotting · Calibration                         |

---

## Sensorless Observers — Quick Reference

The recommended bring-up sequence for sensorless operation:

1. Start with **Measured** — verify d/q and speed loops
2. Switch to **PLL** — tune Kp (50–150), then Ki (1000–5000)
3. Switch to **SMO** for better disturbance rejection — tune Kslide, LPF Alpha, Boundary
4. Switch to **STSMO** for chattering-free wide-range operation — click **Auto-Calibrate**
5. Switch to **ActiveFlux** for IPM motors or field-weakening studies

---

## Project Layout

```
SPINOTOR/
├── main.py                   ← entry point
├── src/
│   ├── control/              ← FOC, V/f, SVM, observers
│   ├── core/                 ← motor model, simulation engine, power model
│   ├── hardware/             ← DAQ abstraction, current sensing
│   ├── ui/                   ← PySide6 GUI and accessible widgets
│   ├── utils/                ← config, logging, speech, regression baseline
│   └── visualization/        ← plotting and FFT
├── tests/                    ← pytest suite (100% src/ coverage gate)
├── docs/                     ← Sphinx documentation
├── examples/                 ← headless usage scripts and calibration examples
│   └── validate_all_observers.py  ← 5-observer validation (all pass in < 30 s)
├── data/motor_profiles/      ← JSON motor profiles (nanotec SPM, IPM salient 48 V)
├── references/               ← control-theory formulas and bibliography
└── Roadmap/                  ← planned and completed milestones
```

---

## Documentation

| Document                                                       | Purpose                                  |
| -------------------------------------------------------------- | ---------------------------------------- |
| [QUICKSTART.md](QUICKSTART.md)                                 | First simulation in 5 minutes            |
| [docs/features.rst](docs/features.rst)                         | Full feature inventory                   |
| [docs/sensorless_observers.rst](docs/sensorless_observers.rst) | Observer math, API, and tuning           |
| [docs/advanced.rst](docs/advanced.rst)                         | FOC, FW, SVM, calibration deep dives     |
| [CONTRIBUTING.md](CONTRIBUTING.md)                             | Contribution policy and quality gates    |
| [Roadmap/ROADMAP.md](Roadmap/ROADMAP.md)                       | Past milestones and planned work         |
| [references/](references/)                                     | Control-theory formulas and bibliography |

---

## Quality Gates

Every commit runs:

- **Ruff** — lint
- **Mypy** — type checking on `src/`
- **Bandit** — security scan
- **pytest** — full test suite with 100% `src/` coverage threshold
- **Sphinx** — HTML documentation build
- **pip-audit** — dependency vulnerability audit

CI enforces the same gates on every push and pull request via GitHub Actions.

To run gates locally:

```bash
pip install -r requirements-dev.txt
python -m pre_commit run --all-files
```

---

## Versioning

```bash
bump2version patch   # x.y.Z
bump2version minor   # x.Y.0
bump2version major   # X.0.0
```

A pre-commit hook auto-increments the patch version before each commit (bypass with `SKIP_AUTO_BUMP=1`).

---

## License & Safety

- **License:** Custom restricted license — no redistribution (see `LICENSE`)
- Provided as-is for research, calibration, and educational use
- Users are responsible for validation before applying any settings to real hardware

---

## Author

**Amine Khettat** · [amine.khettat@blindsystems.org](mailto:amine.khettat@blindsystems.org)
