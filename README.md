# synthesenavion

Longitudinal flight mechanics simulation of a commercial airliner (Airbus A321-200), built as part of the **Aircraft Synthesis** coursework at [ENAC](https://www.enac.fr) (École Nationale de l'Aviation Civile), IENAC program.

The base simulation framework (`atmosphere.py`, `dynamic.py`, `units.py`, `display_utils.py`) was provided by the **CADO team** (Conceptual Airplane Design & Operations, Aircraft & Systems / Sysdyn Optim departments, ENAC). The aerodynamic model, trim/stability analyses and disturbance-response studies on top of it are original work for this project.

## What it does

The project models the longitudinal aerodynamics and flight dynamics of an airliner and uses that model to study:

- **Static stability** — effect of the static margin on the pitching-moment curve `Cm = f(α)` and on the trim deflection required to balance the aircraft at a given angle of attack.
- **Trim computation** — solving for the level-flight equilibrium (throttle, angle of attack, horizontal stabilizer setting) at a given altitude and airspeed.
- **Linearization** — numerical computation of the state-space Jacobian (A, B matrices) around a trim point.
- **Dynamic response** — simulation of the aircraft's response to a perturbation (e.g. a gust or a control input) over time, with automatic plotting of angle of attack, pitch rate, airspeed, flight path angle and altitude.

## Project structure

| File | Description |
|---|---|
| `aero_model.py` | Aerodynamic model (wing + horizontal tail plane), lift/drag/moment coefficients, static margin, `Airbus_A321_200` aircraft definition |
| `atmosphere.py` | ISA atmosphere model (pressure, temperature, density, speed of sound, Reynolds number) |
| `dynamic.py` | Equations of motion, trim solver, numerical linearization (Jacobian), simulation plotting |
| `units.py` | Unit conversion utilities |
| `display_utils.py` | Plotting helpers (figure setup, axis decoration) |
| `programme.py` | Study of the static margin's effect on `Cm(α)` and on the required trim deflection |
| `reponse_dynamique_a_une_perturbation.py` | Time-domain simulation of the aircraft's response to a perturbation |
| `membre3.py` | Additional analysis script |

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

Each analysis script can be run directly, e.g.:

```bash
python programme.py
```

This will compute the aircraft's aerodynamic coefficients for several static margin values and plot the resulting pitching-moment and trim-deflection curves.

## Background

The aircraft model represents an **Airbus A321-200** (MTOM 93,500 kg, cruise Mach 0.78 at FL350) using a simplified analytical wing + horizontal-tail-plane aerodynamic model, with optional stall, buffeting and wave-drag effects.

## Author

Camille Da Silva — ENAC, IENAC25, Telecommunications & Satellite Systems track
