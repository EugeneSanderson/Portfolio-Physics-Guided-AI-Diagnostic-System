# Distillation Column Digital Twin & Fault Diagnostic Suite

A first-principles digital twin for an ethanol/water distillation column with soft
sensors, true-state inference, and neural-network fault detection.

**Author:** Eugene Sanderson

---

## What this does

1. **Simulates a distillation column** from first principles (NRTL thermodynamics,
   tray-by-tray MESH equations, q-line energy balance).

2. **Generates a synthetic dataset** — 100,000+ operating points with injected sensor
   noise and balanced fault conditions (low reflux, low steam, fouled/damaged/missing
   trays).

3. **Trains soft sensors** (MLP neural nets) to reconstruct full temperature and
   composition profiles from noisy plant measurements.

4. **Infers true process state** — a second-stage neural net that corrects sensor drift
   and noise, giving the operator a "clean" view of what the column is actually doing.

5. **Detects and classifies faults** — identifies which fault is present and on which
   tray the physical damage occurred, using only sensor readings.

---

## Pipeline

```
Dataset Generation → RF baseline (operational classifier) → Soft Sensors (4x MLPs)
→ Residual engineering → True State Inference (deep MLP) → Fault Detection (MLP)
→ Diagnostics printout
```

Each stage saves its `.pkl` model to `Models/` and appends its predictions to the
dataset CSV, so the next stage can use them as features.

---

## Installation

```bash
git clone https://github.com/YOUR_USER/distillation-digital-twin.git
cd distillation-digital-twin
pip install -r requirements.txt
```

Tested on Python 3.10–3.13.

---

## Usage

```bash
python Main.py
```

This runs the full pipeline in sequence:

| Step | What happens | Time |
|------|-------------|------|
| Dataset generation | Simulates N_SAMPLES column runs (default 100k) | ~10-30 min |
| Soft sensor training | 4 MLPs for temperature, composition, flows, steam/reflux/pressure | ~5 min |
| Soft sensor inference | Generates soft-sensor profiles for every row | ~1 min |
| True state training | Deep MLP (64→32→16) learns clean state from residuals | ~5 min |
| Fault model training | MLP classifies fault type + locates damaged tray | ~5 min |
| Diagnostics | Prints corrected sensor values vs. raw readings | instant |

To only train from an existing dataset, comment out `generate_dataset()` in `main()`.

To run inference on real plant data, place your CSV in `Data/` (matching the
format of `Plant_Data_Example.csv`) and call:

```python
feed_plant_data()
print_diagnostics()
```

---

## Project structure

```
.
├── Main.py                          # Entry point + pipeline config
├── requirements.txt                 # Python deps
├── Scripts/
│   ├── Thermodynamics.py            # NRTL VLE, Antoine, q-line, MESH solver
│   ├── Column_Model.py              # Tray-by-tray column simulator
│   ├── Dataset_Generator.py         # Synthetic dataset + fault injection
│   ├── Column_RF_Trainer.py         # Random Forest operational classifier
│   ├── Column_Generate_Matrices.py  # Confusion matrices
│   ├── Physics_State_Inference_Trainer_*.py  # Soft sensor trainers (4x)
│   ├── Soft_Sensor_*.py             # Soft sensor inference (4x)
│   ├── Append_Residuals.py          # Residual feature engineering
│   ├── True_State_Trainer.py        # True-state deep MLP
│   ├── True_State_Comparer.py       # Compares noisy vs. true-state predictions
│   ├── NN_Trainer_Fault_Tray_Locator.py  # Fault classification trainer
│   ├── NN_Fault_Detector.py         # Fault detector inference
│   └── Print_Diagnostics.py         # Terminal + plot diagnostics
├── Data/
│   ├── column_diagnostic_dataset.csv  # Generated training dataset
│   └── Plant_Data_Example.csv         # Example real-plant data for inference
├── Models/                            # Trained .pkl files (gitignored, regenerable)
└── Pictures/                          # Output figures (confusion matrices, etc.)
```

---

## How the digital twin works

The column model solves tray-by-tray MESH equations (Material, Equilibrium,
Summation, Heat) using NRTL activity coefficients with literature Antoine constants
for ethanol/water. It's a **first-principles** model — no black-box data fitting in
the core simulation.

Noise is injected at realistic levels (temperature ±0.2 °C, composition ±0.002 mole
fraction, flow ±5%) so the soft sensors learn to filter real measurement noise.

Fault injection spans 5 classes with balanced distribution:
- Normal operation (20%)
- Low reflux (15%)
- Low steam (15%)
- Fouled tray — 30-60% efficiency loss (20%)
- Damaged tray — 80-95% efficiency loss (15%)
- Missing tray — near-zero efficiency (15%)

---

## License

MIT — see [LICENSE](LICENSE) for details.
