# Masters-thesis

Source code for the Master's Thesis investigating fitness diversification
mechanisms across continuous mathematical optimization (CEC 2017 benchmark)
and non-linear 3D agent optimization problem (Framsticks simulator).

**Author**: Szymon Szymankiewicz  
**Supervisor**: prof. dr hab. inż. Maciej Komosiński  
**Poznan University of Technology**, 2026

## Implemented Algorithms

* **StdEA** — Standard steady-state EA with tournament selection and uniform
  random deletion.
* **FUSS** — Fitness Uniform Selection Scheme with uniform-target parent
  selection and random deletion.
* **FUDS** — Fitness Uniform Deletion Scheme: tournament selection with
  deletion from the most crowded fitness bin.
* **ConvSel** — Convection Selection with equal-width multi-population migration.
* **HFC** — Hierarchical Fair Competition using Adaptive Setting of Admission
  Thresholds (HFC-ADM) with per-tier admission buffers.
* **Hybrid 1 (`ConvSel + RandWorst`)** — Convection Selection with complete
  random replacement of the lowest-performing subpopulation during migration.
* **Hybrid 2 (`ConvSel + FUSS`)** — Convection Selection with localized FUSS
  parent selection inside each island.

All algorithms follow the steady-state

## Repository Structure

```text
├── cec2017/            # CEC 2017 numerical benchmark (29 functions)
│   ├── config.py                  # Dimensions, budget, parameter grids
│   ├── utility.py                 # Individual, initialization, operators
│   ├── [algorithm].py             # Core EA loops
│   ├── run_*_cec.py               # Runners with signal handling + checkpointing
│   └── submit_*_cec.sh            # SLURM array job scripts
│
└── framsticks/                # Framsticks 3D-agent optimization (f1 encoding)
    ├── utility_frams.py           # FramsticksLib wrapper, genotype operators
    ├── [algorithm]_frams.py       # Steady-state EA loops for Framsticks
    ├── run_*_array.py             # Cluster runners with mid-run checkpointing
    └── submit_*_v2.sh              # SLURM job scripts
