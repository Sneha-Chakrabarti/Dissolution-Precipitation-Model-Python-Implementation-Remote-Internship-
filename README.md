# CA Dissolution-Precipitation Model
## Python Implementation and Computational Extensions

**Sneha Chakrabarti** · IISER Kolkata  
Supervised by **Prof. Vaughan Voller** (University of Minnesota) and **Prof. Piotr Szymczak** (University of Warsaw)  
June 2026

---

## Overview

Python implementation of a cellular automaton model for coupled mineral dissolution and precipitation, along with parameter explorations, the weighted walk update, and independent pore geometry analysis.

---

## Repository Structure

```
├── base_case_explorations.py    # Task 1: Python translation + four explorations
├── weighted_walk_comparison.py  # Task 2: weighted walk update + comparison
├── chord_analysis_pooled.py     # Extension: global chord-length analysis
├── channel_width_profile.py     # Extension: column-by-column width profiling
├── figures/                     # All output figures (PNG)
├── data/                        # Numerical results (CSV)
└── task1_report.tex             # LaTeX report for Task 1
```

---

## Model Physics

The CA operates on an N x N grid. Each cell holds mineral concentration M in [0,1] and precipitation indicator P in {0,1}. Particles enter at (x=0, y=(N-1)/2) with C=0 and dissolve mineral as they walk downstream. When C reaches Cs the particle deposits.

**Controlling group:** S = Cs / diss\_rate (steps per particle). Pattern morphology depends only on S.

**Weighted walk (Voller email, May 2026):** probability of moving to neighbour i is proportional to w\_i = 2 - M\_i, favouring the most dissolved downstream cell.

**Colour convention:** navy = dissolved, yellow = precipitated, teal = fresh mineral.

**Fixed base parameters:** N = 201, N\_part = 4000, diss\_rate = 0.05, Cs = 1.0, seed = 42.

---

## Scripts

### `base_case_explorations.py` -- Task 1

Python translation of BaseCaseMAy2026V1.m (Voller, MATLAB original). Implements the uniform walk and runs four explorations:

| Exploration | Parameter varied | Key finding |
|-------------|-----------------|-------------|
| 1 -- Symmetry | seed | Asymmetry is stochastic, not structural |
| 2 -- Cs sweep | Cs in [0.25, 1000] | Pattern scale proportional to Cs; branching at S > 400 |
| 3 -- Rate sweep | rate in [0.001, 5] | Only S = Cs/rate governs the pattern |
| 4 -- Relations | Both sweeps | All metrics collapse on S; match random-walk theory |

**Output figures:** `exp1_symmetry.png`, `exp2_Cs_sweep.png`, `exp3_rate_sweep.png`, `exp4_relations.png`

---

### `weighted_walk_comparison.py` -- Task 2

Implements Voller's May 2026 weighted walk update and compares original vs weighted across base case, Cs sweep, rate sweep, and quantitative metrics.

**Key results:**
- Weighted walk transitions to domain-filling patterns at ~10--12 steps vs ~16 for uniform walk
- Weighted model produces narrower, more channelled dissolution body
- Difference is morphological: dissolution and precipitation totals converge at large S

**Output figures:** `cmp_base.png`, `cmp_Cs.png`, `cmp_rate.png`, `cmp_quant.png`

---

### `chord_analysis_pooled.py` -- Independent Extension

Shoots random horizontal and vertical chords through the dissolved pore space at each S value, pooled over 6 seeds. Fits exponential P(l) = (1/L) exp(-l/L) and power law P(l) proportional to l^(-alpha); compares via AIC.

**Key results:**
- Power-law chord statistics preferred at small S (sparse channels)
- Exponential statistics preferred at intermediate S
- Anisotropy Lx/Ly rises to ~2 in the saturated regime, consistent with walk geometry

**Output figures:** `chord_L_vs_S.png`, `chord_aic.png`, `chord_histograms.png`  
**Output data:** `data/chord_L_vs_S.csv`

---

### `channel_width_profile.py` -- Independent Extension

Measures the dissolved body width column by column, averaged over 6 seeds. Gives a geometrically meaningful L(S) that does not degenerate when the domain fills.

**Key results:**
- Sharp spanning transition at S* ~ 13 (260 steps): x-extent jumps to full domain width
- For S >= S*: channel width L ~ 55--60 cells, approximately independent of S
- Width growth exponent beta ~ 0.33 (fill method), near the fully-correlated limit

**Output figures:** `channel_width_profiles.png`, `channel_L_vs_S.png`, `channel_beta.png`  
**Output data:** `data/channel_width.csv`

---

## Running the Code

All scripts are self-contained. Dependencies: `numpy`, `matplotlib`, `scipy`, `numba`.

```bash
pip install numpy matplotlib scipy numba

python base_case_explorations.py
python weighted_walk_comparison.py
python chord_analysis_pooled.py
python channel_width_profile.py
```

Figures are saved to `figures/`. Data to `data/`. Numba JIT-compiles on first run; subsequent runs are faster.

---

## Implementation Notes

- Numba JIT-compiled simulation: 4000 particles on a 201x201 grid runs in ~6s per call.
- RNG: 32-bit LCG inside Numba for full reproducibility.
- Weighted walk: w\_i = 2 - M\_i, normalised over open downstream neighbours only. Cells with P=1 excluded.
- Post-saturation step: exactly 1 extra step (randi(1) in MATLAB always returns 1).
- Pore threshold: M < 0.999.

---

## References

Voller, V.R. (2025). *Cellular automaton model of dissolution-precipitation in porous media.*  
Szymczak, P. & Ladd, A.J.C. (2004). Microscopic simulations of fracture dissolution. *Geophys. Res. Lett.*
