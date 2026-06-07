"""
base_case_explorations.py
=========================
Python translation of BaseCaseMAy2026V1.m (V.R. Voller, University of Minnesota).

Task (Voller email, May 2026)
------------------------------
Reproduce the base CA model in Python and explore:
  Exploration 1 -- Symmetry: vary RNG seed, confirm asymmetry is stochastic
  Exploration 2 -- Saturation threshold Cs: vary Cs, fixed diss_rate
  Exploration 3 -- Dissolution rate: vary diss_rate, fixed Cs
  Exploration 4 -- Quantitative relations: plot 6 metrics vs S = Cs/diss_rate

Model physics (uniform walk)
-----------------------------
Grid: N x N, M=1 (mineral), P=0 (precipitation) everywhere initially.
Particles enter one at a time at (row=mid, col=0), concentration C=0.
At each cell: C += min(diss_rate, M), M -= diss_rate (floored at 0).
Move uniformly to one open (P<1) downstream neighbour: right, right-up, right-down.
If no open neighbour: particle saturates in place (C set to Cs).
When C >= Cs: one extra random step, then P[row,col] = 1.
Controlling group: S = Cs / diss_rate (steps per particle).

Colour convention: navy = dissolved (M<0.999), yellow = precipitated (P=1), teal = fresh.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
import os

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
N        = 201
N_PART   = 4000
DISS_DEF = 0.05
CS_DEF   = 1.0
CMAP     = matplotlib.colors.ListedColormap(["#1a5276", "#f0c030", "#148f77"])

# ── simulation ─────────────────────────────────────────────────────────────────

@njit
def _lcg(state):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF


@njit
def simulate_uniform(N, n_part, diss_rate, Cs, seed):
    """Uniform walk — faithful translation of BaseCaseMAy2026V1.m."""
    M    = np.ones((N, N))
    P    = np.zeros((N, N))
    drow = np.array([0,  1, -1])
    dcol = np.array([1,  1,  1])
    mid  = (N - 1) // 2
    rng  = seed

    for _ in range(n_part):
        row = mid
        col = 0
        C   = 0.0

        while C < Cs:
            if M[row, col] > 0.0:
                dissolved    = min(diss_rate, M[row, col])
                C           += dissolved
                M[row, col] -= dissolved
                if M[row, col] < 0.0:
                    M[row, col] = 0.0
            if C >= Cs:
                break

            n_open   = 0
            open_idx = np.empty(3, dtype=np.int64)
            for k in range(3):
                nr = row + drow[k]
                nc = col + dcol[k]
                if nr < 0 or nr >= N or nc >= N:
                    continue
                if P[nr, nc] >= 1.0:
                    continue
                open_idx[n_open] = k
                n_open          += 1

            if n_open == 0:
                C = Cs
                break

            rng, u  = _lcg(rng)
            chosen  = open_idx[int(u * n_open)]
            row    += drow[chosen]
            col    += dcol[chosen]
            if col >= N - 1:
                C = Cs
                break

        # one extra post-saturation step (randi(1) in MATLAB = always 1)
        n_open   = 0
        open_idx = np.empty(3, dtype=np.int64)
        for k in range(3):
            nr = row + drow[k]
            nc = col + dcol[k]
            if nr < 0 or nr >= N or nc >= N:
                continue
            if P[nr, nc] >= 1.0:
                continue
            open_idx[n_open] = k
            n_open          += 1
        if n_open > 0:
            rng, u  = _lcg(rng)
            chosen  = open_idx[int(u * n_open)]
            row    += drow[chosen]
            col    += dcol[chosen]

        if row < N and col < N:
            P[row, col] = 1.0

    return M, P


def grid_img(M, P, thresh=0.999):
    img = np.full(M.shape, 2, dtype=int)
    img[M < thresh] = 0
    img[P == 1]     = 1
    return img


def show_grid(M, P, ax, title=""):
    ax.imshow(grid_img(M, P), cmap=CMAP, vmin=0, vmax=2,
              origin="lower", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def metrics(M, P, thresh=0.999):
    diss  = int((M < thresh).sum())
    prec  = int((P == 1).sum())
    rows  = np.where((M < thresh).any(axis=1))[0]
    cols  = np.where((M < thresh).any(axis=0))[0]
    xext  = int(cols[-1] - cols[0] + 1) if len(cols) > 0 else 0
    ysp   = int(rows[-1] - rows[0] + 1) if len(rows) > 0 else 0
    return diss, prec, xext, ysp


# JIT warm-up
simulate_uniform(21, 5, DISS_DEF, CS_DEF, 1)


# ── Exploration 1: symmetry ────────────────────────────────────────────────────
print("Exploration 1 — Symmetry")
seeds = [42, 123, 999, 7, 314, 2718]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, seed in zip(axes.flat, seeds):
    M, P = simulate_uniform(N, N_PART, DISS_DEF, CS_DEF, seed)
    show_grid(M, P, ax, f"seed = {seed}")
fig.suptitle(f"Exploration 1 — Symmetry\ndiss_rate={DISS_DEF}, Cs={CS_DEF}, "
             f"N_part={N_PART}", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/exp1_symmetry.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved exp1_symmetry.png")


# ── Exploration 2: Cs sweep ────────────────────────────────────────────────────
print("Exploration 2 — Cs sweep")
Cs_vals = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, Cs in zip(axes.flat, Cs_vals):
    steps = int(Cs / DISS_DEF)
    M, P  = simulate_uniform(N, N_PART, DISS_DEF, Cs, 42)
    show_grid(M, P, ax, f"Cs={Cs} ({steps} steps)")
fig.suptitle(f"Exploration 2 — Saturation threshold Cs\n"
             f"diss_rate={DISS_DEF}, N_part={N_PART}, seed=42", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/exp2_Cs_sweep.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved exp2_Cs_sweep.png")


# ── Exploration 3: diss_rate sweep ────────────────────────────────────────────
print("Exploration 3 — diss_rate sweep")
rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, rate in zip(axes.flat, rates):
    steps = int(CS_DEF / rate)
    M, P  = simulate_uniform(N, N_PART, rate, CS_DEF, 42)
    show_grid(M, P, ax, f"rate={rate} ({steps} steps)")
fig.suptitle(f"Exploration 3 — Dissolution rate\n"
             f"Cs={CS_DEF}, N_part={N_PART}, seed=42", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/exp3_rate_sweep.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved exp3_rate_sweep.png")


# ── Exploration 4: quantitative relations ─────────────────────────────────────
print("Exploration 4 — Quantitative relations")

Cs_sweep   = np.array([0.25,0.5,0.75,1,1.5,2,3,5,7,10,15,20,
                        30,50,75,100,150,200,300,400])
rate_sweep = np.array([0.5,0.2,0.1,0.05,0.03,0.02,0.01,0.005,0.002,0.001])

def run_metrics_sweep(param_vals, mode, seed=42):
    steps_arr, diss_arr, prec_arr, xext_arr, ysp_arr = [], [], [], [], []
    for v in param_vals:
        Cs   = v if mode == "Cs" else CS_DEF
        rate = DISS_DEF if mode == "Cs" else v
        M, P = simulate_uniform(N, N_PART, rate, Cs, seed)
        d, p, x, y = metrics(M, P)
        steps_arr.append(Cs / rate)
        diss_arr.append(d)
        prec_arr.append(p)
        xext_arr.append(x)
        ysp_arr.append(y)
    return (np.array(steps_arr), np.array(diss_arr),
            np.array(prec_arr),  np.array(xext_arr), np.array(ysp_arr))

steps_Cs, diss_Cs, prec_Cs, xext_Cs, ysp_Cs = run_metrics_sweep(Cs_sweep, "Cs")
steps_dr, diss_dr, prec_dr, xext_dr, ysp_dr = run_metrics_sweep(rate_sweep, "rate")

kw1 = dict(marker="o", color="#1a5276", ms=4, lw=1.2, label=r"$C_s$ sweep")
kw2 = dict(marker="s", color="#e74c3c", ms=4, lw=1.2, ls="--", label="rate sweep")

def setup(ax, xl, yl, title):
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

ax = axes[0,0]
ax.plot(steps_Cs, xext_Cs, **kw1)
ax.plot(steps_dr, xext_dr, **kw2)
s_th = np.linspace(1, 400, 200)
ax.plot(s_th, s_th/3, "k:", lw=1, label=r"$S/3$ theory")
setup(ax, "Steps per particle (S)", "x-extent (columns)",
      "Dissolution reach vs steps")
ax.legend(fontsize=7)

ax = axes[0,1]
ax.plot(steps_Cs, ysp_Cs, **kw1)
ax.plot(steps_dr, ysp_dr, **kw2)
ax.plot(s_th, 2*s_th/3, "k:", lw=1, label=r"$2S/3$ theory")
setup(ax, "Steps per particle (S)", "y-spread (rows)", "Dissolution width vs steps")
ax.legend(fontsize=7)

ax = axes[0,2]
ax.plot(steps_Cs, diss_Cs, **kw1)
ax.plot(steps_dr, diss_dr, **kw2)
setup(ax, "Steps per particle (S)", "Dissolved cells (M<0.999)",
      "Total dissolution vs steps")
ax.legend(fontsize=7)

ax = axes[1,0]
ax.plot(steps_Cs, prec_Cs, **kw1)
ax.plot(steps_dr, prec_dr, **kw2)
ax.axhline(N_PART, color="gray", ls=":", lw=1, label=f"max={N_PART}")
setup(ax, "Steps per particle (S)", "Precipitated cells (P=1)",
      "Precipitation vs steps")
ax.legend(fontsize=7)

ax = axes[1,1]
ratio_Cs = prec_Cs / np.maximum(diss_Cs, 1)
ratio_dr = prec_dr / np.maximum(diss_dr, 1)
ax.plot(steps_Cs, ratio_Cs, **kw1)
ax.plot(steps_dr, ratio_dr, **kw2)
ax.axhline(1.0, color="gray", ls=":", lw=1, label="P/D = 1")
setup(ax, "Steps per particle (S)", "Precipitated / Dissolved",
      "P/D ratio vs steps")
ax.legend(fontsize=7)
ax.set_ylim(bottom=0)

ax = axes[1,2]
maxv = max(diss_Cs.max(), diss_dr.max())
sc1  = ax.scatter(diss_Cs, prec_Cs, c=steps_Cs, cmap="Blues",
                  s=60, vmin=0, vmax=400, zorder=3, label=r"$C_s$ sweep")
sc2  = ax.scatter(diss_dr, prec_dr, c=steps_dr, cmap="Reds",
                  s=60, marker="s", vmin=0, vmax=400, zorder=3, label="rate sweep")
ax.plot([0,maxv],[0,maxv],"k--",lw=0.8,label="P=D")
plt.colorbar(sc1, ax=ax, fraction=0.03).set_label("steps", fontsize=7)
ax.set_xlabel("Dissolved cells", fontsize=9)
ax.set_ylabel("Precipitated cells", fontsize=9)
ax.set_title("Precipitated vs dissolved (parametric)", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle(f"Exploration 4 — Quantitative relations\n"
             f"N={N}, N_part={N_PART}, seed=42", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/exp4_relations.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved exp4_relations.png")
print("\nDone.")
