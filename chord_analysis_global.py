"""
chord_analysis.py
=================
Phase 01 of the research project:
    "Reactive Dissolution as a Porous Medium: Active Transport Through
     an Evolving Geometry"  —  Sneha Chakrabarti, 2026

What this script does
---------------------
1. Simulates the Voller (2025) CA with the weighted walk update
   (Voller email, May 2026: w_i = 2 - M_i, probabilities normalised
   over open downstream neighbours only).

2. Extracts the binary pore grid from each simulation (dissolved cells =
   M < M_thresh).

3. Shoots random horizontal and vertical chords through the pore space
   and records chord lengths in each direction separately.

4. Fits P(l) = (1/L) * exp(-l/L) to the chord-length histogram via
   log-linear least squares (robust to sparse tails).

5. Sweeps over S = Cs / diss_rate (the single controlling group) across
   two decades, collecting L_x(S), L_y(S), and the anisotropy ratio.

6. Fits a power law L ~ S^beta on a log-log plot and reports beta.

7. Saves all figures and a CSV of results.

Faithful translation of BaseCaseMAy2026V1.m
--------------------------------------------
- N x N grid, particles injected at (row=1, col=(N-1)/2) in MATLAB
  1-indexing = (row=0, col=(N-1)//2) in Python 0-indexing.
- icr in MATLAB: relative [row; col] increments for the 3 downstream
  neighbours: (row+1, col+1), (row+1, col), (row+1, col-1)  →  right,
  right-up, right-down in the x-direction of the lattice.
- Weighted walk (Voller email): w_i = 2 - M_i for each open neighbour i;
  probability p_i = w_i / sum(w).
- Extra post-saturation step: randi(1) in MATLAB returns 1 always
  (randi(N) returns integer in [1,N]), so exactly 1 extra step is taken.
- Blocking: neighbours with P == 1 are excluded from the candidate list.
- Boundary: particles stop at col == N-1 (right wall).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
import os, time, csv

# ── output directory ──────────────────────────────────────────────────────────
OUTDIR = "/mnt/user-data/outputs/chord_analysis"
os.makedirs(OUTDIR, exist_ok=True)

# ── fixed simulation parameters ───────────────────────────────────────────────
N        = 201          # grid size (must be odd so midpoint is integer)
N_PART   = 4000         # number of particles (Voller default)
SEED     = 42           # RNG seed
M_THRESH = 0.999        # cells with M < M_THRESH are "dissolved" (pore space)
N_CHORDS = 100_000      # random chords per direction per simulation

# ── S sweep: log-spaced from S=2 to S=2000 ───────────────────────────────────
# S = Cs / diss_rate.  We fix diss_rate and vary Cs.
DISS_RATE = 0.05
S_VALUES  = np.unique(np.round(
    np.logspace(np.log10(2), np.log10(2000), 30)
).astype(int)).astype(float)
CS_VALUES = S_VALUES * DISS_RATE   # corresponding Cs values


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CA SIMULATION  (Numba JIT, faithful to Voller MATLAB)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _lcg(state):
    """32-bit LCG random number generator (Numba-safe, no numpy RNG inside jit)."""
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF   # (new_state, float in [0,1))


@njit
def simulate(N, n_part, diss_rate, Cs, seed):
    """
    Run the Voller CA with weighted walk.

    Parameters
    ----------
    N         : int   — grid side length
    n_part    : int   — number of particles
    diss_rate : float — mineral dissolved per step (Voller: 0.05)
    Cs        : float — saturation threshold (Voller: 1.0)
    seed      : int   — RNG seed

    Returns
    -------
    M : (N, N) float64 — mineral content after all particles
    P : (N, N) float64 — precipitation field (0 or 1)
    """
    M = np.ones((N, N))
    P = np.zeros((N, N))

    # Downstream neighbour offsets: (drow, dcol)
    # Matches Voller's icr matrix: right, right-up, right-down
    # In our orientation: col advances → x, row advances → y
    # icr row 1 (col increment): all +1
    # icr row 2 (row increment): 0, +1, -1
    drow = np.array([0,  1, -1])
    dcol = np.array([1,  1,  1])

    mid = (N - 1) // 2   # injection row (y midpoint)
    rng = seed

    for _ in range(n_part):
        row = mid
        col = 0         # left wall, column index 0
        C   = 0.0       # particle precipitant concentration

        # ── dissolution walk ──────────────────────────────────────────────────
        while C < Cs:
            # dissolve mineral at current cell
            if M[row, col] > 0.0:
                dissolved = min(diss_rate, M[row, col])
                C        += dissolved
                M[row, col] -= dissolved
                if M[row, col] < 0.0:
                    M[row, col] = 0.0

            if C >= Cs:
                break

            # collect open downstream neighbours
            n_open = 0
            open_idx   = np.empty(3, dtype=np.int64)
            open_w     = np.empty(3, dtype=np.float64)
            w_sum      = 0.0

            for k in range(3):
                nr = row + drow[k]
                nc = col + dcol[k]
                # boundary checks: stay inside grid, right wall stops particle
                if nr < 0 or nr >= N or nc >= N:
                    continue
                if P[nr, nc] >= 1.0:   # blocked by precipitation
                    continue
                w = 2.0 - M[nr, nc]   # Voller weighted walk: w_i = 2 - M_i
                open_idx[n_open] = k
                open_w[n_open]   = w
                w_sum           += w
                n_open          += 1

            if n_open == 0:
                # no open neighbours → saturate in place
                C = Cs
                break

            # weighted random choice among open neighbours
            rng, u = _lcg(rng)
            u *= w_sum
            cumw = 0.0
            chosen = open_idx[0]
            for i in range(n_open):
                cumw += open_w[i]
                if u <= cumw:
                    chosen = open_idx[i]
                    break

            row += drow[chosen]
            col += dcol[chosen]

            # stop at right wall
            if col >= N - 1:
                C = Cs
                break

        # ── one extra step after saturation (randi(1) in MATLAB = always 1) ──
        for _ in range(1):
            n_open = 0
            open_idx = np.empty(3, dtype=np.int64)
            for k in range(3):
                nr = row + drow[k]
                nc = col + dcol[k]
                if nr < 0 or nr >= N or nc >= N:
                    continue
                if P[nr, nc] >= 1.0:
                    continue
                open_idx[n_open] = k
                n_open += 1
            if n_open > 0:
                rng, u = _lcg(rng)
                chosen = open_idx[int(u * n_open)]
                row += drow[chosen]
                col += dcol[chosen]

        # ── deposit ───────────────────────────────────────────────────────────
        if row < N and col < N:
            P[row, col] = 1.0

    return M, P


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CHORD-LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def chord_lengths_1d(pore_row):
    """
    Given a 1D boolean array (True = pore), return lengths of all pore runs.
    A 'chord' is a maximal contiguous run of True values.
    """
    lengths = []
    in_pore = False
    count   = 0
    for val in pore_row:
        if val:
            in_pore = True
            count  += 1
        else:
            if in_pore:
                lengths.append(count)
            in_pore = False
            count   = 0
    if in_pore and count > 0:
        lengths.append(count)
    return lengths


def fit_exponential(lengths):
    """
    Fit P(l) = (1/L) exp(-l/L) to chord-length data via log-linear regression.

    Returns
    -------
    L    : float — characteristic pore length (cells)
    r2   : float — R² of the log-linear fit
    bins : array — bin centres used
    hist : array — normalised histogram values
    """
    if len(lengths) < 10:
        return np.nan, np.nan, np.array([]), np.array([])

    lengths = np.array(lengths, dtype=float)
    max_l   = int(lengths.max()) + 1
    bins    = np.arange(0.5, max_l + 0.5, 1.0)
    hist, edges = np.histogram(lengths, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # keep only bins with positive counts for log fit
    mask = hist > 0
    if mask.sum() < 3:
        return np.nan, np.nan, centres, hist

    log_hist = np.log(hist[mask])
    x        = centres[mask]

    # log P(l) = -log(L) - l/L  →  linear: intercept = -log(L), slope = -1/L
    coeffs   = np.polyfit(x, log_hist, 1)
    slope    = coeffs[0]
    if slope >= 0:
        return np.nan, np.nan, centres, hist

    L   = -1.0 / slope
    # R²
    predicted = np.polyval(coeffs, x)
    ss_res    = np.sum((log_hist - predicted) ** 2)
    ss_tot    = np.sum((log_hist - log_hist.mean()) ** 2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return L, r2, centres, hist


def analyse_pore_grid(M, thresh=M_THRESH, n_chords=N_CHORDS, rng=None):
    """
    Extract chord lengths from the dissolved pore space in both x and y.

    Strategy: shoot random axis-aligned lines through the grid and collect
    pore-run lengths along each line.  We sample rows/columns with replacement
    (weighted by grid size) up to n_chords total per direction.

    Returns
    -------
    Lx, r2x : characteristic length and fit quality along x (column direction)
    Ly, r2y : characteristic length and fit quality along y (row direction)
    lengths_x, lengths_y : raw chord arrays
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pore = M < thresh   # True where dissolved

    # ── horizontal chords (scan each row along columns) ──────────────────────
    lengths_x = []
    rows_sample = rng.integers(0, N, size=n_chords)
    for r in rows_sample:
        lengths_x.extend(chord_lengths_1d(pore[r, :]))

    # ── vertical chords (scan each column along rows) ─────────────────────────
    lengths_y = []
    cols_sample = rng.integers(0, N, size=n_chords)
    for c in cols_sample:
        lengths_y.extend(chord_lengths_1d(pore[:, c]))

    Lx, r2x, _, _ = fit_exponential(lengths_x)
    Ly, r2y, _, _ = fit_exponential(lengths_y)

    return Lx, r2x, Ly, r2y, lengths_x, lengths_y


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

CMAP = matplotlib.colors.ListedColormap(
    ["#1a5276",   # navy  — dissolved (M reduced)
     "#f0c030",   # yellow — precipitated (P=1)
     "#148f77"])  # teal  — fresh mineral


def grid_image(M, P):
    """
    Encode grid as integer array for the custom colourmap:
      0 = dissolved (M < 0.999), 1 = precipitated (P == 1), 2 = fresh.
    """
    img = np.full(M.shape, 2, dtype=int)   # default: fresh
    img[M < 0.999] = 0                     # dissolved
    img[P == 1]    = 1                     # precipitated (overwrites dissolved)
    return img


def plot_grid(M, P, title="", ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid_image(M, P), cmap=CMAP, vmin=0, vmax=2,
              origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if show:
        plt.tight_layout()
        plt.show()


def plot_chord_fit(lengths, L, r2, direction, S, ax):
    """Plot the chord-length histogram and exponential fit on ax."""
    if len(lengths) == 0 or np.isnan(L):
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        return
    lengths = np.array(lengths, dtype=float)
    max_l   = min(int(lengths.max()), 60)
    bins    = np.arange(0.5, max_l + 1.5, 1.0)
    hist, edges = np.histogram(lengths, bins=bins, density=True)
    centres     = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, hist, width=0.8, color="#5dade2", alpha=0.6,
           label="data")
    x_fit = np.linspace(0.5, max_l, 200)
    ax.plot(x_fit, (1/L) * np.exp(-x_fit / L), "r-", lw=1.5,
            label=f"fit L={L:.1f}, R²={r2:.3f}")
    ax.set_xlabel("chord length (cells)", fontsize=8)
    ax.set_ylabel("P(l)", fontsize=8)
    ax.set_title(f"{direction}, S={S:.0f}", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep():
    print(f"{'S':>8}  {'Cs':>6}  {'Lx':>7}  {'r2x':>6}  {'Ly':>7}  {'r2y':>6}  {'Lx/Ly':>7}  {'t(s)':>6}")
    print("-" * 65)

    results = []
    rng_chord = np.random.default_rng(99)

    # warm up Numba JIT on a tiny grid (avoids counting compile time)
    _ = simulate(21, 10, DISS_RATE, 0.1, 1)

    for S, Cs in zip(S_VALUES, CS_VALUES):
        t0 = time.time()

        M, P = simulate(N, N_PART, DISS_RATE, float(Cs), SEED)
        Lx, r2x, Ly, r2y, lx, ly = analyse_pore_grid(M, rng=rng_chord)
        elapsed = time.time() - t0

        ratio = Lx / Ly if (not np.isnan(Lx) and not np.isnan(Ly) and Ly > 0) else np.nan
        print(f"{S:8.1f}  {Cs:6.3f}  {Lx:7.2f}  {r2x:6.3f}  {Ly:7.2f}  {r2y:6.3f}  {ratio:7.3f}  {elapsed:6.1f}")

        results.append(dict(S=S, Cs=Cs, Lx=Lx, r2x=r2x, Ly=Ly, r2y=r2y,
                            ratio=ratio, n_pore=int((M < M_THRESH).sum())))

    return results


def power_law(x, a, beta):
    return a * x ** beta


def fit_power_law(S_arr, L_arr):
    """Fit L = a * S^beta on valid (non-nan) points."""
    mask = np.isfinite(L_arr) & (L_arr > 0) & (S_arr > 0)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    try:
        popt, pcov = curve_fit(power_law, S_arr[mask], L_arr[mask],
                               p0=[1.0, 0.5], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return popt[0], popt[1], perr[0], perr[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_figures(results):
    S_arr   = np.array([r["S"]    for r in results])
    Lx_arr  = np.array([r["Lx"]   for r in results])
    Ly_arr  = np.array([r["Ly"]   for r in results])
    r2x_arr = np.array([r["r2x"]  for r in results])
    r2y_arr = np.array([r["r2y"]  for r in results])
    rat_arr = np.array([r["ratio"] for r in results])

    # ── Figure 1: L(S) power-law plot ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(S_arr, Lx_arr, "o-", color="#1a5276", ms=5, lw=1.2, label=r"$L_x$ (along dissolution)")
    ax.loglog(S_arr, Ly_arr, "s--", color="#148f77", ms=5, lw=1.2, label=r"$L_y$ (transverse)")

    # power law fits
    ax_fit, bx, _, bx_err = fit_power_law(S_arr, Lx_arr)
    ay_fit, by, _, by_err = fit_power_law(S_arr, Ly_arr)
    S_fine = np.logspace(np.log10(S_arr[0]), np.log10(S_arr[-1]), 200)
    if not np.isnan(bx):
        ax.loglog(S_fine, power_law(S_fine, ax_fit, bx), ":",
                  color="#1a5276", lw=1.5,
                  label=fr"fit $\beta_x={bx:.3f}\pm{bx_err:.3f}$")
    if not np.isnan(by):
        ax.loglog(S_fine, power_law(S_fine, ay_fit, by), ":",
                  color="#148f77", lw=1.5,
                  label=fr"fit $\beta_y={by:.3f}\pm{by_err:.3f}$")

    ax.set_xlabel(r"$S = C_s / \mathrm{rate}$  (steps per particle)", fontsize=11)
    ax.set_ylabel(r"Characteristic pore length $L$ (cells)", fontsize=11)
    ax.set_title(r"$L(S)$ — power-law scaling", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)

    # ── Figure 1b: anisotropy ratio ────────────────────────────────────────────
    ax2 = axes[1]
    mask = np.isfinite(rat_arr)
    ax2.semilogx(S_arr[mask], rat_arr[mask], "^-", color="#922b21",
                 ms=5, lw=1.2)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, label=r"$L_x/L_y=1$ (isotropic)")
    ax2.set_xlabel(r"$S = C_s / \mathrm{rate}$", fontsize=11)
    ax2.set_ylabel(r"Anisotropy ratio $L_x / L_y$", fontsize=11)
    ax2.set_title("Dissolution-induced anisotropy", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax2.tick_params(labelsize=9)

    fig.suptitle(
        f"Chord-length analysis of Voller CA dissolution patterns\n"
        f"N={N}, N_part={N_PART}, seed={SEED}, diss_rate={DISS_RATE}",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_L_vs_S.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_L_vs_S.png")
    if not np.isnan(bx):
        print(f"  Power-law fit: Lx ~ S^{bx:.3f} +/- {bx_err:.3f}")
    if not np.isnan(by):
        print(f"  Power-law fit: Ly ~ S^{by:.3f} +/- {by_err:.3f}")

    # ── Figure 2: R² quality of exponential fits ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(S_arr, r2x_arr, "o-", color="#1a5276", ms=5, lw=1.2, label=r"$R^2_x$")
    ax.semilogx(S_arr, r2y_arr, "s--", color="#148f77", ms=5, lw=1.2, label=r"$R^2_y$")
    ax.axhline(0.95, color="gray", ls=":", lw=0.8, label="R²=0.95")
    ax.set_xlabel(r"$S = C_s / \mathrm{rate}$", fontsize=11)
    ax.set_ylabel(r"$R^2$ of exponential fit", fontsize=11)
    ax.set_title("Goodness of fit — is P(l) truly exponential?", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_r2_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_r2_quality.png")

    # ── Figure 3: example chord-length histograms at 4 representative S ───────
    S_examples = [10, 40, 200, 1000]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    rng_ex = np.random.default_rng(7)

    for col_i, S_ex in enumerate(S_examples):
        Cs_ex = S_ex * DISS_RATE
        M_ex, P_ex = simulate(N, N_PART, DISS_RATE, Cs_ex, SEED)
        Lx_ex, r2x_ex, Ly_ex, r2y_ex, lx_ex, ly_ex = analyse_pore_grid(
            M_ex, n_chords=50_000, rng=rng_ex)

        # top row: grid image
        axes[0, col_i].imshow(grid_image(M_ex, P_ex), cmap=CMAP,
                              vmin=0, vmax=2, origin="lower",
                              interpolation="nearest")
        axes[0, col_i].set_title(f"S = {S_ex}", fontsize=9)
        axes[0, col_i].axis("off")

        # bottom row: chord-length histogram (x direction)
        plot_chord_fit(lx_ex, Lx_ex, r2x_ex, "x-chord", S_ex,
                       axes[1, col_i])

    axes[0, 0].set_ylabel("dissolution grid", fontsize=8)
    axes[1, 0].set_ylabel("P(l)  [x-direction]", fontsize=8)
    fig.suptitle(
        "CA dissolution grids and chord-length distributions at four S values\n"
        f"diss_rate={DISS_RATE}, N_part={N_PART}, seed={SEED}",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_chord_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_chord_examples.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(results):
    path = f"{OUTDIR}/L_vs_S.csv"
    fields = ["S", "Cs", "Lx", "r2x", "Ly", "r2y", "ratio", "n_pore"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Phase 01 — Chord-length analysis of Voller CA dissolution patterns")
    print(f"N={N}, N_part={N_PART}, diss_rate={DISS_RATE}, seed={SEED}")
    print(f"S range: {S_VALUES[0]:.1f} to {S_VALUES[-1]:.1f}  ({len(S_VALUES)} values)")
    print("=" * 65)

    results = run_sweep()

    print("\nGenerating figures ...")
    make_figures(results)
    save_csv(results)

    print("\nDone. Output in:", OUTDIR)
