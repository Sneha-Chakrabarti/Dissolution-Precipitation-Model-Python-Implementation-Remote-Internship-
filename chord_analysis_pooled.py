"""
chord_analysis_v2.py
====================
Phase 01 — second pass.

Three improvements over v1
---------------------------
1. Multi-seed pooling (6 seeds per S value) — fixes noisy small-S regime
   where a single 201x201 grid has too few dissolved cells to fit a
   chord-length distribution reliably.

2. Exponential vs power-law discrimination near the branching transition.
   At each S we fit both:
       exponential:  P(l) = (1/L) exp(-l/L)          [log-linear fit]
       power law:    P(l) = A * l^(-alpha)             [log-log fit]
   and compare via AIC (Akaike Information Criterion).  AIC penalises
   model complexity equally, so the winner is genuinely better supported
   by the data, not just more flexible.

3. Beta theory comparison.
   Naive random-walk prediction: pore width ~ sqrt(2S/3)  => beta = 0.5
   Measured:                     beta_x ~ 0.40
   We plot both on the same axes and annotate the discrepancy.

Output
------
fig1_L_vs_S_pooled.png      — L(S) with error bars (std over seeds)
fig2_aic_discrimination.png — delta_AIC = AIC_exp - AIC_pow vs S
fig3_chord_histograms.png   — log-linear and log-log fits at 4 key S values
fig4_beta_theory.png        — measured beta vs random-walk prediction
L_vs_S_pooled.csv           — full numerical results
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from numba import njit
import os, csv, time, warnings
warnings.filterwarnings("ignore")

OUTDIR = "/mnt/user-data/outputs/chord_analysis_v2"
os.makedirs(OUTDIR, exist_ok=True)

# ── fixed parameters ──────────────────────────────────────────────────────────
N         = 201
N_PART    = 4000
DISS_RATE = 0.05
M_THRESH  = 0.999
SEEDS     = [42, 123, 999, 7, 314, 2718]   # 6 seeds, same as your Exploration 1
N_CHORDS  = 60_000   # per seed per direction (6 seeds => 360k chords total)

# S sweep: denser at small S where structure emerges, coarser at large S
S_SMALL  = np.array([2, 3, 4, 5, 6, 8, 10, 13, 17, 20, 25, 30,
                     40, 50, 70, 100], dtype=float)
S_LARGE  = np.array([150, 200, 300, 400, 500, 700, 1000, 1500, 2000],
                    dtype=float)
S_VALUES = np.concatenate([S_SMALL, S_LARGE])
CS_VALUES = S_VALUES * DISS_RATE


# ═══════════════════════════════════════════════════════════════════════════════
# CA SIMULATION  (identical physics to v1)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _lcg(state):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF

@njit
def simulate(N, n_part, diss_rate, Cs, seed):
    M   = np.ones((N, N))
    P   = np.zeros((N, N))
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
            open_w   = np.empty(3, dtype=np.float64)
            w_sum    = 0.0
            for k in range(3):
                nr = row + drow[k]
                nc = col + dcol[k]
                if nr < 0 or nr >= N or nc >= N:
                    continue
                if P[nr, nc] >= 1.0:
                    continue
                w             = 2.0 - M[nr, nc]
                open_idx[n_open] = k
                open_w[n_open]   = w
                w_sum           += w
                n_open          += 1

            if n_open == 0:
                C = Cs
                break

            rng, u = _lcg(rng)
            u      *= w_sum
            cumw    = 0.0
            chosen  = open_idx[0]
            for i in range(n_open):
                cumw += open_w[i]
                if u <= cumw:
                    chosen = open_idx[i]
                    break
            row += drow[chosen]
            col += dcol[chosen]
            if col >= N - 1:
                C = Cs
                break

        for _ in range(1):           # one extra post-saturation step
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


# ═══════════════════════════════════════════════════════════════════════════════
# CHORD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def chord_lengths_1d(pore_row):
    lengths = []
    count   = 0
    in_pore = False
    for val in pore_row:
        if val:
            in_pore = True
            count  += 1
        else:
            if in_pore and count > 0:
                lengths.append(count)
            in_pore = False
            count   = 0
    if in_pore and count > 0:
        lengths.append(count)
    return lengths


def extract_chords(M, n_chords, rng, thresh=M_THRESH):
    pore = M < thresh
    lx, ly = [], []
    rows = rng.integers(0, N, size=n_chords)
    cols = rng.integers(0, N, size=n_chords)
    for r in rows:
        lx.extend(chord_lengths_1d(pore[r, :]))
    for c in cols:
        ly.extend(chord_lengths_1d(pore[:, c]))
    return np.array(lx, dtype=float), np.array(ly, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FITTING  (exponential + power law + AIC)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_exponential(lengths):
    """Log-linear fit of P(l) = (1/L) exp(-l/L). Returns L, r2, aic."""
    if len(lengths) < 20:
        return np.nan, np.nan, np.nan
    max_l   = min(int(lengths.max()), int(5 * np.median(lengths)) + 5)
    bins    = np.arange(0.5, max_l + 1.5, 1.0)
    hist, edges = np.histogram(lengths, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask    = hist > 0
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    log_h   = np.log(hist[mask])
    x       = centres[mask]
    coeffs  = np.polyfit(x, log_h, 1)
    if coeffs[0] >= 0:
        return np.nan, np.nan, np.nan
    L       = -1.0 / coeffs[0]
    pred    = np.polyval(coeffs, x)
    ss_res  = np.sum((log_h - pred) ** 2)
    ss_tot  = np.sum((log_h - log_h.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # AIC in log-space (k=2 parameters: L, normalisation)
    n       = mask.sum()
    aic     = n * np.log(ss_res / n + 1e-30) + 2 * 2
    return L, r2, aic


def fit_powerlaw(lengths):
    """Log-log fit of P(l) = A * l^(-alpha). Returns alpha, r2, aic."""
    if len(lengths) < 20:
        return np.nan, np.nan, np.nan
    # use l >= 2 to avoid l=1 artefact
    l_arr   = lengths[lengths >= 2]
    if len(l_arr) < 20:
        return np.nan, np.nan, np.nan
    max_l   = min(int(l_arr.max()), int(5 * np.median(l_arr)) + 5)
    bins    = np.arange(1.5, max_l + 1.5, 1.0)
    hist, edges = np.histogram(l_arr, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask    = (hist > 0) & (centres > 1)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    log_h   = np.log(hist[mask])
    log_x   = np.log(centres[mask])
    coeffs  = np.polyfit(log_x, log_h, 1)
    alpha   = -coeffs[0]
    pred    = np.polyval(coeffs, log_x)
    ss_res  = np.sum((log_h - pred) ** 2)
    ss_tot  = np.sum((log_h - log_h.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n       = mask.sum()
    aic     = n * np.log(ss_res / n + 1e-30) + 2 * 2
    return alpha, r2, aic


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP  (multi-seed pooling)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep():
    print(f"{'S':>7}  {'Lx':>7} {'r2x':>6}  {'Ly':>7} {'r2y':>6}  "
          f"{'Lx/Ly':>6}  {'dAIC_x':>8}  {'t':>5}")
    print("-" * 72)

    results = []
    rng     = np.random.default_rng(0)

    # JIT warm-up
    simulate(21, 5, DISS_RATE, 0.1, 1)

    for S, Cs in zip(S_VALUES, CS_VALUES):
        t0 = time.time()

        # pool chords from all seeds
        lx_all, ly_all = [], []
        for seed in SEEDS:
            M, P = simulate(N, N_PART, DISS_RATE, float(Cs), int(seed))
            lx, ly = extract_chords(M, N_CHORDS, rng)
            lx_all.append(lx)
            ly_all.append(ly)

        lx_pool = np.concatenate(lx_all)
        ly_pool = np.concatenate(ly_all)

        # per-seed L values for std estimation
        Lx_seeds, Ly_seeds = [], []
        for lx, ly in zip(lx_all, ly_all):
            Lx_s, _, _ = fit_exponential(lx)
            Ly_s, _, _ = fit_exponential(ly)
            Lx_seeds.append(Lx_s)
            Ly_seeds.append(Ly_s)
        Lx_seeds = np.array(Lx_seeds)
        Ly_seeds = np.array(Ly_seeds)

        # fit pooled data
        Lx,  r2x,  aic_exp_x  = fit_exponential(lx_pool)
        Ly,  r2y,  aic_exp_y  = fit_exponential(ly_pool)
        _,   _,    aic_pow_x  = fit_powerlaw(lx_pool)
        _,   _,    aic_pow_y  = fit_powerlaw(ly_pool)

        # delta_AIC: positive => exponential fits better; negative => power law
        daic_x = (aic_exp_x - aic_pow_x) if not (np.isnan(aic_exp_x) or
                  np.isnan(aic_pow_x)) else np.nan
        daic_y = (aic_exp_y - aic_pow_y) if not (np.isnan(aic_exp_y) or
                  np.isnan(aic_pow_y)) else np.nan

        Lx_std = np.nanstd(Lx_seeds)
        Ly_std = np.nanstd(Ly_seeds)
        ratio  = Lx / Ly if (np.isfinite(Lx) and np.isfinite(Ly) and Ly > 0) \
                 else np.nan

        elapsed = time.time() - t0
        print(f"{S:7.0f}  {Lx:7.2f} {r2x:6.3f}  {Ly:7.2f} {r2y:6.3f}  "
              f"{ratio:6.3f}  {daic_x:8.2f}  {elapsed:5.1f}")

        results.append(dict(
            S=S, Cs=Cs,
            Lx=Lx, Lx_std=Lx_std, r2x=r2x, aic_exp_x=aic_exp_x, aic_pow_x=aic_pow_x, daic_x=daic_x,
            Ly=Ly, Ly_std=Ly_std, r2y=r2y, aic_exp_y=aic_exp_y, aic_pow_y=aic_pow_y, daic_y=daic_y,
            ratio=ratio,
            n_pore_mean=np.mean([(M < M_THRESH).sum() for M, _ in
                                  [simulate(N, N_PART, DISS_RATE, float(Cs), int(s))
                                   for s in SEEDS[:1]]])  # quick estimate
        ))

    return results, lx_all, ly_all   # return last S's data for histogram


# ═══════════════════════════════════════════════════════════════════════════════
# POWER LAW FIT FOR BETA
# ═══════════════════════════════════════════════════════════════════════════════

def fit_beta(S_arr, L_arr, L_std_arr, label=""):
    """Fit L = a * S^beta in the growth regime (before saturation)."""
    mask = (np.isfinite(L_arr) & np.isfinite(L_std_arr) &
            (L_arr > 2) & (L_arr < 120) & (S_arr >= 10))
    if mask.sum() < 5:
        return np.nan, np.nan
    try:
        popt, pcov = curve_fit(
            lambda x, a, b: a * x**b,
            S_arr[mask], L_arr[mask],
            sigma=np.where(L_std_arr[mask] > 0, L_std_arr[mask], 1.0),
            absolute_sigma=True, p0=[1.0, 0.4], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return popt[1], perr[1]
    except Exception:
        return np.nan, np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_figures(results):
    S_arr    = np.array([r["S"]      for r in results])
    Lx_arr   = np.array([r["Lx"]     for r in results])
    Lx_std   = np.array([r["Lx_std"] for r in results])
    Ly_arr   = np.array([r["Ly"]     for r in results])
    Ly_std   = np.array([r["Ly_std"] for r in results])
    r2x_arr  = np.array([r["r2x"]    for r in results])
    r2y_arr  = np.array([r["r2y"]    for r in results])
    daic_x   = np.array([r["daic_x"] for r in results])
    daic_y   = np.array([r["daic_y"] for r in results])
    rat_arr  = np.array([r["ratio"]  for r in results])

    beta_x, beta_x_err = fit_beta(S_arr, Lx_arr, Lx_std, "x")
    beta_y, beta_y_err = fit_beta(S_arr, Ly_arr, Ly_std, "y")
    print(f"\n  beta_x = {beta_x:.3f} +/- {beta_x_err:.3f}")
    print(f"  beta_y = {beta_y:.3f} +/- {beta_y_err:.3f}")
    print(f"  naive random-walk prediction: beta = 0.5")
    print(f"  discrepancy from 0.5: {0.5 - beta_x:.3f} (x),  {0.5 - beta_y:.3f} (y)")

    # ── Fig 1: L(S) with error bars ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    good_x = np.isfinite(Lx_arr) & np.isfinite(Lx_std)
    good_y = np.isfinite(Ly_arr) & np.isfinite(Ly_std)
    ax.errorbar(S_arr[good_x], Lx_arr[good_x], yerr=Lx_std[good_x],
                fmt="o-", color="#1a5276", ms=4, lw=1.2, capsize=3, elinewidth=0.8,
                label=r"$L_x$ (dissolution direction)")
    ax.errorbar(S_arr[good_y], Ly_arr[good_y], yerr=Ly_std[good_y],
                fmt="s--", color="#148f77", ms=4, lw=1.2, capsize=3, elinewidth=0.8,
                label=r"$L_y$ (transverse)")

    # power law fits
    S_fit = np.logspace(1, np.log10(200), 100)
    if not np.isnan(beta_x):
        # find prefactor
        mask = np.isfinite(Lx_arr) & (Lx_arr > 2) & (Lx_arr < 120) & (S_arr >= 10)
        if mask.sum() > 0:
            ax_pre = np.median(Lx_arr[mask] / S_arr[mask]**beta_x)
            ax.loglog(S_fit, ax_pre * S_fit**beta_x, ":", color="#1a5276", lw=1.5,
                      label=fr"$\beta_x = {beta_x:.2f} \pm {beta_x_err:.2f}$")
    if not np.isnan(beta_y):
        mask = np.isfinite(Ly_arr) & (Ly_arr > 2) & (Ly_arr < 120) & (S_arr >= 10)
        if mask.sum() > 0:
            ay_pre = np.median(Ly_arr[mask] / S_arr[mask]**beta_y)
            ax.loglog(S_fit, ay_pre * S_fit**beta_y, ":", color="#148f77", lw=1.5,
                      label=fr"$\beta_y = {beta_y:.2f} \pm {beta_y_err:.2f}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S = C_s\,/\,\mathrm{rate}$", fontsize=11)
    ax.set_ylabel(r"Characteristic pore length $L$ (cells)", fontsize=11)
    ax.set_title(r"$L(S)$ — pooled over 6 seeds", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=9)

    # anisotropy
    ax2 = axes[1]
    mask = np.isfinite(rat_arr)
    ax2.semilogx(S_arr[mask], rat_arr[mask], "^-", color="#922b21", ms=4, lw=1.2)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, label="isotropic")
    ax2.axhline(2.0, color="#f0a500", ls="--", lw=0.8,
                label=r"$L_x/L_y = 2$ (walk theory)")
    ax2.set_xlabel(r"$S$", fontsize=11)
    ax2.set_ylabel(r"$L_x\,/\,L_y$", fontsize=11)
    ax2.set_title("Dissolution-induced anisotropy", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax2.set_ylim(0, 3)
    ax2.tick_params(labelsize=9)

    # R² quality
    ax3 = axes[2]
    ax3.semilogx(S_arr, r2x_arr, "o-", color="#1a5276", ms=4, lw=1.2, label=r"$R^2_x$ (exp fit)")
    ax3.semilogx(S_arr, r2y_arr, "s--", color="#148f77", ms=4, lw=1.2, label=r"$R^2_y$ (exp fit)")
    ax3.axhline(0.95, color="gray", ls=":", lw=0.8, label="R²=0.95")
    ax3.set_xlabel(r"$S$", fontsize=11)
    ax3.set_ylabel(r"$R^2$ of exponential fit", fontsize=11)
    ax3.set_title("Goodness of exponential fit", fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8); ax3.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax3.tick_params(labelsize=9)

    fig.suptitle(f"Chord-length analysis · N={N}, N_part={N_PART}, "
                 f"diss_rate={DISS_RATE} · pooled over {len(SEEDS)} seeds",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_L_vs_S_pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_L_vs_S_pooled.png")

    # ── Fig 2: delta AIC ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = np.isfinite(daic_x)
    ax.semilogx(S_arr[mask], daic_x[mask], "o-", color="#1a5276",
                ms=4, lw=1.2, label=r"$x$-direction")
    mask = np.isfinite(daic_y)
    ax.semilogx(S_arr[mask], daic_y[mask], "s--", color="#148f77",
                ms=4, lw=1.2, label=r"$y$-direction")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(400, color="#e74c3c", ls=":", lw=1.0, label="S* ~ 400 (branching transition)")
    ax.fill_between(S_arr[np.isfinite(daic_x)], 0,
                    daic_x[np.isfinite(daic_x)],
                    where=daic_x[np.isfinite(daic_x)] < 0,
                    alpha=0.15, color="#e74c3c", label="power law preferred")
    ax.fill_between(S_arr[np.isfinite(daic_x)], 0,
                    daic_x[np.isfinite(daic_x)],
                    where=daic_x[np.isfinite(daic_x)] >= 0,
                    alpha=0.15, color="#1a5276", label="exponential preferred")
    ax.set_xlabel(r"$S = C_s\,/\,\mathrm{rate}$", fontsize=11)
    ax.set_ylabel(r"$\Delta\mathrm{AIC} = \mathrm{AIC}_\mathrm{exp} - \mathrm{AIC}_\mathrm{pow}$",
                  fontsize=11)
    ax.set_title("Exponential vs power-law: model comparison via AIC\n"
                 r"$\Delta$AIC > 0 $\Rightarrow$ power law fits better", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_aic_discrimination.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_aic_discrimination.png")

    # ── Fig 3: chord histograms at 4 key S values ─────────────────────────────
    S_examples = [10, 50, 200, 500]
    fig, axes_all = plt.subplots(3, 4, figsize=(15, 10))
    rng_ex = np.random.default_rng(77)
    CMAP   = matplotlib.colors.ListedColormap(["#1a5276", "#f0c030", "#148f77"])

    for col_i, S_ex in enumerate(S_examples):
        Cs_ex = S_ex * DISS_RATE
        # run all seeds, pool
        lx_p, ly_p = [], []
        M_show, P_show = None, None
        for seed in SEEDS:
            M_s, P_s = simulate(N, N_PART, DISS_RATE, Cs_ex, int(seed))
            lx_s, ly_s = extract_chords(M_s, 30_000, rng_ex)
            lx_p.append(lx_s); ly_p.append(ly_s)
            if M_show is None:
                M_show, P_show = M_s, P_s
        lx_pool = np.concatenate(lx_p)
        ly_pool = np.concatenate(ly_p)

        Lx_ex, r2x_ex, aic_exp_x = fit_exponential(lx_pool)
        Ly_ex, r2y_ex, aic_exp_y = fit_exponential(ly_pool)
        ax_ex, r2pw_x, aic_pow_x = fit_powerlaw(lx_pool)
        ay_ex, r2pw_y, aic_pow_y = fit_powerlaw(ly_pool)

        # row 0: grid
        img = np.full(M_show.shape, 2, dtype=int)
        img[M_show < M_THRESH] = 0
        img[P_show == 1]       = 1
        axes_all[0, col_i].imshow(img, cmap=CMAP, vmin=0, vmax=2,
                                  origin="lower", interpolation="nearest")
        axes_all[0, col_i].set_title(f"S = {S_ex}", fontsize=10)
        axes_all[0, col_i].axis("off")

        # row 1: log-linear (exponential test)
        ax_ll = axes_all[1, col_i]
        _plot_hist_loglin(lx_pool, Lx_ex, r2x_ex, ax_ll,
                          f"x-chord, S={S_ex}")

        # row 2: log-log (power-law test)
        ax_lg = axes_all[2, col_i]
        _plot_hist_loglog(lx_pool, ax_ex, r2pw_x, ax_lg,
                          f"x-chord, S={S_ex}")

    axes_all[0, 0].set_ylabel("grid (seed=42)", fontsize=8)
    axes_all[1, 0].set_ylabel("log-linear\n(exp. test)", fontsize=8)
    axes_all[2, 0].set_ylabel("log-log\n(power-law test)", fontsize=8)

    fig.suptitle("Chord-length distribution: exponential vs power-law at four S values\n"
                 "Pooled over 6 seeds", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_chord_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_chord_histograms.png")

    # ── Fig 4: beta theory comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    S_fit  = np.logspace(1, np.log10(200), 200)

    # measured Lx with error bars
    good = np.isfinite(Lx_arr) & np.isfinite(Lx_std) & (Lx_arr > 2) & (Lx_arr < 120) & (S_arr >= 10)
    ax.errorbar(S_arr[good], Lx_arr[good], yerr=Lx_std[good],
                fmt="o", color="#1a5276", ms=5, capsize=3, elinewidth=0.8,
                label=r"$L_x$ (measured, pooled)")

    # measured power law
    if not np.isnan(beta_x):
        mask = good
        pre  = np.median(Lx_arr[mask] / S_arr[mask]**beta_x)
        ax.loglog(S_fit, pre * S_fit**beta_x, "-", color="#1a5276", lw=1.8,
                  label=fr"measured: $\beta_x = {beta_x:.2f} \pm {beta_x_err:.2f}$")

    # naive random-walk prediction beta=0.5
    pre_rw = np.sqrt(2.0 / 3.0)   # from lateral diffusion: sigma ~ sqrt(2S/3)
    ax.loglog(S_fit, pre_rw * S_fit**0.5, "--", color="#e74c3c", lw=1.8,
              label=r"random walk: $\beta = 0.5$  ($\sim\!\sqrt{2S/3}$)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S = C_s\,/\,\mathrm{rate}$", fontsize=11)
    ax.set_ylabel(r"$L_x$ (cells)", fontsize=11)
    ax.set_title(r"Measured $\beta$ vs naive random-walk prediction", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=9)

    # annotate discrepancy
    if not np.isnan(beta_x):
        ax.annotate(
            fr"$\Delta\beta = 0.50 - {beta_x:.2f} = {0.5 - beta_x:.2f}$" + "\n"
            r"(weighted walk $\Rightarrow$ correlated tracks)",
            xy=(40, pre * 40**beta_x), xytext=(80, pre * 5**beta_x),
            fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_beta_theory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_beta_theory.png")


def _plot_hist_loglin(lengths, L, r2, ax, title):
    if len(lengths) < 10 or np.isnan(L):
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", fontsize=8, color="gray")
        return
    max_l = min(int(lengths.max()), int(5 * np.median(lengths)))
    bins  = np.arange(0.5, max_l + 1.5, 1.0)
    h, e  = np.histogram(lengths, bins=bins, density=True)
    c     = 0.5 * (e[:-1] + e[1:])
    ax.bar(c, h, width=0.8, color="#5dade2", alpha=0.55)
    x_fit = np.linspace(0.5, max_l, 200)
    ax.semilogy(x_fit, (1/L) * np.exp(-x_fit / L), "r-", lw=1.5,
                label=f"exp, L={L:.1f}\nR²={r2:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("l (cells)", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


def _plot_hist_loglog(lengths, alpha, r2, ax, title):
    l_arr = lengths[lengths >= 2] if len(lengths) > 0 else np.array([])
    if len(l_arr) < 10 or np.isnan(alpha):
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", fontsize=8, color="gray")
        return
    max_l = min(int(l_arr.max()), int(5 * np.median(l_arr)))
    bins  = np.arange(1.5, max_l + 1.5, 1.0)
    h, e  = np.histogram(l_arr, bins=bins, density=True)
    c     = 0.5 * (e[:-1] + e[1:])
    mask  = h > 0
    ax.loglog(c[mask], h[mask], "o", color="#5dade2", ms=3, alpha=0.7)
    if not np.isnan(alpha):
        x_fit = np.logspace(np.log10(2), np.log10(max_l), 100)
        pre   = np.exp(np.mean(np.log(h[mask]) + alpha * np.log(c[mask])))
        ax.loglog(x_fit, pre * x_fit**(-alpha), "r-", lw=1.5,
                  label=fr"$\alpha={alpha:.2f}$, R²={r2:.3f}")
    ax.set_xlabel("l (cells)", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(results):
    path   = f"{OUTDIR}/L_vs_S_pooled.csv"
    fields = ["S","Cs","Lx","Lx_std","r2x","aic_exp_x","aic_pow_x","daic_x",
              "Ly","Ly_std","r2y","aic_exp_y","aic_pow_y","daic_y","ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("Phase 01 v2 — multi-seed pooling + model discrimination + beta theory")
    print(f"N={N}, N_part={N_PART}, diss_rate={DISS_RATE}")
    print(f"Seeds: {SEEDS}")
    print(f"S range: {S_VALUES[0]:.0f} to {S_VALUES[-1]:.0f}  ({len(S_VALUES)} values)")
    print("=" * 72)

    results, _, _ = run_sweep()

    print("\nGenerating figures ...")
    make_all_figures(results)
    save_csv(results)
    print("\nDone. Output:", OUTDIR)
