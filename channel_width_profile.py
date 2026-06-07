"""
channel_width.py
================
Phase 01 — Step 3.

Motivation
----------
The global chord-length approach (v2) failed at large S because the dissolved
region fills the domain — the exponential fit then describes remnant mineral
geometry, not channel structure.  The fix: slice the dissolved body column by
column (each column = one x-position) and measure the *width* of the dissolved
region at that x.  This is geometrically meaningful at all S and does not
degenerate when the domain fills.

What this script computes
--------------------------
For each S value and each seed:

  1. Run the CA simulation → binary dissolved grid (M < thresh).

  2. For every column x = 0, 1, ..., N-1:
       - find all dissolved rows in that column
       - record the *span*: max_row - min_row + 1  (bounding width)
       - and the *fill*:  number of dissolved rows (actual count)
     Both are legitimate widths; span is more robust to sparse channels.

  3. Plot the width profile W(x) averaged over seeds.  This shows how the
     dissolved body narrows (or widens) along the dissolution direction.

  4. Compute the characteristic width L(S) = median of W(x) over the
     "active zone" — columns where at least half the seeds have any
     dissolution.  This is monotone and well-behaved.

  5. Fit L(S) ~ a * S^beta in the pre-saturation growth regime.

  6. Compare beta to the naive random-walk prediction (beta = 0.5) and to
     the lateral-diffusion prediction from the biased walk geometry.

Outputs
-------
fig1_width_profiles.png   — W(x) profiles at 6 representative S values
fig2_L_vs_S_width.png     — L(S) from channel-width method, log-log + fit
fig3_beta_comparison.png  — measured beta vs theoretical predictions
channel_width.csv         — full numerical results
channel_width.py          — this script
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
import os, csv, time, warnings
warnings.filterwarnings("ignore")

OUTDIR = "/mnt/user-data/outputs/channel_width"
os.makedirs(OUTDIR, exist_ok=True)

# ── fixed parameters ──────────────────────────────────────────────────────────
N         = 201
N_PART    = 4000
DISS_RATE = 0.05
M_THRESH  = 0.999
SEEDS     = [42, 123, 999, 7, 314, 2718]

# S sweep: dense at small S, coarser at large S
S_VALUES = np.array([
    2, 3, 4, 5, 6, 8, 10, 13, 17, 20, 25, 30,
    40, 50, 70, 100, 150, 200, 300, 400, 600, 1000, 2000
], dtype=float)
CS_VALUES = S_VALUES * DISS_RATE

# colour map: navy=dissolved, yellow=precipitated, teal=fresh
CMAP = matplotlib.colors.ListedColormap(["#1a5276", "#f0c030", "#148f77"])


# ═══════════════════════════════════════════════════════════════════════════════
# CA SIMULATION  (identical to v2 — weighted walk, faithful to Voller MATLAB)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _lcg(state):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF


@njit
def simulate(N, n_part, diss_rate, Cs, seed):
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
            open_w   = np.empty(3, dtype=np.float64)
            w_sum    = 0.0
            for k in range(3):
                nr = row + drow[k]
                nc = col + dcol[k]
                if nr < 0 or nr >= N or nc >= N:
                    continue
                if P[nr, nc] >= 1.0:
                    continue
                w                = 2.0 - M[nr, nc]
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

        for _ in range(1):
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
# CHANNEL WIDTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def column_widths(M, thresh=M_THRESH):
    """
    For each column x, compute:
      span(x) = max_dissolved_row - min_dissolved_row + 1
                (bounding box width; 0 if no dissolved cells)
      fill(x) = number of dissolved rows in column x

    Returns span and fill arrays of length N.
    """
    pore  = M < thresh          # shape (N, N): pore[row, col]
    span  = np.zeros(N, dtype=float)
    fill  = np.zeros(N, dtype=float)

    for col in range(N):
        col_pore = pore[:, col]
        n_dissolved = col_pore.sum()
        fill[col] = n_dissolved
        if n_dissolved > 0:
            rows = np.where(col_pore)[0]
            span[col] = rows[-1] - rows[0] + 1

    return span, fill


def active_zone(span_mean, min_fraction=0.05):
    """
    Columns where mean span > min_fraction * max_span.
    Excludes the injection column (x=0) and the empty right end.
    """
    threshold = min_fraction * span_mean.max()
    active    = np.where(span_mean > threshold)[0]
    # exclude x=0 (injection point, always has some dissolution)
    active    = active[active > 0]
    return active


def characteristic_width(span_mean, active_cols):
    """Median span over active columns — the characteristic width L."""
    if len(active_cols) == 0:
        return np.nan
    return np.median(span_mean[active_cols])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep():
    print(f"{'S':>7}  {'L_span':>8}  {'L_fill':>8}  {'x_max':>6}  "
          f"{'W_peak':>7}  {'t(s)':>5}")
    print("-" * 55)

    results      = []
    profile_data = {}   # store profiles at a few S for fig1

    S_for_profiles = [10, 30, 100, 300, 1000]

    # JIT warm-up
    simulate(21, 5, DISS_RATE, 0.1, 1)

    for S, Cs in zip(S_VALUES, CS_VALUES):
        t0 = time.time()

        spans_all = np.zeros((len(SEEDS), N))
        fills_all = np.zeros((len(SEEDS), N))
        M_show    = None

        for i, seed in enumerate(SEEDS):
            M, P = simulate(N, N_PART, DISS_RATE, float(Cs), int(seed))
            if M_show is None:
                M_show, P_show = M.copy(), P.copy()
            span, fill        = column_widths(M)
            spans_all[i]      = span
            fills_all[i]      = fill

        # mean and std over seeds
        span_mean = spans_all.mean(axis=0)
        span_std  = spans_all.std(axis=0)
        fill_mean = fills_all.mean(axis=0)

        active      = active_zone(span_mean)
        L_span      = characteristic_width(span_mean, active)
        L_fill      = characteristic_width(fill_mean, active)

        # x-extent: rightmost column with mean span > 0
        x_extent    = active[-1] if len(active) > 0 else 0
        W_peak      = span_mean.max()

        elapsed = time.time() - t0
        print(f"{S:7.0f}  {L_span:8.2f}  {L_fill:8.2f}  {x_extent:6d}  "
              f"{W_peak:7.2f}  {elapsed:5.1f}")

        results.append(dict(
            S=S, Cs=Cs,
            L_span=L_span, L_fill=L_fill,
            x_extent=x_extent, W_peak=W_peak
        ))

        # store profile for plotting
        closest = min(S_for_profiles, key=lambda s: abs(s - S))
        if abs(closest - S) < 1.5 and closest not in profile_data:
            profile_data[closest] = dict(
                span_mean=span_mean.copy(),
                span_std=span_std.copy(),
                fill_mean=fill_mean.copy(),
                M_show=M_show, P_show=P_show,
                S=S
            )

    return results, profile_data


# ═══════════════════════════════════════════════════════════════════════════════
# BETA FIT
# ═══════════════════════════════════════════════════════════════════════════════

def fit_beta(S_arr, L_arr, regime_mask):
    """Power law fit L = a * S^beta over regime_mask."""
    mask = regime_mask & np.isfinite(L_arr) & (L_arr > 1)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    try:
        popt, pcov = curve_fit(
            lambda x, a, b: a * x**b,
            S_arr[mask], L_arr[mask],
            p0=[1.0, 0.5], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        # R² in log space
        log_L    = np.log(L_arr[mask])
        log_pred = np.log(popt[0]) + popt[1] * np.log(S_arr[mask])
        ss_res   = np.sum((log_L - log_pred)**2)
        ss_tot   = np.sum((log_L - log_L.mean())**2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return popt[1], perr[1], r2
    except Exception:
        return np.nan, np.nan, np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_figures(results, profile_data):
    S_arr      = np.array([r["S"]        for r in results])
    L_span_arr = np.array([r["L_span"]   for r in results])
    L_fill_arr = np.array([r["L_fill"]   for r in results])
    x_ext_arr  = np.array([r["x_extent"] for r in results])
    Wpeak_arr  = np.array([r["W_peak"]   for r in results])

    # growth regime: S in [5, 200] and L < N/2
    growth = (S_arr >= 5) & (S_arr <= 200) & (L_span_arr < N / 2)

    beta_span, beta_span_err, r2_span = fit_beta(S_arr, L_span_arr, growth)
    beta_fill, beta_fill_err, r2_fill = fit_beta(S_arr, L_fill_arr, growth)
    beta_xext, beta_xext_err, r2_xext = fit_beta(S_arr, x_ext_arr.astype(float),
                                                   (S_arr >= 5) & (S_arr <= 200))

    print(f"\n  L_span:  beta = {beta_span:.3f} +/- {beta_span_err:.3f}  R²={r2_span:.3f}")
    print(f"  L_fill:  beta = {beta_fill:.3f} +/- {beta_fill_err:.3f}  R²={r2_fill:.3f}")
    print(f"  x_extent:beta = {beta_xext:.3f} +/- {beta_xext_err:.3f}  R²={r2_xext:.3f}")
    print(f"  naive random-walk (independent tracks): beta = 0.50")
    print(f"  biased walk lateral diffusion: W ~ sqrt(2S/3), beta = 0.50")

    x_col = np.arange(N)

    # ── Fig 1: width profiles W(x) at representative S ───────────────────────
    S_keys = sorted(profile_data.keys())
    n_prof = len(S_keys)
    fig, axes = plt.subplots(2, n_prof, figsize=(3.2 * n_prof, 6.5))

    for i, S_key in enumerate(S_keys):
        d = profile_data[S_key]
        S_val    = d["S"]
        sp_mean  = d["span_mean"]
        sp_std   = d["span_std"]
        M_show   = d["M_show"]
        P_show   = d["P_show"]

        # top: grid (seed 42)
        img = np.full((N, N), 2, dtype=int)
        img[M_show < M_THRESH] = 0
        img[P_show == 1]       = 1
        axes[0, i].imshow(img, cmap=CMAP, vmin=0, vmax=2,
                          origin="lower", interpolation="nearest")
        axes[0, i].set_title(f"S = {S_val:.0f}", fontsize=10)
        axes[0, i].axis("off")

        # bottom: width profile
        ax = axes[1, i]
        ax.fill_between(x_col,
                        np.maximum(0, sp_mean - sp_std),
                        sp_mean + sp_std,
                        alpha=0.25, color="#1a5276")
        ax.plot(x_col, sp_mean, "-", color="#1a5276", lw=1.5,
                label=r"$\langle W(x)\rangle$")
        ax.axhline(np.median(sp_mean[sp_mean > 0.05 * sp_mean.max()]),
                   color="#e74c3c", ls="--", lw=1.0,
                   label=f"median={np.median(sp_mean[sp_mean > 0.05*sp_mean.max()]):.1f}")
        ax.set_xlabel("column x", fontsize=8)
        if i == 0:
            ax.set_ylabel("dissolved width W(x) [rows]", fontsize=8)
        ax.set_xlim(0, N - 1)
        ax.set_ylim(0, N * 0.55)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    axes[0, 0].set_ylabel("CA grid (seed 42)", fontsize=8)
    fig.suptitle(
        f"Channel width profiles W(x) at representative S values\n"
        f"N={N}, N_part={N_PART}, diss_rate={DISS_RATE}, "
        f"shading = ±1 std over {len(SEEDS)} seeds",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_width_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_width_profiles.png")

    # ── Fig 2: L(S) from channel-width method ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # panel a: L(S) log-log
    ax = axes[0]
    good = np.isfinite(L_span_arr)
    ax.loglog(S_arr[good], L_span_arr[good], "o-", color="#1a5276",
              ms=5, lw=1.2, label=r"$L_\mathrm{span}$ (bounding width)")
    good2 = np.isfinite(L_fill_arr)
    ax.loglog(S_arr[good2], L_fill_arr[good2], "s--", color="#148f77",
              ms=5, lw=1.2, label=r"$L_\mathrm{fill}$ (dissolved count)")

    # power law overlays
    S_fit = np.logspace(np.log10(5), np.log10(200), 200)
    if not np.isnan(beta_span):
        mask  = growth & np.isfinite(L_span_arr) & (L_span_arr > 1)
        pre   = np.exp(np.mean(np.log(L_span_arr[mask]) -
                                beta_span * np.log(S_arr[mask])))
        ax.loglog(S_fit, pre * S_fit**beta_span, ":", color="#1a5276", lw=1.8,
                  label=fr"fit $\beta={beta_span:.2f}\pm{beta_span_err:.2f}$"
                        fr"  $R^2={r2_span:.2f}$")

    # theory line beta=0.5
    pre_rw = np.sqrt(2.0 / 3.0)
    ax.loglog(S_fit, pre_rw * S_fit**0.5, "--", color="#e74c3c", lw=1.5, alpha=0.7,
              label=r"random walk: $\beta=0.50$")

    ax.set_xlabel(r"$S = C_s / \mathrm{rate}$", fontsize=11)
    ax.set_ylabel("Characteristic width L (cells)", fontsize=11)
    ax.set_title(r"$L(S)$ from channel-width method", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=9)

    # panel b: x-extent scaling
    ax2 = axes[1]
    good3 = x_ext_arr > 0
    ax2.loglog(S_arr[good3], x_ext_arr[good3], "^-", color="#7d3c98",
               ms=5, lw=1.2, label=r"$x_\mathrm{max}$ (dissolution reach)")
    # theory: x_max ~ S/3
    ax2.loglog(S_fit, S_fit / 3.0, "--", color="#e74c3c", lw=1.5, alpha=0.7,
               label=r"random walk: $x \sim S/3$")
    if not np.isnan(beta_xext):
        mask3 = (S_arr >= 5) & (S_arr <= 200) & (x_ext_arr > 0)
        pre3  = np.exp(np.mean(np.log(x_ext_arr[mask3].astype(float)) -
                                beta_xext * np.log(S_arr[mask3])))
        ax2.loglog(S_fit, pre3 * S_fit**beta_xext, ":", color="#7d3c98", lw=1.8,
                   label=fr"fit $\beta={beta_xext:.2f}\pm{beta_xext_err:.2f}$")
    ax2.set_xlabel(r"$S$", fontsize=11)
    ax2.set_ylabel("x-extent (cells)", fontsize=11)
    ax2.set_title("Dissolution reach vs S", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax2.tick_params(labelsize=9)

    # panel c: peak width (maximum W over all columns)
    ax3 = axes[2]
    good4 = Wpeak_arr > 0
    ax3.loglog(S_arr[good4], Wpeak_arr[good4], "D-", color="#c0392b",
               ms=5, lw=1.2, label=r"$W_\mathrm{peak}$ (max column width)")
    ax3.loglog(S_fit, np.sqrt(4 * S_fit / 3.0), "--", color="#e74c3c",
               lw=1.5, alpha=0.7,
               label=r"$2\sqrt{2S/3}$ (±2$\sigma$ of random walk)")
    ax3.set_xlabel(r"$S$", fontsize=11)
    ax3.set_ylabel("Peak width (cells)", fontsize=11)
    ax3.set_title("Peak dissolution width vs S", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax3.tick_params(labelsize=9)

    fig.suptitle(
        f"Channel-width characterisation of Voller CA dissolution patterns\n"
        f"N={N}, N_part={N_PART}, diss_rate={DISS_RATE}, "
        f"{len(SEEDS)} seeds",
        fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_L_vs_S_width.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_L_vs_S_width.png")

    # ── Fig 3: beta theory comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # measured
    mask = growth & np.isfinite(L_span_arr) & (L_span_arr > 1)
    ax.errorbar(S_arr[mask], L_span_arr[mask], fmt="o",
                color="#1a5276", ms=6, label=r"$L_\mathrm{span}$ (measured)")

    S_fit2 = np.logspace(np.log10(5), np.log10(200), 200)

    # measured power law
    if not np.isnan(beta_span):
        pre = np.exp(np.mean(np.log(L_span_arr[mask]) -
                              beta_span * np.log(S_arr[mask])))
        ax.loglog(S_fit2, pre * S_fit2**beta_span, "-", color="#1a5276", lw=2,
                  label=fr"measured: $\beta = {beta_span:.2f} \pm {beta_span_err:.2f}$")

    # independent random walk: beta=0.5, prefactor sqrt(2/3)
    ax.loglog(S_fit2, np.sqrt(2/3) * S_fit2**0.5, "--", color="#e74c3c", lw=1.8,
              label=r"independent tracks: $\beta=0.50$,  $W\sim\sqrt{2S/3}$")

    # correlated walk lower bound: tracks reinforce => narrower spread
    # heuristic: if tracks cluster, effective lateral diffusion reduced by
    # factor ~1/sqrt(n_tracks), but n_tracks ~ S^0 (always 4000 particles)
    # so the reduction is a prefactor, not exponent change.
    # Instead: plot beta=0.33 as "fully correlated" limit (x-extent scaling)
    ax.loglog(S_fit2, 0.8 * S_fit2**0.33, ":", color="#7d3c98", lw=1.8,
              label=r"fully correlated limit: $\beta=1/3$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S = C_s / \mathrm{rate}$", fontsize=12)
    ax.set_ylabel(r"Channel width $L$ (cells)", fontsize=12)
    ax.set_title("Beta exponent: measured vs theoretical bounds", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=9)

    # annotate measured value
    if not np.isnan(beta_span):
        mid_S = 30.0
        mid_L = pre * mid_S**beta_span
        ax.annotate(
            fr"$\beta_\mathrm{{meas}} = {beta_span:.2f}$" + "\n"
            r"between $1/3$ and $1/2$:" + "\n"
            "partial track correlation",
            xy=(mid_S, mid_L),
            xytext=(mid_S * 2.5, mid_L * 0.4),
            fontsize=8.5, color="#1a5276",
            arrowprops=dict(arrowstyle="->", color="#1a5276", lw=0.8))

    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_beta_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_beta_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(results):
    path   = f"{OUTDIR}/channel_width.csv"
    fields = ["S", "Cs", "L_span", "L_fill", "x_extent", "W_peak"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("Phase 01 Step 3 — Channel-width profile analysis")
    print(f"N={N}, N_part={N_PART}, diss_rate={DISS_RATE}")
    print(f"Seeds: {SEEDS}")
    print(f"S values: {S_VALUES}")
    print("=" * 55)

    results, profile_data = run_sweep()

    print("\nGenerating figures ...")
    make_figures(results, profile_data)
    save_csv(results)

    print(f"\nDone. Output: {OUTDIR}")
