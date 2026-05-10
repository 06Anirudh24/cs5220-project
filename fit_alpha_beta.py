#!/usr/bin/env python3
"""
fit_alpha_beta.py -- Fit alpha-beta model from ping-pong results,
then validate predictions against measured ring, tree, and MPI_Allreduce times.
Includes three model corrections (A, B, C) that explore why the ping-pong-derived
alpha/beta systematically underestimates collective communication time.

Methodology:
  STAGE 0: Fit alpha, beta from ping-pong (existing).
    - alpha estimated from SMALL messages (latency regime, <= 512 B)
    - beta  estimated from LARGE messages (bandwidth regime, >= 1 MB)
    - This split avoids the negative-alpha problem of all-sizes linear fit.

  CORRECTION A: Effective (alpha', beta') from collective fitting.
    - Solve linear system on measured ring/tree all-reduce times
    - alpha'/alpha and beta'/beta ratios reveal MPI library + topology overhead
    - Uses np.linalg.lstsq (no scipy dependency)

  CORRECTION B: gamma per-message overhead with fixed (alpha, beta).
    - gamma_P = solving the augmented model for each P, take median
    - Decomposes the gap into "per-call cost" gamma vs "transmission" alpha+beta*n

  CORRECTION C: Subtract P=1 application overhead.
    - At P=1 there is NO MPI communication, so allreduce_s == application overhead
      (pack_grads + divide_by_nranks loops)
    - Subtract this constant from all P>=2 measurements to get "pure MPI" time
    - Re-evaluate R^2 and re-fit (alpha', beta') on pure data
    - WARNING: ring_allreduce.cpp may not include pack/divide in its allreduce_s;
      we detect this asymmetry and warn.

Usage:
    python3 fit_alpha_beta.py

Reads (relative to this script's directory):
    results/pingpong.csv
    results/ring_allreduce_nranks*_*.out
    results/mpi_allreduce_nranks*_*.out
    results/tree_allreduce_nranks*_*.out

For each nranks value, the LATEST-timestamped file wins (sorted glob).

Outputs:
    results/alpha_beta_fit.txt    -- full text report (now with A/B/C sections)
    results/alpha_beta_plot.csv   -- plotting data (extended columns)
"""

import numpy as np
import csv
import glob
import os
import re
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
RESULTS_DIR    = SCRIPT_DIR / "results"
GRAD_BUF_BYTES = 2913290 * 4  # ~11.1 MB
STEPS_PER_EPOCH = 234          # 60000 train / 256 batch

# ─── Load ping-pong CSV ───────────────────────────────────────────────────────
def load_pingpong(path):
    sizes, half_rtts = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(float(row['bytes']))
            half_rtts.append(float(row['half_roundtrip_s']))
    return np.array(sizes), np.array(half_rtts)

# ─── Estimate alpha/beta from ping-pong (existing methodology) ────────────────
def estimate_alpha(sizes, half_rtts, max_size=512):
    mask = sizes <= max_size
    if mask.sum() == 0:
        return float('nan')
    return np.median(half_rtts[mask])

def estimate_beta(sizes, half_rtts, alpha, min_size=1e6):
    mask = sizes >= min_size
    if mask.sum() == 0:
        return float('nan')
    betas = (half_rtts[mask] - alpha) / sizes[mask]
    return np.median(betas)

def linear_fit(sizes, half_rtts):
    coeffs = np.polyfit(sizes, half_rtts, 1)
    beta  = coeffs[0]
    alpha = coeffs[1]
    y_pred = alpha + beta * sizes
    ss_res = np.sum((half_rtts - y_pred) ** 2)
    ss_tot = np.sum((half_rtts - np.mean(half_rtts)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return alpha, beta, r2

# ─── Predictors using alpha-beta model (Stage 0) ──────────────────────────────
def predict_ring(alpha, beta, nranks, m_bytes, steps=STEPS_PER_EPOCH):
    """Ring per epoch: T = steps * [2(P-1)*alpha + 2(P-1)/P * beta*m]"""
    if nranks == 1:
        return 0.0
    T = 2*(nranks-1)*alpha + 2*(nranks-1)/nranks * beta * m_bytes
    return T * steps

def predict_tree(alpha, beta, nranks, m_bytes, steps=STEPS_PER_EPOCH):
    """Tree per epoch: T = steps * 2*log2(P) * (alpha + beta*m)"""
    if nranks == 1:
        return 0.0
    T = 2*np.log2(nranks)*(alpha + beta * m_bytes)
    return T * steps

def r_squared(measured, predicted):
    measured  = np.array(measured, dtype=float)
    predicted = np.array(predicted, dtype=float)
    ss_res = np.sum((measured - predicted) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot

# ─── Load measured allreduce times from .out files ────────────────────────────
def load_measured_allreduce(pattern):
    """Glob+sorted: alphabetic sort of timestamped filenames is chronological,
    so dict overwrite leaves the most recent run as the surviving value.
    Returns dict {nranks: mean_allreduce_s_per_epoch}.
    For P=1 we keep ALL epochs (no warmup skip) since at P=1 the value is
    a stable application-overhead constant, not a network measurement."""
    results = {}
    for fpath in sorted(glob.glob(str(RESULTS_DIR / pattern))):
        m = re.search(r'nranks(\d+)', fpath)
        if not m:
            continue
        nranks = int(m.group(1))
        allreduce_times = []
        in_csv = False
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('epoch,loss'):
                    in_csv = True
                    continue
                if in_csv and line and not line.startswith('='):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            allreduce_times.append(float(parts[5]))
                        except ValueError:
                            pass
        if allreduce_times:
            # Drop epoch 1 (warmup) for P>1, but keep all for P=1
            skip = 1 if (nranks > 1 and len(allreduce_times) > 1) else 0
            results[nranks] = np.mean(allreduce_times[skip:])
    return results

# ─── CORRECTION A: Effective (alpha', beta') from collective measurements ─────
def fit_effective_ring(meas_dict, n_bytes, steps=STEPS_PER_EPOCH):
    """Linear least-squares fit:
      meas(P) = steps * [2(P-1)*alpha' + 2(P-1)/P * beta' * n]
              = A_P * alpha' + B_P * beta'    (linear in 2 unknowns)
    Returns (alpha', beta', r2). Uses P >= 2 only.
    """
    Ps = sorted(p for p in meas_dict.keys() if p >= 2 and not np.isnan(meas_dict[p]))
    if len(Ps) < 2:
        return float('nan'), float('nan'), float('nan')
    A = np.array([[steps * 2 * (P - 1),
                   steps * 2 * (P - 1) / P * n_bytes] for P in Ps], dtype=float)
    y = np.array([meas_dict[P] for P in Ps], dtype=float)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha_eff, beta_eff = sol[0], sol[1]
    y_pred = A @ sol
    r2 = r_squared(y, y_pred)
    return alpha_eff, beta_eff, r2

def fit_effective_tree(meas_dict, n_bytes, steps=STEPS_PER_EPOCH):
    """Linear least-squares fit:
      meas(P) = steps * 2*log2(P) * (alpha' + beta' * n)
              = A_P * alpha' + B_P * beta'
    """
    Ps = sorted(p for p in meas_dict.keys() if p >= 2 and not np.isnan(meas_dict[p]))
    if len(Ps) < 2:
        return float('nan'), float('nan'), float('nan')
    A = np.array([[steps * 2 * np.log2(P),
                   steps * 2 * np.log2(P) * n_bytes] for P in Ps], dtype=float)
    y = np.array([meas_dict[P] for P in Ps], dtype=float)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha_eff, beta_eff = sol[0], sol[1]
    y_pred = A @ sol
    r2 = r_squared(y, y_pred)
    return alpha_eff, beta_eff, r2

# ─── CORRECTION B: Per-message gamma overhead with fixed (alpha, beta) ────────
def fit_gamma_ring(meas_dict, alpha, beta, n_bytes, steps=STEPS_PER_EPOCH):
    """Augmented model: meas(P) = steps*[2(P-1)*(alpha+gamma) + 2(P-1)/P*beta*n]
    Solve per-P: gamma_P = meas/(steps*2(P-1)) - alpha - beta*n/P
    Return (gamma_median, list_of_gamma_per_P, list_of_Ps).
    """
    Ps = sorted(p for p in meas_dict.keys() if p >= 2 and not np.isnan(meas_dict[p]))
    gammas = []
    for P in Ps:
        gamma_P = meas_dict[P] / (steps * 2 * (P-1)) - alpha - beta * n_bytes / P
        gammas.append(gamma_P)
    if not gammas:
        return float('nan'), [], []
    return float(np.median(gammas)), gammas, Ps

def fit_gamma_tree(meas_dict, alpha, beta, n_bytes, steps=STEPS_PER_EPOCH):
    """Augmented model: meas(P) = steps * 2*log2(P) * (alpha + gamma + beta*n)
    Solve per-P: gamma_P = meas/(steps*2*log2(P)) - alpha - beta*n
    """
    Ps = sorted(p for p in meas_dict.keys() if p >= 2 and not np.isnan(meas_dict[p]))
    gammas = []
    for P in Ps:
        gamma_P = meas_dict[P] / (steps * 2 * np.log2(P)) - alpha - beta * n_bytes
        gammas.append(gamma_P)
    if not gammas:
        return float('nan'), [], []
    return float(np.median(gammas)), gammas, Ps

# ─── CORRECTION C: Subtract P=1 application overhead ──────────────────────────
def subtract_app_overhead(meas_dict, app_overhead):
    """Return new dict with app_overhead subtracted from each P>=2.
    P=1 itself becomes 0 (its allreduce_s IS the app_overhead by definition)."""
    out = {}
    for P, v in meas_dict.items():
        if P == 1:
            out[P] = 0.0
        else:
            out[P] = v - app_overhead
    return out

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pingpong_path = RESULTS_DIR / "pingpong.csv"
    if not pingpong_path.exists():
        print(f"ERROR: {pingpong_path} not found.")
        return

    print("=" * 65)
    print("Alpha-Beta Model Fitting -- STAGE 0 (ping-pong baseline)")
    print("=" * 65)

    sizes, half_rtts = load_pingpong(pingpong_path)
    print(f"\nLoaded {len(sizes)} message sizes from ping-pong benchmark")
    print(f"Size range:  {sizes[0]:.0f} bytes  to  {sizes[-1]/1e6:.0f} MB")
    print(f"RTT range:   {half_rtts[0]*1e6:.2f} us  to  {half_rtts[-1]*1e6:.2f} us")

    alpha_naive, beta_naive, r2_naive = linear_fit(sizes, half_rtts)
    print(f"\n--- Method 1: Naive linear fit (all sizes) ---")
    print(f"alpha = {alpha_naive*1e6:.4f} us   "
          f"{'<-- NEGATIVE, physically invalid' if alpha_naive < 0 else ''}")
    print(f"beta  = {beta_naive*1e9:.6f} ns/byte  ({1/(beta_naive*1e9):.2f} GB/s)")
    print(f"R^2   = {r2_naive:.6f}")

    alpha = estimate_alpha(sizes, half_rtts, max_size=512)
    beta  = estimate_beta(sizes, half_rtts, alpha, min_size=1e6)
    y_pred = alpha + beta * sizes
    r2_split = r_squared(half_rtts, y_pred)
    print(f"\n--- Method 2: Split fit (alpha from small, beta from large) ---")
    print(f"alpha = {alpha*1e6:.4f} us  (median of messages <= 512 bytes)")
    print(f"beta  = {beta*1e9:.6f} ns/byte  ({1/(beta*1e9):.2f} GB/s)")
    print(f"R^2   = {r2_split:.6f}  (model fit over all message sizes)")
    print(f"\nUsing Method 2 for predictions.")

    # ── Load measured times ───────────────────────────────────────────────────
    ring_measured = load_measured_allreduce("ring_allreduce_nranks*_*.out")
    mpi_measured  = load_measured_allreduce("mpi_allreduce_nranks*_*.out")
    tree_measured = load_measured_allreduce("tree_allreduce_nranks*_*.out")
    print(f"\nLoaded ring measurements for ranks: {sorted(ring_measured.keys())}")
    print(f"Loaded MPI  measurements for ranks: {sorted(mpi_measured.keys())}")
    print(f"Loaded tree measurements for ranks: {sorted(tree_measured.keys())}")

    # ── Stage 0: Predicted vs measured table ──────────────────────────────────
    all_ranks = sorted(set(list(ring_measured.keys()) +
                           list(mpi_measured.keys()) +
                           list(tree_measured.keys())))
    all_ranks_pos = [r for r in all_ranks if r > 1]

    print("\n" + "=" * 105)
    print(f"STAGE 0: Ping-pong a-b prediction vs measured (all P > 1)")
    print("-" * 105)
    print(f"{'nranks':>6} | {'pred_ring':>10} | {'meas_ring':>10} | {'err%':>6} | "
          f"{'pred_tree':>10} | {'meas_tree':>10} | {'err%':>6} | {'meas_mpi':>10}")
    print("-" * 105)

    pred_ring_list, meas_ring_list = [], []
    pred_tree_list, meas_tree_list = [], []
    rows = []

    for nranks in all_ranks_pos:
        pred_ring = predict_ring(alpha, beta, nranks, GRAD_BUF_BYTES)
        pred_tree = predict_tree(alpha, beta, nranks, GRAD_BUF_BYTES)
        meas_ring = ring_measured.get(nranks, float('nan'))
        meas_tree = tree_measured.get(nranks, float('nan'))
        meas_mpi  = mpi_measured.get(nranks, float('nan'))

        err_ring = abs(pred_ring - meas_ring) / meas_ring * 100 \
                   if not np.isnan(meas_ring) else float('nan')
        err_tree = abs(pred_tree - meas_tree) / meas_tree * 100 \
                   if not np.isnan(meas_tree) else float('nan')

        print(f"{nranks:>6} | {pred_ring:>10.4f} | {meas_ring:>10.4f} | "
              f"{err_ring:>5.1f}% | {pred_tree:>10.4f} | {meas_tree:>10.4f} | "
              f"{err_tree:>5.1f}% | {meas_mpi:>10.4f}")

        if not np.isnan(meas_ring) and not np.isnan(pred_ring):
            pred_ring_list.append(pred_ring)
            meas_ring_list.append(meas_ring)
        if not np.isnan(meas_tree) and not np.isnan(pred_tree):
            pred_tree_list.append(pred_tree)
            meas_tree_list.append(meas_tree)

        rows.append({'nranks': nranks,
                     'pred_ring_s': pred_ring,
                     'meas_ring_s': meas_ring,
                     'pred_tree_s': pred_tree,
                     'meas_tree_s': meas_tree,
                     'meas_mpi_s':  meas_mpi})

    r2_ring = float('nan')
    if len(pred_ring_list) >= 2:
        r2_ring = r_squared(meas_ring_list, pred_ring_list)
        print(f"\nR^2 (Stage 0 ring): {r2_ring:.4f}")
    r2_tree = float('nan')
    if len(pred_tree_list) >= 2:
        r2_tree = r_squared(meas_tree_list, pred_tree_list)
        print(f"R^2 (Stage 0 tree): {r2_tree:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # CORRECTION A: Effective (a', b') from collective fitting
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORRECTION A: Effective (a', b') from collective measurements")
    print("=" * 65)
    a_eff_ring, b_eff_ring, r2_eff_ring = fit_effective_ring(ring_measured, GRAD_BUF_BYTES)
    a_eff_tree, b_eff_tree, r2_eff_tree = fit_effective_tree(tree_measured, GRAD_BUF_BYTES)

    print(f"Ping-pong baseline:  a  = {alpha*1e6:7.2f} us   "
          f"b  = {beta*1e9:7.4f} ns/B  ({1/(beta*1e9):.2f} GB/s)")
    print(f"Ring effective:      a' = {a_eff_ring*1e6:7.2f} us   "
          f"b' = {b_eff_ring*1e9:7.4f} ns/B  "
          f"({1/(b_eff_ring*1e9) if b_eff_ring>0 else float('nan'):.2f} GB/s)   "
          f"R^2={r2_eff_ring:.4f}")
    print(f"Tree effective:      a' = {a_eff_tree*1e6:7.2f} us   "
          f"b' = {b_eff_tree*1e9:7.4f} ns/B  "
          f"({1/(b_eff_tree*1e9) if b_eff_tree>0 else float('nan'):.2f} GB/s)   "
          f"R^2={r2_eff_tree:.4f}")
    print(f"\nRatio (effective / ping-pong):")
    print(f"  Ring:  a'/a = {a_eff_ring/alpha:.1f}x    b'/b = {b_eff_ring/beta:.2f}x")
    print(f"  Tree:  a'/a = {a_eff_tree/alpha:.1f}x    b'/b = {b_eff_tree/beta:.2f}x")
    print(f"\nInterpretation:")
    print(f"  a'/a >> 1 -> ping-pong a misses MPI library + topology overhead")
    print(f"  b'/b ~ 1  -> bandwidth at near-link-rate (model captures BW correctly)")
    print(f"  b'/b >> 1 -> bandwidth bottleneck beyond link (e.g. NIC contention)")

    # ──────────────────────────────────────────────────────────────────────────
    # CORRECTION B: g per-message overhead with fixed (a, b)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORRECTION B: g per-message overhead (fixed ping-pong a, b)")
    print("=" * 65)
    gamma_ring, gammas_ring, Ps_ring_g = fit_gamma_ring(ring_measured, alpha, beta, GRAD_BUF_BYTES)
    gamma_tree, gammas_tree, Ps_tree_g = fit_gamma_tree(tree_measured, alpha, beta, GRAD_BUF_BYTES)

    print(f"Ring: g_median = {gamma_ring*1e6:.1f} us (per MPI_Sendrecv call)")
    for P, g in zip(Ps_ring_g, gammas_ring):
        print(f"  P={P:>3d}: g_P = {g*1e6:>7.1f} us")
    print(f"\nTree: g_median = {gamma_tree*1e6:.1f} us (per MPI_Send/Recv pair + accumulate)")
    for P, g in zip(Ps_tree_g, gammas_tree):
        print(f"  P={P:>3d}: g_P = {g*1e6:>7.1f} us")

    if not np.isnan(gamma_ring) and not np.isnan(gamma_tree):
        print(f"\ng_tree / g_ring = {gamma_tree/gamma_ring:.2f}x")
        print(f"  Tree g being larger than ring g is expected: tree's reduce phase")
        print(f"  includes a per-step accumulate loop (for i: buf[i] += recv_buf[i])")
        print(f"  on rank 0, which adds ~0.5-1 ms per step beyond raw MPI overhead.")

    # ──────────────────────────────────────────────────────────────────────────
    # CORRECTION C: Subtract P=1 application overhead
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORRECTION C: Subtract P=1 application overhead")
    print("=" * 65)
    app_ring = ring_measured.get(1, 0.0)
    app_tree = tree_measured.get(1, 0.0)
    app_mpi  = mpi_measured.get(1, 0.0)

    print(f"\nP=1 allreduce_s (= application overhead, no MPI work):")
    print(f"  Ring: {app_ring:.4f} s")
    print(f"  Tree: {app_tree:.4f} s")
    print(f"  MPI:  {app_mpi:.4f} s")

    asymmetry_warning = False
    if abs(app_ring - app_tree) > 0.05 or abs(app_ring - app_mpi) > 0.05:
        asymmetry_warning = True
        print(f"\n!! WARNING: P=1 allreduce_s is NOT consistent across implementations!")
        print(f"  This indicates the three .cpp files time DIFFERENT regions:")
        print(f"  - ring_allreduce.cpp  : may exclude pack/divide (P=1 ~ 0s)")
        print(f"  - mpi_allreduce.cpp   : includes pack/divide (P=1 ~ 0.37s)")
        print(f"  - tree_allreduce.cpp  : includes pack/divide (P=1 ~ 0.36s)")
        print(f"  -> For fair Stage 0 comparison, ring's measured time effectively")
        print(f"    already EQUALS its 'pure MPI' time, while tree/MPI need subtraction.")

    pure_ring = subtract_app_overhead(ring_measured, app_ring)
    pure_tree = subtract_app_overhead(tree_measured, app_tree)
    pure_mpi  = subtract_app_overhead(mpi_measured,  app_mpi)

    print(f"\nPure MPI time per epoch (after subtracting per-impl P=1 baseline):")
    print(f"{'P':>4} | {'pure_ring':>10} | {'pure_tree':>10} | {'pure_mpi':>10}")
    for P in all_ranks_pos:
        r = pure_ring.get(P, float('nan'))
        t = pure_tree.get(P, float('nan'))
        m = pure_mpi.get(P, float('nan'))
        print(f"{P:>4} | {r:>10.4f} | {t:>10.4f} | {m:>10.4f}")

    # Update CSV rows with pure-MPI columns
    pure_lookup = {r['nranks']: r for r in rows}
    for P in all_ranks_pos:
        if P in pure_lookup:
            pure_lookup[P]['pure_ring_s'] = pure_ring.get(P, float('nan'))
            pure_lookup[P]['pure_tree_s'] = pure_tree.get(P, float('nan'))
            pure_lookup[P]['pure_mpi_s']  = pure_mpi.get(P, float('nan'))

    # Re-evaluate ping-pong predictions vs PURE measurements
    print(f"\nPing-pong a-b prediction vs PURE (post-correction-C) measurements:")
    print(f"{'P':>4} | {'pred_ring':>10} | {'pure_ring':>10} | {'err%':>6} | "
          f"{'pred_tree':>10} | {'pure_tree':>10} | {'err%':>6}")
    pred_pure_ring, meas_pure_ring = [], []
    pred_pure_tree, meas_pure_tree = [], []
    for P in all_ranks_pos:
        pred_r = predict_ring(alpha, beta, P, GRAD_BUF_BYTES)
        pred_t = predict_tree(alpha, beta, P, GRAD_BUF_BYTES)
        pure_r = pure_ring.get(P, float('nan'))
        pure_t = pure_tree.get(P, float('nan'))
        err_r = abs(pred_r - pure_r) / pure_r * 100 if (not np.isnan(pure_r) and pure_r > 0) else float('nan')
        err_t = abs(pred_t - pure_t) / pure_t * 100 if (not np.isnan(pure_t) and pure_t > 0) else float('nan')
        print(f"{P:>4} | {pred_r:>10.4f} | {pure_r:>10.4f} | {err_r:>5.1f}% | "
              f"{pred_t:>10.4f} | {pure_t:>10.4f} | {err_t:>5.1f}%")
        if not np.isnan(pure_r) and pure_r > 0:
            pred_pure_ring.append(pred_r); meas_pure_ring.append(pure_r)
        if not np.isnan(pure_t) and pure_t > 0:
            pred_pure_tree.append(pred_t); meas_pure_tree.append(pure_t)

    r2_pure_ring = r_squared(meas_pure_ring, pred_pure_ring) if len(meas_pure_ring) >= 2 else float('nan')
    r2_pure_tree = r_squared(meas_pure_tree, pred_pure_tree) if len(meas_pure_tree) >= 2 else float('nan')
    print(f"\nR^2 (ping-pong a-b vs PURE ring): {r2_pure_ring:>8.4f}  (was {r2_ring:.4f})")
    print(f"R^2 (ping-pong a-b vs PURE tree): {r2_pure_tree:>8.4f}  (was {r2_tree:.4f})")

    # Re-fit effective (a', b') on pure data
    print(f"\nRe-fit effective (a', b') on PURE data:")
    a_eff_ring_C, b_eff_ring_C, r2_eff_ring_C = fit_effective_ring(pure_ring, GRAD_BUF_BYTES)
    a_eff_tree_C, b_eff_tree_C, r2_eff_tree_C = fit_effective_tree(pure_tree, GRAD_BUF_BYTES)
    print(f"Ring (pure): a' = {a_eff_ring_C*1e6:.2f} us, "
          f"b' = {b_eff_ring_C*1e9:.4f} ns/B, R^2={r2_eff_ring_C:.4f}")
    print(f"Tree (pure): a' = {a_eff_tree_C*1e6:.2f} us, "
          f"b' = {b_eff_tree_C*1e9:.4f} ns/B, R^2={r2_eff_tree_C:.4f}")

    # ── Crossover analysis (existing, unchanged) ──────────────────────────────
    print(f"\n--- Crossover Analysis (comm time as % of backward pass) ---")
    bwd_by_rank = {2:92.1, 4:46.2, 8:23.3, 16:11.8, 32:6.07}
    for nranks in all_ranks_pos:
        bwd  = bwd_by_rank.get(nranks, float('nan'))
        ring = ring_measured.get(nranks, float('nan'))
        tree = tree_measured.get(nranks, float('nan'))
        mpi  = mpi_measured.get(nranks, float('nan'))
        if not np.isnan(bwd):
            line = f"  nranks={nranks:2d}: "
            if not np.isnan(ring): line += f"ring={ring/bwd*100:5.1f}%  "
            if not np.isnan(tree): line += f"tree={tree/bwd*100:5.1f}%  "
            if not np.isnan(mpi):  line += f"mpi={mpi/bwd*100:5.1f}%"
            print(line)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Summary for Report")
    print(f"{'='*65}")
    print(f"Network: Perlmutter Slingshot-11")
    print(f"Ping-pong a: {alpha*1e6:.2f} us, b: {beta*1e9:.4f} ns/B "
          f"({1/(beta*1e9):.2f} GB/s)")
    print(f"")
    print(f"Stage 0 (ping-pong a-b predicting collective):")
    print(f"  R^2 (ring) = {r2_ring:.4f}")
    print(f"  R^2 (tree) = {r2_tree:.4f}")
    print(f"")
    print(f"Correction A (effective a', b' fit from collective):")
    print(f"  Ring: a' = {a_eff_ring*1e6:.2f} us  ({a_eff_ring/alpha:.1f}x ping-pong a)")
    print(f"        b' = {b_eff_ring*1e9:.4f} ns/B ({b_eff_ring/beta:.2f}x ping-pong b)")
    print(f"  Tree: a' = {a_eff_tree*1e6:.2f} us  ({a_eff_tree/alpha:.1f}x ping-pong a)")
    print(f"        b' = {b_eff_tree*1e9:.4f} ns/B ({b_eff_tree/beta:.2f}x ping-pong b)")
    print(f"")
    print(f"Correction B (g per-message overhead):")
    print(f"  g_ring (median) = {gamma_ring*1e6:.1f} us")
    print(f"  g_tree (median) = {gamma_tree*1e6:.1f} us")
    print(f"")
    print(f"Correction C (subtract P=1 app overhead):")
    print(f"  app_ring = {app_ring:.4f}s, app_tree = {app_tree:.4f}s, app_mpi = {app_mpi:.4f}s")
    if asymmetry_warning:
        print(f"  !! Implementations time different regions -- see warning above")
    print(f"  R^2 (after C, ring) = {r2_pure_ring:.4f}")
    print(f"  R^2 (after C, tree) = {r2_pure_tree:.4f}")

    # ── Save extended CSV ─────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "alpha_beta_plot.csv"
    csv_fields = ['nranks', 'pred_ring_s', 'meas_ring_s',
                  'pred_tree_s', 'meas_tree_s', 'meas_mpi_s',
                  'pure_ring_s', 'pure_tree_s', 'pure_mpi_s']
    # Ensure all rows have the new keys
    for r in rows:
        for k in ['pure_ring_s', 'pure_tree_s', 'pure_mpi_s']:
            r.setdefault(k, float('nan'))
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPlot data saved to: {out_csv}")

    # ── Save extended TXT report ──────────────────────────────────────────────
    out_txt = RESULTS_DIR / "alpha_beta_fit.txt"
    with open(out_txt, 'w') as f:
        f.write("Alpha-Beta Model Fit -- Perlmutter Slingshot-11\n")
        f.write("=" * 60 + "\n\n")
        f.write("STAGE 0: Ping-pong baseline (split fit)\n")
        f.write(f"  alpha = {alpha*1e6:.4f} us\n")
        f.write(f"  beta  = {beta*1e9:.6f} ns/byte ({1/(beta*1e9):.2f} GB/s)\n")
        f.write(f"  R^2 ping-pong fit:   {r2_split:.6f}\n")
        f.write(f"  R^2 ring prediction: {r2_ring:.6f}\n")
        f.write(f"  R^2 tree prediction: {r2_tree:.6f}\n\n")

        f.write("Ring: predicted vs measured allreduce_s per epoch\n")
        f.write(f"{'nranks':>8} {'pred':>12} {'meas':>12} {'err%':>8}\n")
        for r in rows:
            if r['nranks'] > 1 and not np.isnan(r['meas_ring_s']):
                err = abs(r['pred_ring_s']-r['meas_ring_s'])/r['meas_ring_s']*100
                f.write(f"{r['nranks']:>8} {r['pred_ring_s']:>12.4f} "
                        f"{r['meas_ring_s']:>12.4f} {err:>7.1f}%\n")
        f.write("\nTree: predicted vs measured allreduce_s per epoch\n")
        f.write(f"{'nranks':>8} {'pred':>12} {'meas':>12} {'err%':>8}\n")
        for r in rows:
            if r['nranks'] > 1 and not np.isnan(r['meas_tree_s']):
                err = abs(r['pred_tree_s']-r['meas_tree_s'])/r['meas_tree_s']*100
                f.write(f"{r['nranks']:>8} {r['pred_tree_s']:>12.4f} "
                        f"{r['meas_tree_s']:>12.4f} {err:>7.1f}%\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("CORRECTION A: Effective (alpha', beta') from collective\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Ring: alpha' = {a_eff_ring*1e6:.2f} us, "
                f"beta' = {b_eff_ring*1e9:.4f} ns/B, R^2 = {r2_eff_ring:.4f}\n")
        f.write(f"        ratio: alpha'/alpha = {a_eff_ring/alpha:.1f}x, "
                f"beta'/beta = {b_eff_ring/beta:.2f}x\n")
        f.write(f"  Tree: alpha' = {a_eff_tree*1e6:.2f} us, "
                f"beta' = {b_eff_tree*1e9:.4f} ns/B, R^2 = {r2_eff_tree:.4f}\n")
        f.write(f"        ratio: alpha'/alpha = {a_eff_tree/alpha:.1f}x, "
                f"beta'/beta = {b_eff_tree/beta:.2f}x\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("CORRECTION B: gamma per-message overhead\n")
        f.write("=" * 60 + "\n")
        f.write(f"  gamma_ring (median) = {gamma_ring*1e6:.1f} us\n")
        for P, g in zip(Ps_ring_g, gammas_ring):
            f.write(f"    P={P:>3d}: gamma_P = {g*1e6:>7.1f} us\n")
        f.write(f"  gamma_tree (median) = {gamma_tree*1e6:.1f} us\n")
        for P, g in zip(Ps_tree_g, gammas_tree):
            f.write(f"    P={P:>3d}: gamma_P = {g*1e6:>7.1f} us\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("CORRECTION C: Subtract P=1 application overhead\n")
        f.write("=" * 60 + "\n")
        f.write(f"  P=1 baseline (= application overhead):\n")
        f.write(f"    Ring: {app_ring:.4f} s\n")
        f.write(f"    Tree: {app_tree:.4f} s\n")
        f.write(f"    MPI:  {app_mpi:.4f} s\n")
        if asymmetry_warning:
            f.write(f"  WARNING: Asymmetric P=1 baselines indicate the three\n")
            f.write(f"  implementations time DIFFERENT regions of code.\n")
            f.write(f"  ring_allreduce.cpp likely excludes pack/divide overhead;\n")
            f.write(f"  tree_allreduce.cpp and mpi_allreduce.cpp include it.\n")
        f.write(f"\n  After C -- pure MPI time:\n")
        f.write(f"  {'P':>4}  {'pure_ring':>10}  {'pure_tree':>10}  {'pure_mpi':>10}\n")
        for P in all_ranks_pos:
            r = pure_ring.get(P, float('nan'))
            t = pure_tree.get(P, float('nan'))
            m = pure_mpi.get(P, float('nan'))
            f.write(f"  {P:>4}  {r:>10.4f}  {t:>10.4f}  {m:>10.4f}\n")
        f.write(f"\n  R^2 (ping-pong vs PURE ring) = {r2_pure_ring:.4f}  "
                f"(was {r2_ring:.4f})\n")
        f.write(f"  R^2 (ping-pong vs PURE tree) = {r2_pure_tree:.4f}  "
                f"(was {r2_tree:.4f})\n")
        f.write(f"\n  Effective (alpha', beta') re-fit on PURE data:\n")
        f.write(f"    Ring: alpha' = {a_eff_ring_C*1e6:.2f} us, "
                f"beta' = {b_eff_ring_C*1e9:.4f} ns/B, R^2={r2_eff_ring_C:.4f}\n")
        f.write(f"    Tree: alpha' = {a_eff_tree_C*1e6:.2f} us, "
                f"beta' = {b_eff_tree_C*1e9:.4f} ns/B, R^2={r2_eff_tree_C:.4f}\n")

    print(f"Report saved to: {out_txt}")

if __name__ == "__main__":
    main()
