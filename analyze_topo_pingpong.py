#!/usr/bin/env python3
"""
analyze_topo_pingpong.py — Analyze topology ping-pong results to quantify
Dragonfly topology effects on point-to-point communication.

Reads:  results/topo_pingpong_nranks*.csv
Prints: per-rank-count summary of latency and bandwidth across all pairs,
        showing whether RTT increases as ranks span more Dragonfly groups.

Usage: python3 analyze_topo_pingpong.py
"""

import csv
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR  = Path("/pscratch/sd/a/anirudh6/cs5220/project/results")
GRAD_BUF_MSG = 11665160   # bytes — matches our gradient buffer

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'nranks':     int(row['nranks']),
                'pair_id':    int(row['pair_id']),
                'rank_a':     int(row['rank_a']),
                'rank_b':     int(row['rank_b']),
                'node_a':     row['node_a'].strip(),
                'node_b':     row['node_b'].strip(),
                'bytes':      int(row['bytes']),
                'half_rtt_s': float(row['half_rtt_s']),
                'half_rtt_us':float(row['half_rtt_us']),
                'bw_GBps':    float(row['bandwidth_GBps']),
                'min_rtt_s':  float(row['min_rtt_s']),
                'stddev_s':   float(row['stddev_s']),
            })
    return rows

def main():
    files = sorted(glob.glob(str(RESULTS_DIR / "topo_pingpong_nranks*.csv")))
    if not files:
        print("ERROR: No topo_pingpong_nranks*.csv files found in results/")
        print("Run submit_topo_pingpong.sh first.")
        return

    print("=" * 70)
    print("Topology Ping-Pong Analysis — Dragonfly Effects on Perlmutter")
    print("=" * 70)

    all_rows = []
    for f in files:
        all_rows.extend(load_csv(f))
    print(f"\nLoaded {len(all_rows)} rows from {len(files)} files.")

    # ── Per rank count summary ─────────────────────────────────────────────────
    print("\n--- Latency (alpha) at small messages (64 bytes) ---")
    print(f"{'nranks':>8} {'pairs':>7} {'min_us':>10} {'mean_us':>10} "
          f"{'max_us':>10} {'stddev_us':>12} {'spread':>10}")
    print("-" * 70)

    by_nranks = defaultdict(list)
    for row in all_rows:
        by_nranks[row['nranks']].append(row)

    latency_by_nranks = {}   # nranks -> mean half_rtt_us at 64 bytes
    bw_by_nranks = {}        # nranks -> mean bw_GBps at grad_buf size

    for nranks in sorted(by_nranks.keys()):
        rows = by_nranks[nranks]

        # Latency: 64-byte messages
        lat_rows = [r for r in rows if r['bytes'] == 64]
        if lat_rows:
            lats = [r['half_rtt_us'] for r in lat_rows]
            latency_by_nranks[nranks] = np.mean(lats)
            spread = (max(lats) - min(lats)) / np.mean(lats) * 100
            print(f"{nranks:>8} {len(lat_rows):>7} {min(lats):>10.2f} "
                  f"{np.mean(lats):>10.2f} {max(lats):>10.2f} "
                  f"{np.std(lats):>12.2f} {spread:>9.1f}%")

    print("\n--- Bandwidth at grad_buf size (~11.1 MB) ---")
    print(f"{'nranks':>8} {'pairs':>7} {'min_GBps':>10} {'mean_GBps':>11} "
          f"{'max_GBps':>10} {'spread':>10}")
    print("-" * 65)

    for nranks in sorted(by_nranks.keys()):
        rows = by_nranks[nranks]
        bw_rows = [r for r in rows if r['bytes'] == GRAD_BUF_MSG]
        if bw_rows:
            bws = [r['bw_GBps'] for r in bw_rows]
            bw_by_nranks[nranks] = np.mean(bws)
            spread = (max(bws) - min(bws)) / np.mean(bws) * 100
            print(f"{nranks:>8} {len(bw_rows):>7} {min(bws):>10.3f} "
                  f"{np.mean(bws):>11.3f} {max(bws):>10.3f} {spread:>9.1f}%")

    # ── Key question: does latency increase with nranks? ──────────────────────
    print("\n--- Topology Effect Summary ---")
    print("If latency grows with nranks, ranks are spanning Dragonfly group")
    print("boundaries, causing inter-group routing overhead.\n")
    print(f"{'nranks':>8} {'lat_us':>10} {'lat_increase':>14} {'bw_GBps':>10} {'bw_decrease':>13}")
    print("-" * 60)

    base_lat = latency_by_nranks.get(2, None)
    base_bw  = bw_by_nranks.get(2, None)

    for nranks in sorted(latency_by_nranks.keys()):
        lat = latency_by_nranks[nranks]
        bw  = bw_by_nranks.get(nranks, float('nan'))
        lat_inc = f"+{(lat/base_lat - 1)*100:.1f}%" if base_lat and nranks > 2 else "baseline"
        bw_dec  = f"-{(1 - bw/base_bw)*100:.1f}%"  if base_bw  and nranks > 2 else "baseline"
        print(f"{nranks:>8} {lat:>10.2f} {lat_inc:>14} {bw:>10.3f} {bw_dec:>13}")

    # ── Quantify the gap in alpha-beta prediction ──────────────────────────────
    print("\n--- Gap Analysis: Predicted vs Measured Ring Allreduce ---")
    print("Using per-nranks alpha/beta to re-predict ring allreduce time.")
    print("Compare to your measured ring allreduce_s values.\n")

    # From your existing results
    meas_ring = {2: 0.2461, 4: 0.5307, 8: 0.6618, 16: 0.8827, 32: 1.0653}
    steps = 234
    m = GRAD_BUF_MSG

    print(f"{'nranks':>8} {'alpha_us':>10} {'bw_GBps':>10} "
          f"{'pred_ring_s':>13} {'meas_ring_s':>13} {'err%':>8}")
    print("-" * 65)

    for nranks in sorted(latency_by_nranks.keys()):
        if nranks < 2:
            continue
        alpha = latency_by_nranks[nranks] * 1e-6   # us -> seconds
        bw    = bw_by_nranks.get(nranks, float('nan'))
        beta  = 1.0 / (bw * 1e9) if bw > 0 else float('nan')

        # Ring formula: T = 2*(P-1)*alpha + 2*(P-1)/P * beta * m
        pred = (2*(nranks-1)*alpha + 2*(nranks-1)/nranks * beta * m) * steps
        meas = meas_ring.get(nranks, float('nan'))
        err  = (pred - meas) / meas * 100 if not np.isnan(meas) else float('nan')

        print(f"{nranks:>8} {latency_by_nranks[nranks]:>10.2f} {bw:>10.3f} "
              f"{pred:>13.4f} {meas:>13.4f} {err:>7.1f}%")

    print("\nIf err% is now smaller (closer to 0) than your original analysis,")
    print("it confirms that topology-varying alpha/beta explains the gap.")
    print("If err% is still large, other effects dominate (MPI call overhead,")
    print("contention from simultaneous communication, etc.)\n")

    # ── Per-pair detail for largest message ───────────────────────────────────
    print("--- Per-pair RTT detail at grad_buf size (11.1 MB), nranks=32 ---")
    print("(Shows variability across pairs — high variance = topology effects)\n")
    print(f"{'pair':>6} {'node_a':>12} {'node_b':>12} {'half_rtt_us':>13} {'bw_GBps':>10}")
    print("-" * 58)

    detail = [r for r in all_rows
              if r['nranks'] == max(by_nranks.keys())
              and r['bytes'] == GRAD_BUF_MSG]
    detail.sort(key=lambda r: r['pair_id'])
    for r in detail:
        na = r['node_a'][-8:]  # last 8 chars of hostname
        nb = r['node_b'][-8:]
        print(f"{r['pair_id']:>6} {na:>12} {nb:>12} "
              f"{r['half_rtt_us']:>13.2f} {r['bw_GBps']:>10.3f}")

    print("\n=== Analysis complete ===")
    print("Key questions to answer from these results:")
    print("1. Does mean latency increase with nranks? (Dragonfly group spanning)")
    print("2. Is there high variance across pairs at large nranks? (non-uniform topology)")
    print("3. Does using per-nranks alpha/beta reduce the prediction error?")

if __name__ == "__main__":
    main()
