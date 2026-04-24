#!/bin/bash
# run_ring_allreduce.sh — Compile and run ring all-reduce MLP across rank counts
#
# Usage:  bash run_ring_allreduce.sh
# Output: results/ring_allreduce_nranks<N>_<timestamp>.out
#
# Run this inside a salloc session with enough nodes, e.g.:
#   salloc --nodes=8 --ntasks-per-node=1 --constraint=cpu \
#          --qos=interactive --time=02:00:00 --account=m4341

set -e

PROJECT_DIR="/pscratch/sd/a/anirudh6/cs5220/project"
SRC="$PROJECT_DIR/ring_allreduce.cpp"
BIN="$PROJECT_DIR/ring_allreduce"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUT_DIR"

# ── Compile ───────────────────────────────────────────────────────────────────
echo "=== Compiling ==="
mpicxx -O2 -std=c++17 "$SRC" -o "$BIN"
echo "Compiled: $BIN"
echo ""

# ── Rank counts to sweep ──────────────────────────────────────────────────────
# Start with 1 2 4 to verify correctness, then add 8 16 32 for scaling
RANK_COUNTS=(1 2 4 8 16 32)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for NRANKS in "${RANK_COUNTS[@]}"; do
    OUTFILE="$OUT_DIR/ring_allreduce_nranks${NRANKS}_${TIMESTAMP}.out"

    echo "=== Starting run: nranks=$NRANKS ===" | tee "$OUTFILE"
    echo "date:        $(date)"                  | tee -a "$OUTFILE"
    echo "host:        $(hostname)"              | tee -a "$OUTFILE"
    echo "nranks:      $NRANKS"                  | tee -a "$OUTFILE"
    echo "algorithm:   ring_allreduce"           | tee -a "$OUTFILE"
    echo "binary:      $BIN"                     | tee -a "$OUTFILE"
    echo "outfile:     $OUTFILE"                 | tee -a "$OUTFILE"
    echo ""                                      | tee -a "$OUTFILE"

    echo "=== Training ===" | tee -a "$OUTFILE"

    srun --ntasks="$NRANKS" \
         --ntasks-per-node=1 \
         "$BIN" 2>&1 | tee -a "$OUTFILE"

    echo ""                             | tee -a "$OUTFILE"
    echo "=== Done: nranks=$NRANKS ===" | tee -a "$OUTFILE"
    echo "Saved: $OUTFILE"
    echo ""
done

echo "All ring runs complete. Results in: $OUT_DIR"
