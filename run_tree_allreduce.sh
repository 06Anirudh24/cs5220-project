#!/bin/bash
# run_tree_allreduce.sh — Compile and run hand-written tree all-reduce across rank counts
#
# Usage:  bash run_tree_allreduce.sh
# Output: results/tree_allreduce_nranks<N>_<timestamp>.out for each rank count
#
# NOTE: rank counts must be powers of 2 — tree_allreduce aborts otherwise.
# For scaling experiments use: (1 2 4 8 16 32)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"  # binaries use relative DATA_DIR, so CWD must be project root
SRC="$PROJECT_DIR/tree_allreduce.cpp"
BIN="$PROJECT_DIR/tree_allreduce"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUT_DIR"

# ── Compile once ──────────────────────────────────────────────────────────────
echo "=== Compiling ==="
mpicxx -O2 -std=c++17 "$SRC" -o "$BIN"
echo "Compiled: $BIN"
echo ""

# ── Rank counts to sweep (powers of 2 only) ───────────────────────────────────
RANK_COUNTS=(1 2 4 8 16)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for NRANKS in "${RANK_COUNTS[@]}"; do
    OUTFILE="$OUT_DIR/tree_allreduce_nranks${NRANKS}_${TIMESTAMP}.out"

    echo "=== Starting run: nranks=$NRANKS ===" | tee "$OUTFILE"
    echo "date:        $(date)"                  | tee -a "$OUTFILE"
    echo "host:        $(hostname)"              | tee -a "$OUTFILE"
    echo "nranks:      $NRANKS"                  | tee -a "$OUTFILE"
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

echo "All runs complete. Results in: $OUT_DIR"
