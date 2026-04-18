#!/bin/bash
# run_mpi_allreduce.sh — Compile and run MPI_Allreduce MLP across rank counts
#
# Usage:  bash run_mpi_allreduce.sh
# Output: results/mpi_allreduce_nranks<N>_<timestamp>.out for each rank count
#
# This script runs the same binary at 1, 2, 4, 8, 16, 32 ranks.
# Each run gets its own .out file so results are easy to compare.
# Edit RANK_COUNTS below to run only specific counts during development.

set -e

PROJECT_DIR="/pscratch/sd/a/anirudh6/cs5220/project"
SRC="$PROJECT_DIR/mpi_allreduce.cpp"
BIN="$PROJECT_DIR/mpi_allreduce"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUT_DIR"

# ── Compile once ──────────────────────────────────────────────────────────────
echo "=== Compiling ==="
mpicxx -O2 -std=c++17 "$SRC" -o "$BIN"
echo "Compiled: $BIN"
echo ""

# ── Rank counts to sweep ───────────────────────────────────────────────────────
# For initial correctness testing just use 1 and 2.
# For scaling experiments use the full list: 1 2 4 8 16 32
RANK_COUNTS=(1 2 4 8)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for NRANKS in "${RANK_COUNTS[@]}"; do
    OUTFILE="$OUT_DIR/mpi_allreduce_nranks${NRANKS}_${TIMESTAMP}.out"

    echo "=== Starting run: nranks=$NRANKS ===" | tee "$OUTFILE"
    echo "date:        $(date)"                  | tee -a "$OUTFILE"
    echo "host:        $(hostname)"              | tee -a "$OUTFILE"
    echo "nranks:      $NRANKS"                  | tee -a "$OUTFILE"
    echo "binary:      $BIN"                     | tee -a "$OUTFILE"
    echo "outfile:     $OUTFILE"                 | tee -a "$OUTFILE"
    echo ""                                      | tee -a "$OUTFILE"

    echo "=== Training ===" | tee -a "$OUTFILE"

    # srun is the Perlmutter job launcher (SLURM)
    # --ntasks-per-node=1 forces each rank onto its own node (cross-node comms)
    # Remove --ntasks-per-node to run multiple ranks on one node for quick tests
    srun --ntasks="$NRANKS" \
         --ntasks-per-node=1 \
         "$BIN" 2>&1 | tee -a "$OUTFILE"

    echo ""                             | tee -a "$OUTFILE"
    echo "=== Done: nranks=$NRANKS ===" | tee -a "$OUTFILE"
    echo "Saved: $OUTFILE"
    echo ""
done

echo "All runs complete. Results in: $OUT_DIR"
