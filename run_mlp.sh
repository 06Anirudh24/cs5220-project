#!/bin/bash
# run_mlp.sh — Compile and run single-node MLP, saving output to timestamped .out file
#
# Usage:  bash run_mlp.sh
# Output: results/mlp_baseline_<timestamp>.out

set -e

PROJECT_DIR="/pscratch/sd/a/anirudh6/cs5220/project"
SRC="$PROJECT_DIR/mlp.cpp"
BIN="$PROJECT_DIR/mlp"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="$OUT_DIR/mlp_baseline_${TIMESTAMP}.out"

echo "=== Compiling ===" | tee "$OUTFILE"
g++ -O2 -std=c++17 "$SRC" -o "$BIN" 2>&1 | tee -a "$OUTFILE"
echo "Compiled: $BIN" | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "=== Run info ===" | tee -a "$OUTFILE"
echo "date:      $(date)"          | tee -a "$OUTFILE"
echo "host:      $(hostname)"      | tee -a "$OUTFILE"
echo "ranks:     1 (single-node)"  | tee -a "$OUTFILE"
echo "binary:    $BIN"             | tee -a "$OUTFILE"
echo "outfile:   $OUTFILE"         | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "=== Training ===" | tee -a "$OUTFILE"
"$BIN" 2>&1 | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "=== Done ===" | tee -a "$OUTFILE"
echo "Results saved to: $OUTFILE"
