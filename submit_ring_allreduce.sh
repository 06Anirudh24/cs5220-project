#!/bin/bash
#SBATCH --job-name=ring_allreduce
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/pscratch/sd/a/anirudh6/cs5220/project/results/slurm_ring_%j.out

cd /pscratch/sd/a/anirudh6/cs5220/project

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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RANK_COUNTS=(8 16 32)

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
         --nodes="$NRANKS" \
         "$BIN" 2>&1 | tee -a "$OUTFILE"

    echo ""                             | tee -a "$OUTFILE"
    echo "=== Done: nranks=$NRANKS ===" | tee -a "$OUTFILE"
    echo "Saved: $OUTFILE"
    echo ""
done

echo "All ring runs complete. Results in: $OUT_DIR"
