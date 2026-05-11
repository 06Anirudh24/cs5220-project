#!/bin/bash
#SBATCH --job-name=topo_pingpong
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/pscratch/sd/a/anirudh6/cs5220/project/results/slurm_topo_pingpong_%j.out

cd /pscratch/sd/a/anirudh6/cs5220/project

PROJECT_DIR="/pscratch/sd/a/anirudh6/cs5220/project"
SRC="$PROJECT_DIR/topo_pingpong.cpp"
BIN="$PROJECT_DIR/topo_pingpong"
OUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUT_DIR"

# ── Compile ───────────────────────────────────────────────────────────────────
echo "=== Compiling ==="
mpicxx -O2 -std=c++17 "$SRC" -o "$BIN"
if [ ! -f "$BIN" ]; then
    echo "ERROR: Compilation failed. Exiting."
    exit 1
fi
echo "Compiled: $BIN"
echo ""

# ── Run at increasing rank counts ─────────────────────────────────────────────
# The key insight: at 2 ranks we likely use 1 Dragonfly group,
# at higher ranks we span more groups, increasing inter-group traffic.
# By comparing RTTs across rank counts we can see the topology effect.
RANK_COUNTS=(2 4 8 16 32)

for NRANKS in "${RANK_COUNTS[@]}"; do
    echo "========================================"
    echo "Running topo_pingpong with nranks=$NRANKS"
    echo "date: $(date)"
    echo "========================================"

    srun --ntasks="$NRANKS" \
         --ntasks-per-node=1 \
         --nodes="$NRANKS" \
         "$BIN"

    echo ""
    echo "Done: nranks=$NRANKS"
    echo ""
done

echo "=== All runs complete ==="
echo "Results in: $OUT_DIR/topo_pingpong_nranks*.csv"
echo "Next step: python3 analyze_topo_pingpong.py"
