#!/bin/bash
#SBATCH --job-name=pingpong
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/a/anirudh6/cs5220/project/results/slurm_pingpong_%j.out

cd /pscratch/sd/a/anirudh6/cs5220/project

PROJECT_DIR="/pscratch/sd/a/anirudh6/cs5220/project"
SRC="$PROJECT_DIR/pingpong.cpp"
BIN="$PROJECT_DIR/pingpong"

echo "=== Compiling ==="
mpicxx -O2 -std=c++17 "$SRC" -o "$BIN"
if [ ! -f "$BIN" ]; then
    echo "ERROR: Compilation failed, binary not found. Exiting."
    exit 1
fi
echo "Compiled: $BIN"
echo ""

echo "=== Running ping-pong benchmark ==="
echo "date:   $(date)"
echo "host:   $(hostname)"
echo "nodes:  2 (cross-node, --ntasks-per-node=1)"
echo ""

srun --ntasks=2 --ntasks-per-node=1 "$BIN"

echo ""
echo "=== Done ==="
echo "date: $(date)"
echo "Results saved to: $PROJECT_DIR/results/pingpong.csv"
