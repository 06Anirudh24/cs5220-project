#!/bin/bash
#SBATCH --job-name=tree_allreduce
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_tree_%j.out

# $SLURM_SUBMIT_DIR is where you ran `sbatch` — submit from the project root.
cd "$SLURM_SUBMIT_DIR"
bash run_tree_allreduce.sh
