#!/bin/bash
#SBATCH --job-name=mpi_allreduce
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/pscratch/sd/a/anirudh6/cs5220/project/results/slurm_%j.out

cd /pscratch/sd/a/anirudh6/cs5220/project
bash run_mpi_allreduce.sh