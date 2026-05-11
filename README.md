# CS 5220 — Parallel Neural Network Training with MPI

**Cornell University, Spring 2026**  
Yuting Chen & Anirudh Atmakuru

Data-parallel MLP training on MNIST benchmarked across three gradient
aggregation strategies — `MPI_Allreduce`, hand-written ring all-reduce,
and hand-written binary tree all-reduce — on Perlmutter's CPU partition
at NERSC.

---

## Key Results

| Algorithm | Speedup (32 ranks) | Efficiency (32 ranks) | Comm% (32 ranks) |
|-----------|-------------------|----------------------|-----------------|
| Ring      | 23.5×             | 73%                  | 10.2%           |
| MPI_Allreduce | 22.4×         | 70%                  | 21.5%           |
| Tree      | 19.9×             | 62%                  | 30.4%           |

Ring all-reduce is 2–4× faster in communication time than MPI_Allreduce
and 2.6–5.1× faster than tree across all tested rank counts.

---

## Repository Structure
├── grad_packed.cpp          # single-node baseline (flat gradient buffer)
├── mpi_allreduce.cpp        # distributed training via MPI_Allreduce
├── ring_allreduce.cpp       # hand-written ring all-reduce
├── tree_allreduce.cpp       # hand-written binary tree all-reduce
├── pingpong.cpp             # alpha-beta network benchmark
├── topo_pingpong.cpp        # multi-pair topology-aware benchmark
├── fit_alpha_beta.py        # alpha-beta model fitting and corrections
├── analyze_topo_pingpong.py # topology benchmark analysis
├── scripts/
│   └── prepare_mnist.py     # MNIST preprocessing
├── results/                 # raw .out files from all runs
└── submit_*.sh              # SLURM batch scripts

---

## Reproducing the Experiments

### Prerequisites
- Perlmutter CPU partition access (NERSC allocation)
- MPI-enabled C++ compiler (`mpicxx`)
- Python 3 with numpy

### 1. Prepare data
```bash
# Download raw MNIST
cd data/mnist/raw
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz

# Preprocess
module load python
python3 scripts/prepare_mnist.py
```

### 2. Single-node baseline
```bash
g++ -O2 -std=c++17 grad_packed.cpp -o grad_packed
./grad_packed
```

### 3. Distributed scaling runs (submit via SLURM)
```bash
sbatch submit_mpi_allreduce.sh
sbatch submit_ring_allreduce.sh
sbatch submit_tree_allreduce.sh
```

### 4. Network benchmarks
```bash
sbatch submit_pingpong.sh
sbatch submit_topo_pingpong.sh
```

### 5. Alpha-beta analysis
```bash
module load python
python3 fit_alpha_beta.py
python3 analyze_topo_pingpong.py
```

---

## Model

4-layer MLP: **784 → 1024 → 1024 → 1024 → 10**  
Dataset: MNIST (60k train / 10k test)  
Parameters: ~2.9M (gradient buffer: 11.1 MB FP32)  
Optimizer: SGD with momentum (µ=0.9), lr=0.01, global batch=256  
Platform: Perlmutter CPU nodes, Slingshot-11 network, 1 rank/node
