# CS 5220 Project — Parallel Neural Network Training with MPI
**Cornell University, Spring 2026**
Yuting Chen & Anirudh Atmakuru

Comparing gradient aggregation strategies (MPI_Allreduce, Ring All-Reduce, Tree Reduction) for data-parallel SGD training of a multi-layer perceptron on MNIST.

---

## Project Structure

```
project/
├── data/
│   └── mnist/
│       ├── raw/          # downloaded .gz files (not in git)
│       └── processed/    # binary files produced by prepare_mnist.py (not in git)
├── results/              # .out files from training runs
├── scripts/
│   └── prepare_mnist.py  # data preprocessing script
├── test_mnist_loader.cpp # sanity check for processed data
├── mlp.cpp               # single-node MLP baseline (no MPI)
├── grad_packed.cpp       # single-node MLP with flat gradient buffer (MPI-ready)
├── mpi_allreduce.cpp     # distributed MLP using MPI_Allreduce
├── run_mlp.sh            # run script for mlp
├── run_grad_packed.sh    # run script for grad_packed
└── run_mpi_allreduce.sh  # run script for mpi_allreduce
```

---

## Step 1 — Download and Prepare MNIST Data

### 1a. Download raw MNIST files

```bash
mkdir -p /pscratch/sd/a/anirudh6/cs5220/project/data/mnist/raw
mkdir -p /pscratch/sd/a/anirudh6/cs5220/project/data/mnist/processed
cd /pscratch/sd/a/anirudh6/cs5220/project/data/mnist/raw

curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
curl -O https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
```

### 1b. Preprocess into flat binary files

```bash
module load python
python3 /pscratch/sd/a/anirudh6/cs5220/project/scripts/prepare_mnist.py
```

This reads the raw `.gz` files, normalizes pixel values to `[0, 1]` float32, and writes four flat binary files to `data/mnist/processed/`:
- `train_images.bin` — 60000 × 784 float32
- `train_labels.bin` — 60000 uint8
- `test_images.bin`  — 10000 × 784 float32
- `test_labels.bin`  — 10000 uint8

It also writes `metadata.txt` with shape and dtype info.

### 1c. Verify the processed data (optional but recommended)

```bash
cd /pscratch/sd/a/anirudh6/cs5220/project
g++ -O2 -std=c++17 test_mnist_loader.cpp -o test_mnist_loader
./test_mnist_loader
```

Expected output:
```
Loaded MNIST successfully
train_images elements = 47040000
train_labels elements = 60000
test_images elements  = 7840000
test_labels elements  = 10000
First train label = 5
First pixel of first image = 0
Last pixel of first image = 0
```

---

## Step 2 — Single-Node MLP Baseline (`mlp.cpp`)

A straightforward single-node MLP with no MPI. Used to verify that the model
can reach >97% test accuracy before any distributed training is added.

**Architecture:** 784 → 1024 → 1024 → 1024 → 10
**Optimizer:** SGD with momentum (0.9)
**Loss:** Cross-entropy

### Compile and run

```bash
cp mlp.cpp /pscratch/sd/a/anirudh6/cs5220/project/
cp run_mlp.sh /pscratch/sd/a/anirudh6/cs5220/project/
bash run_mlp.sh
```

Output is saved to `results/mlp_baseline_<timestamp>.out`.

### What to expect

```
epoch,loss,test_acc,fwd_s,bwd_s,sgd_s,epoch_s,grad_norm
1,0.48,0.89,12.3,44.8,0.02,57.6,2.34
...
25,0.08,0.97,12.1,44.6,0.02,57.3,0.81
```

Test accuracy should reach **>97%** by epoch 20–25. The `epoch_s` column
is the single-node baseline used for computing speedup in later experiments.

---

## Step 3 — Gradient-Packed Baseline (`grad_packed.cpp`)

Identical to `mlp.cpp` in math and results, but all gradients are packed into
a single contiguous flat buffer after each backward pass. This is the
**MPI-ready baseline** — when MPI is added, only one line changes
(the all-reduce call in the training loop).

**Flat buffer layout:** `dW1 | db1 | dW2 | db2 | dW3 | db3 | dW4 | db4`
**Total size:** ~11.1 MB (2,912,290 float32 values)

### Compile and run

```bash
cp grad_packed.cpp /pscratch/sd/a/anirudh6/cs5220/project/
cp run_grad_packed.sh /pscratch/sd/a/anirudh6/cs5220/project/
bash run_grad_packed.sh
```

Output is saved to `results/grad_packed_<timestamp>.out`.

### What to expect

Results should be **identical** to `mlp.cpp` — same loss, same test accuracy,
same grad_norm. The only difference is a new `pack_s` column measuring the
time for pack + unpack (should be ~0.01–0.02s, negligible).

```
epoch,loss,test_acc,fwd_s,bwd_s,pack_s,sgd_s,epoch_s,grad_norm
1,0.48,0.89,12.3,44.8,0.01,0.02,57.6,2.34
...
```

Also prints at startup:
```
grad_buf_size = 2912290 floats = 11.12 MB
```

---

## Step 4 — Distributed Training with MPI_Allreduce (`mpi_allreduce.cpp`)

Data-parallel distributed training using `MPI_Allreduce` as the gradient
aggregation method. This is the **correctness reference** — ring and tree
all-reduce implementations will be verified by comparing their output against
this.

**How it works:**
- Each rank loads the full dataset but trains on its own shard of each batch
- Global batch size = 256; local batch per rank = 256 / nranks
- After backward(), all ranks all-reduce `grad_buf` (sum, then divide by nranks)
- All ranks apply identical gradient updates — weights stay in sync
- Only rank 0 evaluates accuracy and prints output

### Compile

```bash
mpicxx -O2 -std=c++17 mpi_allreduce.cpp -o mpi_allreduce
```

### Run (must be inside a SLURM allocation — do not run on login node)

**Option A — Interactive session (recommended for development):**
```bash
salloc --nodes=8 --ntasks-per-node=1 --constraint=cpu \
       --qos=interactive --time=01:00:00 --account=m4341
# once granted:
bash run_mpi_allreduce.sh
```

**Option B — Batch job:**
```bash
sbatch submit_mpi_allreduce.sh
squeue -u anirudh6   # monitor job
```

The run script sweeps across rank counts defined in `RANK_COUNTS` inside
`run_mpi_allreduce.sh`. Default for development: `(1 2 4 8)`.
For full scaling experiments change to: `(1 2 4 8 16 32)`.

Each rank count gets its own output file:
```
results/mpi_allreduce_nranks1_<timestamp>.out
results/mpi_allreduce_nranks2_<timestamp>.out
...
```

### What to expect

```
=== MPI_Allreduce run ===
nranks:        4
global_batch:  256
local_batch:   64
grad_buf_size: 2912290 floats = 11.12 MB

epoch,loss,test_acc,fwd_s,bwd_s,allreduce_s,sgd_s,epoch_s,grad_norm,speedup,efficiency
1,0.48,0.89,3.1,11.2,0.45,0.01,15.3,2.34,0,0
...
25,0.08,0.97,3.0,11.1,0.44,0.01,15.0,0.81,0,0
```

### Correctness checks before moving on

- `test_acc` at epoch 25 should be within ~0.5% of `grad_packed` single-node result
- `grad_norm` should be in the same ballpark as single-node
- Loss curve should decrease smoothly with the same shape
- At nranks=1 results should be nearly identical to `grad_packed`

### Computing speedup and efficiency (post-processing)

The `speedup` and `efficiency` columns are printed as `0` during the run.
Compute them after by comparing `epoch_s` against the `grad_packed` baseline:

```
speedup    = baseline_epoch_s / this_run_epoch_s
efficiency = speedup / nranks
```

---

## Planned Next Steps

- [ ] Hand-written Ring All-Reduce
- [ ] Hand-written Tree Reduction
- [ ] Strong scaling experiments (1–32 ranks)
- [ ] Alpha-beta model fitting
- [ ] Runtime breakdown analysis
