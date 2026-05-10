// tree_allreduce.cpp — Data-parallel MLP with hand-written tree all-reduce
// Architecture: 784 -> 1024 -> 1024 -> 1024 -> 10
// Compile: mpicxx -O2 -std=c++17 tree_allreduce.cpp -o tree_allreduce
// Run:     srun -n <ranks> ./tree_allreduce
//
// Same training logic as mpi_allreduce.cpp; only the gradient all-reduce is
// replaced by a hand-written binary tree (reduce-to-root + broadcast-from-root).
//
// Algorithm: two phases, each log2(P) steps, each step sends the full buffer.
//   Phase 1 (reduce): leaves -> rank 0, summed in place along the way
//   Phase 2 (broadcast): rank 0 -> leaves, overwriting stale buffers
// Total: 2·log2(P) steps. Cost model: 2·log2(P) · (α + n·β)
//
// Constraint: P must be a power of 2 (validated in main).
// For non-power-of-2 support we would need an extra "fold-in" round that
// sums extra ranks into the 2^k subset before running the tree; omitted here.

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

// ─── Config ───────────────────────────────────────────────────────────────────
// Relative to CWD — run binaries from the project root directory.
const std::string DATA_DIR = "data/mnist/processed/";

const int INPUT_DIM  = 784;
const int H1         = 1024;
const int H2         = 1024;
const int H3         = 1024;
const int OUTPUT_DIM = 10;

const int   TRAIN_N    = 60000;
const int   TEST_N     = 10000;
const int   BATCH_SIZE = 256;
const int   EPOCHS     = 10;
const float LR         = 0.01f;
const float MOMENTUM   = 0.9f;

// ─── Hand-written tree all-reduce ─────────────────────────────────────────────
// Sums `buf` across all ranks in `comm`, in-place. After the call every rank
// holds the same total (NOT averaged — caller divides by nranks for average,
// matching MPI_Allreduce(MPI_SUM) semantics).
//
// Preconditions (not checked here — validate in main):
//   - P = size(comm) is a power of 2
//   - recv_buf is caller-allocated, length >= n, distinct from buf
//
// Role selection at step k (stride = 1<<k, mask = (2*stride)-1):
//   (rank & mask) == 0       → "subtree root": recv from rank+stride and sum
//                                               (reduce) / send to rank+stride
//                                               (broadcast)
//   (rank & mask) == stride  → "partner": send to rank-stride and leave reduce
//                                         / recv from rank-stride (broadcast)
//   otherwise                → idle this step
static void tree_allreduce(float* buf, float* recv_buf, size_t n, MPI_Comm comm) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (P == 1) return;  // single rank: nothing to reduce

    // log2(P); P is assumed to be a power of 2.
    int log2P = 0;
    while ((1 << log2P) < P) log2P++;

    const int TAG_REDUCE = 100;
    const int TAG_BCAST  = 200;

    // ── Phase 1: reduce to rank 0 ─────────────────────────────────────────────
    // stride doubles each step (1, 2, 4, ...). A rank that sends exits the
    // loop via `break` — it will rejoin in the broadcast phase below to
    // receive the final global sum.
    for (int step = 0; step < log2P; step++) {
        int stride = 1 << step;
        int mask   = (2 * stride) - 1;

        if ((rank & mask) == 0) {
            int partner = rank + stride;
            MPI_Recv(recv_buf, (int)n, MPI_FLOAT, partner, TAG_REDUCE,
                     comm, MPI_STATUS_IGNORE);
            for (size_t i = 0; i < n; i++) buf[i] += recv_buf[i];
        }
        else if ((rank & mask) == stride) {
            int partner = rank - stride;
            MPI_Send(buf, (int)n, MPI_FLOAT, partner, TAG_REDUCE, comm);
            break;  // done with reduce for this rank
        }
        // else: this rank already sent in an earlier step; idle.
    }

    // ── Phase 2: broadcast from rank 0 ────────────────────────────────────────
    // Same role mask, but stride iterates P/2 -> 1, and message direction
    // is flipped: subtree roots send, partners recv. Receivers write directly
    // into `buf` because the old contents are stale (only rank 0 had valid
    // data after reduce) and we want to overwrite them anyway.
    for (int step = log2P - 1; step >= 0; step--) {
        int stride = 1 << step;
        int mask   = (2 * stride) - 1;

        if ((rank & mask) == 0) {
            int partner = rank + stride;
            MPI_Send(buf, (int)n, MPI_FLOAT, partner, TAG_BCAST, comm);
        }
        else if ((rank & mask) == stride) {
            int partner = rank - stride;
            MPI_Recv(buf, (int)n, MPI_FLOAT, partner, TAG_BCAST,
                     comm, MPI_STATUS_IGNORE);
        }
        // else: not yet reached by broadcast; will receive in a later step.
    }
}

// ─── Data loading ─────────────────────────────────────────────────────────────
std::vector<float> load_float_bin(const std::string& path, size_t count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!f) throw std::runtime_error("Read failed: " + path);
    return data;
}

std::vector<uint8_t> load_u8_bin(const std::string& path, size_t count) {
    std::vector<uint8_t> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint8_t));
    if (!f) throw std::runtime_error("Read failed: " + path);
    return data;
}

// ─── Weight init (Xavier uniform) ─────────────────────────────────────────────
void xavier_init(std::vector<float>& w, int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    for (auto& v : w)
        v = ((float)rand() / RAND_MAX) * 2 * limit - limit;
}

// ─── Activations ──────────────────────────────────────────────────────────────
void relu(std::vector<float>& x) {
    for (auto& v : x) v = std::max(0.0f, v);
}

void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ─── Matrix multiply: out[M x N] = A[M x K] * B[K x N] ───────────────────────
void matmul(const float* A, const float* B, float* out, int M, int K, int N) {
    std::fill(out, out + M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float aik = A[i * K + k];
            for (int j = 0; j < N; j++)
                out[i * N + j] += aik * B[k * N + j];
        }
}

// ─── Gradient norm over flat buffer ───────────────────────────────────────────
float grad_norm(const std::vector<float>& buf) {
    double sum = 0.0;
    for (float v : buf) sum += (double)v * v;
    return (float)std::sqrt(sum);
}

// ─── MLP ──────────────────────────────────────────────────────────────────────
struct MLP {
    std::vector<float> W1, b1;
    std::vector<float> W2, b2;
    std::vector<float> W3, b3;
    std::vector<float> W4, b4;

    std::vector<float> dW1, db1;
    std::vector<float> dW2, db2;
    std::vector<float> dW3, db3;
    std::vector<float> dW4, db4;

    // Flat gradient buffer — what tree_allreduce operates on.
    // Layout: dW1 | db1 | dW2 | db2 | dW3 | db3 | dW4 | db4
    std::vector<float> grad_buf;
    size_t grad_buf_size = 0;

    // Persistent receive buffer for the tree reduce phase (allocated once,
    // ~11 MB, reused across all training steps to avoid per-step malloc).
    std::vector<float> recv_buf;

    std::vector<float> vW1, vb1;
    std::vector<float> vW2, vb2;
    std::vector<float> vW3, vb3;
    std::vector<float> vW4, vb4;

    std::vector<float> z1, a1;
    std::vector<float> z2, a2;
    std::vector<float> z3, a3;
    std::vector<float> z4, a4;

    MLP() {
        W1.resize(INPUT_DIM * H1);   b1.resize(H1, 0.0f);
        W2.resize(H1 * H2);          b2.resize(H2, 0.0f);
        W3.resize(H2 * H3);          b3.resize(H3, 0.0f);
        W4.resize(H3 * OUTPUT_DIM);  b4.resize(OUTPUT_DIM, 0.0f);

        dW1.resize(INPUT_DIM * H1, 0.0f); db1.resize(H1, 0.0f);
        dW2.resize(H1 * H2, 0.0f);        db2.resize(H2, 0.0f);
        dW3.resize(H2 * H3, 0.0f);        db3.resize(H3, 0.0f);
        dW4.resize(H3 * OUTPUT_DIM, 0.0f); db4.resize(OUTPUT_DIM, 0.0f);

        grad_buf_size = dW1.size() + db1.size()
                      + dW2.size() + db2.size()
                      + dW3.size() + db3.size()
                      + dW4.size() + db4.size();
        grad_buf.resize(grad_buf_size, 0.0f);
        recv_buf.resize(grad_buf_size, 0.0f);

        vW1.resize(INPUT_DIM * H1, 0.0f); vb1.resize(H1, 0.0f);
        vW2.resize(H1 * H2, 0.0f);        vb2.resize(H2, 0.0f);
        vW3.resize(H2 * H3, 0.0f);        vb3.resize(H3, 0.0f);
        vW4.resize(H3 * OUTPUT_DIM, 0.0f); vb4.resize(OUTPUT_DIM, 0.0f);

        xavier_init(W1, INPUT_DIM, H1);
        xavier_init(W2, H1, H2);
        xavier_init(W3, H2, H3);
        xavier_init(W4, H3, OUTPUT_DIM);
    }

    void pack_grads() {
        float* p = grad_buf.data();
        auto copy_in = [&](const std::vector<float>& v) {
            std::memcpy(p, v.data(), v.size() * sizeof(float));
            p += v.size();
        };
        copy_in(dW1); copy_in(db1);
        copy_in(dW2); copy_in(db2);
        copy_in(dW3); copy_in(db3);
        copy_in(dW4); copy_in(db4);
    }

    void unpack_grads() {
        const float* p = grad_buf.data();
        auto copy_out = [&](std::vector<float>& v) {
            std::memcpy(v.data(), p, v.size() * sizeof(float));
            p += v.size();
        };
        copy_out(dW1); copy_out(db1);
        copy_out(dW2); copy_out(db2);
        copy_out(dW3); copy_out(db3);
        copy_out(dW4); copy_out(db4);
    }

    float forward(const float* x, const uint8_t* labels, int bs) {
        z1.resize(bs * H1); a1.resize(bs * H1);
        z2.resize(bs * H2); a2.resize(bs * H2);
        z3.resize(bs * H3); a3.resize(bs * H3);
        z4.resize(bs * OUTPUT_DIM); a4.resize(bs * OUTPUT_DIM);

        matmul(x, W1.data(), z1.data(), bs, INPUT_DIM, H1);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H1; j++) z1[i*H1+j] += b1[j];
        a1 = z1; relu(a1);

        matmul(a1.data(), W2.data(), z2.data(), bs, H1, H2);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H2; j++) z2[i*H2+j] += b2[j];
        a2 = z2; relu(a2);

        matmul(a2.data(), W3.data(), z3.data(), bs, H2, H3);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H3; j++) z3[i*H3+j] += b3[j];
        a3 = z3; relu(a3);

        matmul(a3.data(), W4.data(), z4.data(), bs, H3, OUTPUT_DIM);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < OUTPUT_DIM; j++) z4[i*OUTPUT_DIM+j] += b4[j];
        a4 = z4;
        for (int i = 0; i < bs; i++) softmax(&a4[i*OUTPUT_DIM], OUTPUT_DIM);

        float loss = 0.0f;
        for (int i = 0; i < bs; i++)
            loss -= std::log(a4[i*OUTPUT_DIM + labels[i]] + 1e-9f);
        return loss / bs;
    }

    void backward(const float* x, const uint8_t* labels, int bs) {
        std::fill(dW1.begin(), dW1.end(), 0.0f); std::fill(db1.begin(), db1.end(), 0.0f);
        std::fill(dW2.begin(), dW2.end(), 0.0f); std::fill(db2.begin(), db2.end(), 0.0f);
        std::fill(dW3.begin(), dW3.end(), 0.0f); std::fill(db3.begin(), db3.end(), 0.0f);
        std::fill(dW4.begin(), dW4.end(), 0.0f); std::fill(db4.begin(), db4.end(), 0.0f);

        float scale = 1.0f / bs;

        std::vector<float> d4(bs * OUTPUT_DIM);
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++)
                d4[i*OUTPUT_DIM+j] = a4[i*OUTPUT_DIM+j];
            d4[i*OUTPUT_DIM + labels[i]] -= 1.0f;
        }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < H3; j++)
                for (int k = 0; k < OUTPUT_DIM; k++)
                    dW4[j*OUTPUT_DIM+k] += a3[i*H3+j] * d4[i*OUTPUT_DIM+k];
            for (int k = 0; k < OUTPUT_DIM; k++)
                db4[k] += d4[i*OUTPUT_DIM+k];
        }

        std::vector<float> d3(bs * H3);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H3; j++) {
                float val = 0.0f;
                for (int k = 0; k < OUTPUT_DIM; k++)
                    val += d4[i*OUTPUT_DIM+k] * W4[j*OUTPUT_DIM+k];
                d3[i*H3+j] = val * (z3[i*H3+j] > 0.0f ? 1.0f : 0.0f);
            }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < H2; j++)
                for (int k = 0; k < H3; k++)
                    dW3[j*H3+k] += a2[i*H2+j] * d3[i*H3+k];
            for (int k = 0; k < H3; k++)
                db3[k] += d3[i*H3+k];
        }

        std::vector<float> d2(bs * H2);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H2; j++) {
                float val = 0.0f;
                for (int k = 0; k < H3; k++)
                    val += d3[i*H3+k] * W3[j*H3+k];
                d2[i*H2+j] = val * (z2[i*H2+j] > 0.0f ? 1.0f : 0.0f);
            }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < H1; j++)
                for (int k = 0; k < H2; k++)
                    dW2[j*H2+k] += a1[i*H1+j] * d2[i*H2+k];
            for (int k = 0; k < H2; k++)
                db2[k] += d2[i*H2+k];
        }

        std::vector<float> d1(bs * H1);
        for (int i = 0; i < bs; i++)
            for (int j = 0; j < H1; j++) {
                float val = 0.0f;
                for (int k = 0; k < H2; k++)
                    val += d2[i*H2+k] * W2[j*H2+k];
                d1[i*H1+j] = val * (z1[i*H1+j] > 0.0f ? 1.0f : 0.0f);
            }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < INPUT_DIM; j++)
                for (int k = 0; k < H1; k++)
                    dW1[j*H1+k] += x[i*INPUT_DIM+j] * d1[i*H1+k];
            for (int k = 0; k < H1; k++)
                db1[k] += d1[i*H1+k];
        }

        for (auto& v : dW1) v *= scale;  for (auto& v : db1) v *= scale;
        for (auto& v : dW2) v *= scale;  for (auto& v : db2) v *= scale;
        for (auto& v : dW3) v *= scale;  for (auto& v : db3) v *= scale;
        for (auto& v : dW4) v *= scale;  for (auto& v : db4) v *= scale;
    }

    void sgd_step(float lr) {
        auto update = [&](std::vector<float>& w, std::vector<float>& dw,
                          std::vector<float>& v) {
            for (size_t i = 0; i < w.size(); i++) {
                v[i] = MOMENTUM * v[i] - lr * dw[i];
                w[i] += v[i];
            }
        };
        update(W1, dW1, vW1);  update(b1, db1, vb1);
        update(W2, dW2, vW2);  update(b2, db2, vb2);
        update(W3, dW3, vW3);  update(b3, db3, vb3);
        update(W4, dW4, vW4);  update(b4, db4, vb4);
    }

    float accuracy(const float* x, const uint8_t* labels, int n) {
        int correct = 0;
        std::vector<float> out(OUTPUT_DIM);
        std::vector<float> h1(H1), h2(H2), h3(H3);

        for (int i = 0; i < n; i++) {
            const float* xi = x + i * INPUT_DIM;
            for (int j = 0; j < H1; j++) {
                float v = b1[j];
                for (int k = 0; k < INPUT_DIM; k++) v += xi[k] * W1[k*H1+j];
                h1[j] = std::max(0.0f, v);
            }
            for (int j = 0; j < H2; j++) {
                float v = b2[j];
                for (int k = 0; k < H1; k++) v += h1[k] * W2[k*H2+j];
                h2[j] = std::max(0.0f, v);
            }
            for (int j = 0; j < H3; j++) {
                float v = b3[j];
                for (int k = 0; k < H2; k++) v += h2[k] * W3[k*H3+j];
                h3[j] = std::max(0.0f, v);
            }
            for (int j = 0; j < OUTPUT_DIM; j++) {
                float v = b4[j];
                for (int k = 0; k < H3; k++) v += h3[k] * W4[k*OUTPUT_DIM+j];
                out[j] = v;
            }
            int pred = std::max_element(out.begin(), out.end()) - out.begin();
            if (pred == labels[i]) correct++;
        }
        return (float)correct / n;
    }
};

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // All ranks share the same seed so initial weights match.
    srand(42);

    if (rank == 0) std::cout << "Loading MNIST...\n";
    auto train_images = load_float_bin(DATA_DIR + "train_images.bin", TRAIN_N * INPUT_DIM);
    auto train_labels = load_u8_bin(DATA_DIR + "train_labels.bin", TRAIN_N);
    auto test_images  = load_float_bin(DATA_DIR + "test_images.bin", TEST_N * INPUT_DIM);
    auto test_labels  = load_u8_bin(DATA_DIR + "test_labels.bin", TEST_N);
    if (rank == 0) std::cout << "Loaded.\n\n";

    // Sanity 1: global batch must divide evenly across ranks.
    if (BATCH_SIZE % nranks != 0) {
        if (rank == 0)
            std::cerr << "ERROR: BATCH_SIZE (" << BATCH_SIZE
                      << ") must be divisible by nranks (" << nranks << ")\n";
        MPI_Finalize();
        return 1;
    }

    // Sanity 2: this tree implementation requires P to be a power of 2.
    if ((nranks & (nranks - 1)) != 0) {
        if (rank == 0)
            std::cerr << "ERROR: nranks (" << nranks
                      << ") must be a power of 2 for tree_allreduce\n";
        MPI_Finalize();
        return 1;
    }

    int local_bs = BATCH_SIZE / nranks;

    int log2P = 0;
    while ((1 << log2P) < nranks) log2P++;

    if (rank == 0) {
        std::cout << "=== Tree All-Reduce run ===\n";
        std::cout << "nranks:         " << nranks << "\n";
        std::cout << "global_batch:   " << BATCH_SIZE << "\n";
        std::cout << "local_batch:    " << local_bs << "\n";
        std::cout << "algorithm:      binary tree (reduce + broadcast)\n";
        std::cout << "tree_steps:     " << log2P << " reduce + " << log2P
                  << " broadcast = " << (2 * log2P) << " total\n";
        std::cout << "epochs:         " << EPOCHS << "\n";
        std::cout << "lr:             " << LR << "\n";
        std::cout << "momentum:       " << MOMENTUM << "\n\n";
    }

    MLP model;

    if (rank == 0) {
        std::cout << "grad_buf_size:  " << model.grad_buf_size
                  << " floats = "
                  << (model.grad_buf_size * 4) / (1024.0 * 1024.0)
                  << " MB\n\n";

        // CSV header — kept identical to mpi_allreduce.cpp so the three
        // implementations' output files can be combined directly.
        std::cout << "epoch,"
                  << "loss,"
                  << "test_acc,"
                  << "fwd_s,"
                  << "bwd_s,"
                  << "allreduce_s,"
                  << "sgd_s,"
                  << "epoch_s,"
                  << "grad_norm,"
                  << "speedup,"
                  << "efficiency\n";
    }

    std::vector<int> idx(TRAIN_N);
    std::iota(idx.begin(), idx.end(), 0);

    int steps_per_epoch = TRAIN_N / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Re-seed so all ranks shuffle identically.
        srand(42 + epoch);
        for (int i = TRAIN_N - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(idx[i], idx[j]);
        }

        float  epoch_loss    = 0.0f;
        double t_fwd         = 0.0;
        double t_bwd         = 0.0;
        double t_allreduce   = 0.0;
        double t_sgd         = 0.0;

        std::vector<float>   local_x(local_bs * INPUT_DIM);
        std::vector<uint8_t> local_y(local_bs);

        auto epoch_start = Clock::now();

        for (int step = 0; step < steps_per_epoch; step++) {
            int global_offset = step * BATCH_SIZE + rank * local_bs;
            for (int b = 0; b < local_bs; b++) {
                int s = idx[global_offset + b];
                std::memcpy(local_x.data() + b * INPUT_DIM,
                            train_images.data() + s * INPUT_DIM,
                            INPUT_DIM * sizeof(float));
                local_y[b] = train_labels[s];
            }

            auto t0 = Clock::now();
            float loss = model.forward(local_x.data(), local_y.data(), local_bs);
            auto t1 = Clock::now();

            model.backward(local_x.data(), local_y.data(), local_bs);
            auto t2 = Clock::now();

            model.pack_grads();

            // ── Hand-written tree all-reduce (sum across ranks) ───────────────
            // Replaces MPI_Allreduce in mpi_allreduce.cpp. After this call
            // every rank holds Σ grad_buf; the divide below turns that into
            // the mean, matching data-parallel SGD semantics exactly.
            tree_allreduce(model.grad_buf.data(),
                           model.recv_buf.data(),
                           model.grad_buf_size,
                           MPI_COMM_WORLD);

            float inv_n = 1.0f / nranks;
            for (auto& v : model.grad_buf) v *= inv_n;

            auto t3 = Clock::now();

            model.unpack_grads();

            model.sgd_step(LR);
            auto t4 = Clock::now();

            epoch_loss   += loss;
            t_fwd        += Sec(t1 - t0).count();
            t_bwd        += Sec(t2 - t1).count();
            t_allreduce  += Sec(t3 - t2).count();  // pack + tree_allreduce + average
            t_sgd        += Sec(t4 - t3).count();
        }

        double epoch_s    = Sec(Clock::now() - epoch_start).count();
        float  train_loss = epoch_loss / steps_per_epoch;

        if (rank == 0) {
            float test_acc = model.accuracy(test_images.data(), test_labels.data(), TEST_N);
            float gnorm    = grad_norm(model.grad_buf);

            std::cout << epoch + 1    << ","
                      << train_loss   << ","
                      << test_acc     << ","
                      << t_fwd        << ","
                      << t_bwd        << ","
                      << t_allreduce  << ","
                      << t_sgd        << ","
                      << epoch_s      << ","
                      << gnorm        << ","
                      << 0.0          << ","
                      << 0.0          << "\n";
            std::cout.flush();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
