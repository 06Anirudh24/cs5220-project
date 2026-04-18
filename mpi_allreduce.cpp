// mpi_allreduce.cpp — Data-parallel MLP with MPI_Allreduce (correctness reference)
// Architecture: 784 -> 1024 -> 1024 -> 1024 -> 10
// Compile: mpicxx -O2 -std=c++17 mpi_allreduce.cpp -o mpi_allreduce
// Run:     srun -n <ranks> ./mpi_allreduce
//
// This is the distributed correctness reference. Ring and tree all-reduce
// implementations will be verified by comparing their output against this.
//
// What this does differently from grad_packed.cpp:
//   - Each rank loads the full dataset but trains on its own shard
//   - After backward(), all ranks all-reduce grad_buf (sum, then divide by nranks)
//   - All ranks apply identical gradient updates -> weights stay in sync
//   - Only rank 0 prints output and evaluates accuracy

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
const std::string DATA_DIR = "/pscratch/sd/a/anirudh6/cs5220/project/data/mnist/processed/";

const int INPUT_DIM  = 784;
const int H1         = 1024;
const int H2         = 1024;
const int H3         = 1024;
const int OUTPUT_DIM = 10;

const int   TRAIN_N    = 60000;
const int   TEST_N     = 10000;
const int   BATCH_SIZE = 256;   // global batch size — each rank processes BATCH_SIZE/nranks
const int   EPOCHS     = 10;
const float LR         = 0.01f;
const float MOMENTUM   = 0.9f;

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

    // Flat gradient buffer — this is what MPI_Allreduce operates on
    // Layout: dW1 | db1 | dW2 | db2 | dW3 | db3 | dW4 | db4
    std::vector<float> grad_buf;
    size_t grad_buf_size = 0;

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

    // All ranks use the same seed so weights are identical at init
    srand(42);

    // ── Load data (all ranks load full dataset) ────────────────────────────────
    // Each rank will slice out its own shard of training indices each step.
    // Test data is only used by rank 0 for evaluation.
    if (rank == 0) std::cout << "Loading MNIST...\n";
    auto train_images = load_float_bin(DATA_DIR + "train_images.bin", TRAIN_N * INPUT_DIM);
    auto train_labels = load_u8_bin(DATA_DIR + "train_labels.bin", TRAIN_N);
    auto test_images  = load_float_bin(DATA_DIR + "test_images.bin", TEST_N * INPUT_DIM);
    auto test_labels  = load_u8_bin(DATA_DIR + "test_labels.bin", TEST_N);
    if (rank == 0) std::cout << "Loaded.\n\n";

    // ── Sanity check: global batch must divide evenly across ranks ─────────────
    if (BATCH_SIZE % nranks != 0) {
        if (rank == 0)
            std::cerr << "ERROR: BATCH_SIZE (" << BATCH_SIZE
                      << ") must be divisible by nranks (" << nranks << ")\n";
        MPI_Finalize();
        return 1;
    }
    int local_bs = BATCH_SIZE / nranks;  // each rank's local batch size

    // ── Print header (rank 0 only) ─────────────────────────────────────────────
    if (rank == 0) {
        std::cout << "=== MPI_Allreduce run ===\n";
        std::cout << "nranks:         " << nranks         << "\n";
        std::cout << "global_batch:   " << BATCH_SIZE     << "\n";
        std::cout << "local_batch:    " << local_bs       << "\n";
        std::cout << "grad_buf_size:  will print after model init\n";
        std::cout << "epochs:         " << EPOCHS         << "\n";
        std::cout << "lr:             " << LR             << "\n";
        std::cout << "momentum:       " << MOMENTUM       << "\n\n";
    }

    MLP model;

    if (rank == 0) {
        std::cout << "grad_buf_size:  " << model.grad_buf_size
                  << " floats = "
                  << (model.grad_buf_size * 4) / (1024.0 * 1024.0)
                  << " MB\n\n";

        // CSV header — one row per epoch, printed by rank 0
        std::cout << "epoch,"
                  << "loss,"          // avg cross-entropy loss (rank 0's local loss)
                  << "test_acc,"      // accuracy on full test set (rank 0 only)
                  << "fwd_s,"         // seconds in forward pass (rank 0)
                  << "bwd_s,"         // seconds in backward pass (rank 0)
                  << "allreduce_s,"   // seconds blocked in MPI_Allreduce (rank 0)
                  << "sgd_s,"         // seconds in SGD update (rank 0)
                  << "epoch_s,"       // total wall-clock seconds for epoch (rank 0)
                  << "grad_norm,"     // L2 norm of grad_buf after allreduce (rank 0)
                  << "speedup,"       // epoch_s_1rank / epoch_s (filled in post-processing)
                  << "efficiency\n";  // speedup / nranks
    }

    // All ranks need the same shuffled indices each epoch.
    // We generate them on all ranks with the same seed so they agree.
    std::vector<int> idx(TRAIN_N);
    std::iota(idx.begin(), idx.end(), 0);

    int steps_per_epoch = TRAIN_N / BATCH_SIZE;

    // Reference epoch time for speedup calculation (set after epoch 1)
    // Speedup and efficiency are placeholders here — compute in post-processing
    // by dividing the grad_packed single-node epoch_s by this run's epoch_s.

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // ── Shuffle (same on all ranks) ────────────────────────────────────────
        // Re-seed with epoch number so all ranks shuffle identically
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

        // Local batch buffers for this rank
        std::vector<float>   local_x(local_bs * INPUT_DIM);
        std::vector<uint8_t> local_y(local_bs);

        auto epoch_start = Clock::now();

        for (int step = 0; step < steps_per_epoch; step++) {
            // ── Each rank takes its own slice of the global batch ──────────────
            // Global batch indices: step*BATCH_SIZE ... step*BATCH_SIZE+BATCH_SIZE-1
            // Rank r takes: offset = rank * local_bs within that global batch
            int global_offset = step * BATCH_SIZE + rank * local_bs;
            for (int b = 0; b < local_bs; b++) {
                int s = idx[global_offset + b];
                std::memcpy(local_x.data() + b * INPUT_DIM,
                            train_images.data() + s * INPUT_DIM,
                            INPUT_DIM * sizeof(float));
                local_y[b] = train_labels[s];
            }

            // ── Forward ───────────────────────────────────────────────────────
            auto t0 = Clock::now();
            float loss = model.forward(local_x.data(), local_y.data(), local_bs);
            auto t1 = Clock::now();

            // ── Backward ──────────────────────────────────────────────────────
            model.backward(local_x.data(), local_y.data(), local_bs);
            auto t2 = Clock::now();

            // ── Pack gradients into flat buffer ───────────────────────────────
            model.pack_grads();

            // ── MPI_Allreduce: sum grad_buf across all ranks ───────────────────
            // After this call, every rank has the sum of all ranks' gradients.
            // We then divide by nranks to get the average gradient, which is
            // mathematically equivalent to computing the gradient over the full
            // global batch. This is the core of data-parallel SGD.
            MPI_Allreduce(MPI_IN_PLACE,
                          model.grad_buf.data(),
                          (int)model.grad_buf_size,
                          MPI_FLOAT,
                          MPI_SUM,
                          MPI_COMM_WORLD);

            // Divide by nranks to average (not sum) across ranks
            float inv_n = 1.0f / nranks;
            for (auto& v : model.grad_buf) v *= inv_n;

            auto t3 = Clock::now();

            // ── Unpack averaged gradients back to individual vectors ───────────
            model.unpack_grads();

            // ── SGD update ────────────────────────────────────────────────────
            model.sgd_step(LR);
            auto t4 = Clock::now();

            epoch_loss   += loss;
            t_fwd        += Sec(t1 - t0).count();
            t_bwd        += Sec(t2 - t1).count();
            t_allreduce  += Sec(t3 - t2).count();  // includes pack + allreduce + unpack
            t_sgd        += Sec(t4 - t3).count();
        }

        double epoch_s    = Sec(Clock::now() - epoch_start).count();
        float  train_loss = epoch_loss / steps_per_epoch;

        // Only rank 0 evaluates accuracy and prints
        if (rank == 0) {
            float test_acc = model.accuracy(test_images.data(), test_labels.data(), TEST_N);
            float gnorm    = grad_norm(model.grad_buf);

            // speedup and efficiency are left as 0 here —
            // compute them in post-processing: speedup = baseline_epoch_s / epoch_s
            std::cout << epoch + 1    << ","
                      << train_loss   << ","
                      << test_acc     << ","
                      << t_fwd        << ","
                      << t_bwd        << ","
                      << t_allreduce  << ","
                      << t_sgd        << ","
                      << epoch_s      << ","
                      << gnorm        << ","
                      << 0.0          << ","   // speedup placeholder
                      << 0.0          << "\n"; // efficiency placeholder
            std::cout.flush();
        }

        // All ranks must stay in sync — barrier at end of each epoch
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
