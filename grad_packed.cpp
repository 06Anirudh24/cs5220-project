// grad_packed.cpp — Single-node MLP with flat gradient buffer (MPI-ready baseline)
// Architecture: 784 -> 1024 -> 1024 -> 1024 -> 10
// Compile: g++ -O2 -std=c++17 grad_packed.cpp -o grad_packed
// Run:     ./grad_packed

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
const int   BATCH_SIZE = 256;
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
    // Weights and biases
    std::vector<float> W1, b1;
    std::vector<float> W2, b2;
    std::vector<float> W3, b3;
    std::vector<float> W4, b4;

    // Gradients (backward() writes here; pack_grads() reads from here)
    std::vector<float> dW1, db1;
    std::vector<float> dW2, db2;
    std::vector<float> dW3, db3;
    std::vector<float> dW4, db4;

    // ── Flat gradient buffer ──────────────────────────────────────────────────
    // Layout: dW1 | db1 | dW2 | db2 | dW3 | db3 | dW4 | db4
    // This is the single array that MPI_Allreduce / ring / tree will operate on.
    // Nothing else in the training loop needs to change when MPI is added.
    std::vector<float> grad_buf;
    size_t grad_buf_size = 0;

    // Momentum velocity buffers
    std::vector<float> vW1, vb1;
    std::vector<float> vW2, vb2;
    std::vector<float> vW3, vb3;
    std::vector<float> vW4, vb4;

    // Activations (per batch)
    std::vector<float> z1, a1;
    std::vector<float> z2, a2;
    std::vector<float> z3, a3;
    std::vector<float> z4, a4;

    MLP() {
        // Weights
        W1.resize(INPUT_DIM * H1);   b1.resize(H1, 0.0f);
        W2.resize(H1 * H2);          b2.resize(H2, 0.0f);
        W3.resize(H2 * H3);          b3.resize(H3, 0.0f);
        W4.resize(H3 * OUTPUT_DIM);  b4.resize(OUTPUT_DIM, 0.0f);

        // Gradients
        dW1.resize(INPUT_DIM * H1, 0.0f); db1.resize(H1, 0.0f);
        dW2.resize(H1 * H2, 0.0f);        db2.resize(H2, 0.0f);
        dW3.resize(H2 * H3, 0.0f);        db3.resize(H3, 0.0f);
        dW4.resize(H3 * OUTPUT_DIM, 0.0f); db4.resize(OUTPUT_DIM, 0.0f);

        // Flat buffer
        grad_buf_size = dW1.size() + db1.size()
                      + dW2.size() + db2.size()
                      + dW3.size() + db3.size()
                      + dW4.size() + db4.size();
        grad_buf.resize(grad_buf_size, 0.0f);

        // Velocity buffers
        vW1.resize(INPUT_DIM * H1, 0.0f); vb1.resize(H1, 0.0f);
        vW2.resize(H1 * H2, 0.0f);        vb2.resize(H2, 0.0f);
        vW3.resize(H2 * H3, 0.0f);        vb3.resize(H3, 0.0f);
        vW4.resize(H3 * OUTPUT_DIM, 0.0f); vb4.resize(OUTPUT_DIM, 0.0f);

        xavier_init(W1, INPUT_DIM, H1);
        xavier_init(W2, H1, H2);
        xavier_init(W3, H2, H3);
        xavier_init(W4, H3, OUTPUT_DIM);

        std::cout << "grad_buf_size = " << grad_buf_size
                  << " floats = " << (grad_buf_size * 4) / (1024.0 * 1024.0)
                  << " MB\n";
    }

    // ── Pack: individual grad vectors -> flat buffer ───────────────────────────
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

    // ── Unpack: flat buffer -> individual grad vectors ─────────────────────────
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

    // ── Forward pass ──────────────────────────────────────────────────────────
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

    // ── Backward pass ─────────────────────────────────────────────────────────
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

    // ── Momentum SGD update (reads from individual grad vectors) ──────────────
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

    // ── Accuracy evaluation ───────────────────────────────────────────────────
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
int main() {
    srand(42);

    std::cout << "Loading MNIST...\n";
    auto train_images = load_float_bin(DATA_DIR + "train_images.bin", TRAIN_N * INPUT_DIM);
    auto train_labels = load_u8_bin(DATA_DIR + "train_labels.bin", TRAIN_N);
    auto test_images  = load_float_bin(DATA_DIR + "test_images.bin", TEST_N * INPUT_DIM);
    auto test_labels  = load_u8_bin(DATA_DIR + "test_labels.bin", TEST_N);
    std::cout << "Loaded.\n\n";

    std::cout << "epoch,loss,test_acc,fwd_s,bwd_s,pack_s,sgd_s,epoch_s,grad_norm\n";

    MLP model;

    std::vector<int> idx(TRAIN_N);
    std::iota(idx.begin(), idx.end(), 0);

    int steps_per_epoch = TRAIN_N / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = TRAIN_N - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(idx[i], idx[j]);
        }

        float  epoch_loss = 0.0f;
        double t_fwd = 0, t_bwd = 0, t_pack = 0, t_sgd = 0;

        std::vector<float>   batch_x(BATCH_SIZE * INPUT_DIM);
        std::vector<uint8_t> batch_y(BATCH_SIZE);

        auto epoch_start = Clock::now();

        for (int step = 0; step < steps_per_epoch; step++) {
            for (int b = 0; b < BATCH_SIZE; b++) {
                int s = idx[step * BATCH_SIZE + b];
                std::memcpy(batch_x.data() + b * INPUT_DIM,
                            train_images.data() + s * INPUT_DIM,
                            INPUT_DIM * sizeof(float));
                batch_y[b] = train_labels[s];
            }

            auto t0 = Clock::now();
            float loss = model.forward(batch_x.data(), batch_y.data(), BATCH_SIZE);
            auto t1 = Clock::now();
            model.backward(batch_x.data(), batch_y.data(), BATCH_SIZE);
            auto t2 = Clock::now();

            model.pack_grads();
            // ── MPI ALL-REDUCE GOES HERE ──────────────────────────────────────
            // MPI_Allreduce(MPI_IN_PLACE, model.grad_buf.data(),
            //               model.grad_buf_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            // (then divide by num_ranks if not using MPI_SUM averaging)
            // ring_allreduce(model.grad_buf.data(), model.grad_buf_size);
            // tree_allreduce(model.grad_buf.data(), model.grad_buf_size);
            // ─────────────────────────────────────────────────────────────────
            model.unpack_grads();

            auto t3 = Clock::now();
            model.sgd_step(LR);
            auto t4 = Clock::now();

            epoch_loss += loss;
            t_fwd  += Sec(t1 - t0).count();
            t_bwd  += Sec(t2 - t1).count();
            t_pack += Sec(t3 - t2).count();  // pack + (future: allreduce) + unpack
            t_sgd  += Sec(t4 - t3).count();
        }

        double epoch_s   = Sec(Clock::now() - epoch_start).count();
        float  train_loss = epoch_loss / steps_per_epoch;
        float  test_acc   = model.accuracy(test_images.data(), test_labels.data(), TEST_N);
        float  gnorm      = grad_norm(model.grad_buf);

        std::cout << epoch + 1   << ","
                  << train_loss  << ","
                  << test_acc    << ","
                  << t_fwd       << ","
                  << t_bwd       << ","
                  << t_pack      << ","
                  << t_sgd       << ","
                  << epoch_s     << ","
                  << gnorm       << "\n";
        std::cout.flush();
    }

    return 0;
}
