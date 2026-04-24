// ring_allreduce.cpp — Data-parallel MLP with hand-written ring all-reduce
// Architecture: 784 -> 1024 -> 1024 -> 1024 -> 10
// Compile: mpicxx -O2 -std=c++17 ring_allreduce.cpp -o ring_allreduce
// Run:     srun -n <ranks> ./ring_allreduce
//
// Ring all-reduce algorithm:
//   Phase 1 — Reduce-Scatter: P-1 steps, each rank accumulates one chunk
//   Phase 2 — All-Gather:     P-1 steps, each rank broadcasts its chunk
//   Total communication per rank: 2*(P-1)/P * m bytes (bandwidth optimal)

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

// ─── Weight init ──────────────────────────────────────────────────────────────
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

// ─── Matrix multiply ──────────────────────────────────────────────────────────
void matmul(const float* A, const float* B, float* out, int M, int K, int N) {
    std::fill(out, out + M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float aik = A[i * K + k];
            for (int j = 0; j < N; j++)
                out[i * N + j] += aik * B[k * N + j];
        }
}

// ─── Gradient norm ────────────────────────────────────────────────────────────
float grad_norm(const std::vector<float>& buf) {
    double sum = 0.0;
    for (float v : buf) sum += (double)v * v;
    return (float)std::sqrt(sum);
}

// ─── Ring All-Reduce ──────────────────────────────────────────────────────────
// Operates on buf[0..n-1] in-place.
// After this call every rank holds the average of all ranks' input buffers.
//
// Returns: time spent in MPI communication only (seconds)
double ring_allreduce(float* buf, size_t n, int rank, int nranks) {
    if (nranks == 1) return 0.0;  // nothing to do

    int send_to   = (rank + 1) % nranks;           // right neighbor
    int recv_from = (rank - 1 + nranks) % nranks;  // left neighbor

    // Compute chunk sizes. Last chunk absorbs any remainder.
    size_t base_chunk = n / nranks;
    size_t remainder  = n % nranks;

    // chunk_start[r] = start index of chunk r in buf
    // chunk_size[r]  = number of floats in chunk r
    std::vector<size_t> chunk_start(nranks), chunk_size(nranks);
    for (int r = 0; r < nranks; r++) {
        chunk_size[r]  = base_chunk + (r == nranks - 1 ? remainder : 0);
        chunk_start[r] = r * base_chunk;
    }

    // Temporary recv buffer — large enough for the biggest chunk
    size_t max_chunk = base_chunk + remainder;
    std::vector<float> recv_buf(max_chunk);

    double t_comm = 0.0;

    // ── Phase 1: Reduce-Scatter (P-1 steps) ───────────────────────────────────
    // In step i, rank r sends chunk (r-i+P)%P and receives into chunk (r-i-1+P)%P
    // then accumulates: buf[recv_chunk] += recv_buf
    for (int step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank - step + nranks) % nranks;
        int recv_chunk = (rank - step - 1 + nranks) % nranks;

        float* send_ptr = buf + chunk_start[send_chunk];
        float* recv_ptr = recv_buf.data();
        int    send_cnt = (int)chunk_size[send_chunk];
        int    recv_cnt = (int)chunk_size[recv_chunk];

        auto t0 = Clock::now();
        MPI_Sendrecv(send_ptr, send_cnt, MPI_FLOAT, send_to,   0,
                     recv_ptr, recv_cnt, MPI_FLOAT, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t_comm += Sec(Clock::now() - t0).count();

        // Accumulate received data into buf
        float* acc_ptr = buf + chunk_start[recv_chunk];
        for (int i = 0; i < recv_cnt; i++)
            acc_ptr[i] += recv_ptr[i];
    }

    // After reduce-scatter: rank r owns the fully summed chunk r.
    // Now average: divide chunk r by nranks.
    {
        float* my_chunk = buf + chunk_start[rank];
        int    my_size  = (int)chunk_size[rank];
        float  inv_n    = 1.0f / nranks;
        for (int i = 0; i < my_size; i++)
            my_chunk[i] *= inv_n;
    }

    // ── Phase 2: All-Gather (P-1 steps) ───────────────────────────────────────
    // In step i, rank r sends chunk (r-i+1+P)%P (already averaged) and
    // receives chunk (r-i+P)%P. No accumulation — just copy into place.
    for (int step = 0; step < nranks - 1; step++) {
        int send_chunk = (rank - step + 1 + nranks) % nranks;
        int recv_chunk = (rank - step + nranks) % nranks;

        float* send_ptr = buf + chunk_start[send_chunk];
        float* recv_ptr = buf + chunk_start[recv_chunk];  // write directly into buf
        int    send_cnt = (int)chunk_size[send_chunk];
        int    recv_cnt = (int)chunk_size[recv_chunk];

        auto t0 = Clock::now();
        MPI_Sendrecv(send_ptr, send_cnt, MPI_FLOAT, send_to,   1,
                     recv_ptr, recv_cnt, MPI_FLOAT, recv_from, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t_comm += Sec(Clock::now() - t0).count();
    }

    // All ranks now hold the averaged gradient buffer.
    return t_comm;
}

// ─── MLP ──────────────────────────────────────────────────────────────────────
struct MLP {
    std::vector<float> W1, b1, W2, b2, W3, b3, W4, b4;
    std::vector<float> dW1, db1, dW2, db2, dW3, db3, dW4, db4;
    std::vector<float> grad_buf;
    size_t grad_buf_size = 0;
    std::vector<float> vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4;
    std::vector<float> z1, a1, z2, a2, z3, a3, z4, a4;

    MLP() {
        W1.resize(INPUT_DIM*H1); b1.resize(H1,0);
        W2.resize(H1*H2);        b2.resize(H2,0);
        W3.resize(H2*H3);        b3.resize(H3,0);
        W4.resize(H3*OUTPUT_DIM); b4.resize(OUTPUT_DIM,0);

        dW1.resize(INPUT_DIM*H1,0); db1.resize(H1,0);
        dW2.resize(H1*H2,0);        db2.resize(H2,0);
        dW3.resize(H2*H3,0);        db3.resize(H3,0);
        dW4.resize(H3*OUTPUT_DIM,0); db4.resize(OUTPUT_DIM,0);

        grad_buf_size = dW1.size()+db1.size()+dW2.size()+db2.size()
                      + dW3.size()+db3.size()+dW4.size()+db4.size();
        grad_buf.resize(grad_buf_size, 0.0f);

        vW1.resize(INPUT_DIM*H1,0); vb1.resize(H1,0);
        vW2.resize(H1*H2,0);        vb2.resize(H2,0);
        vW3.resize(H2*H3,0);        vb3.resize(H3,0);
        vW4.resize(H3*OUTPUT_DIM,0); vb4.resize(OUTPUT_DIM,0);

        xavier_init(W1,INPUT_DIM,H1);
        xavier_init(W2,H1,H2);
        xavier_init(W3,H2,H3);
        xavier_init(W4,H3,OUTPUT_DIM);
    }

    void pack_grads() {
        float* p = grad_buf.data();
        auto cp = [&](const std::vector<float>& v){
            std::memcpy(p,v.data(),v.size()*sizeof(float)); p+=v.size();
        };
        cp(dW1);cp(db1);cp(dW2);cp(db2);cp(dW3);cp(db3);cp(dW4);cp(db4);
    }

    void unpack_grads() {
        const float* p = grad_buf.data();
        auto cp = [&](std::vector<float>& v){
            std::memcpy(v.data(),p,v.size()*sizeof(float)); p+=v.size();
        };
        cp(dW1);cp(db1);cp(dW2);cp(db2);cp(dW3);cp(db3);cp(dW4);cp(db4);
    }

    float forward(const float* x, const uint8_t* labels, int bs) {
        z1.resize(bs*H1); a1.resize(bs*H1);
        z2.resize(bs*H2); a2.resize(bs*H2);
        z3.resize(bs*H3); a3.resize(bs*H3);
        z4.resize(bs*OUTPUT_DIM); a4.resize(bs*OUTPUT_DIM);

        matmul(x,W1.data(),z1.data(),bs,INPUT_DIM,H1);
        for(int i=0;i<bs;i++) for(int j=0;j<H1;j++) z1[i*H1+j]+=b1[j];
        a1=z1; relu(a1);

        matmul(a1.data(),W2.data(),z2.data(),bs,H1,H2);
        for(int i=0;i<bs;i++) for(int j=0;j<H2;j++) z2[i*H2+j]+=b2[j];
        a2=z2; relu(a2);

        matmul(a2.data(),W3.data(),z3.data(),bs,H2,H3);
        for(int i=0;i<bs;i++) for(int j=0;j<H3;j++) z3[i*H3+j]+=b3[j];
        a3=z3; relu(a3);

        matmul(a3.data(),W4.data(),z4.data(),bs,H3,OUTPUT_DIM);
        for(int i=0;i<bs;i++) for(int j=0;j<OUTPUT_DIM;j++) z4[i*OUTPUT_DIM+j]+=b4[j];
        a4=z4;
        for(int i=0;i<bs;i++) softmax(&a4[i*OUTPUT_DIM],OUTPUT_DIM);

        float loss=0.0f;
        for(int i=0;i<bs;i++)
            loss -= std::log(a4[i*OUTPUT_DIM+labels[i]]+1e-9f);
        return loss/bs;
    }

    void backward(const float* x, const uint8_t* labels, int bs) {
        std::fill(dW1.begin(),dW1.end(),0); std::fill(db1.begin(),db1.end(),0);
        std::fill(dW2.begin(),dW2.end(),0); std::fill(db2.begin(),db2.end(),0);
        std::fill(dW3.begin(),dW3.end(),0); std::fill(db3.begin(),db3.end(),0);
        std::fill(dW4.begin(),dW4.end(),0); std::fill(db4.begin(),db4.end(),0);

        float scale = 1.0f/bs;

        std::vector<float> d4(bs*OUTPUT_DIM);
        for(int i=0;i<bs;i++){
            for(int j=0;j<OUTPUT_DIM;j++) d4[i*OUTPUT_DIM+j]=a4[i*OUTPUT_DIM+j];
            d4[i*OUTPUT_DIM+labels[i]]-=1.0f;
        }
        for(int i=0;i<bs;i++){
            for(int j=0;j<H3;j++)
                for(int k=0;k<OUTPUT_DIM;k++)
                    dW4[j*OUTPUT_DIM+k]+=a3[i*H3+j]*d4[i*OUTPUT_DIM+k];
            for(int k=0;k<OUTPUT_DIM;k++) db4[k]+=d4[i*OUTPUT_DIM+k];
        }

        std::vector<float> d3(bs*H3);
        for(int i=0;i<bs;i++)
            for(int j=0;j<H3;j++){
                float val=0;
                for(int k=0;k<OUTPUT_DIM;k++) val+=d4[i*OUTPUT_DIM+k]*W4[j*OUTPUT_DIM+k];
                d3[i*H3+j]=val*(z3[i*H3+j]>0?1.0f:0.0f);
            }
        for(int i=0;i<bs;i++){
            for(int j=0;j<H2;j++)
                for(int k=0;k<H3;k++) dW3[j*H3+k]+=a2[i*H2+j]*d3[i*H3+k];
            for(int k=0;k<H3;k++) db3[k]+=d3[i*H3+k];
        }

        std::vector<float> d2(bs*H2);
        for(int i=0;i<bs;i++)
            for(int j=0;j<H2;j++){
                float val=0;
                for(int k=0;k<H3;k++) val+=d3[i*H3+k]*W3[j*H3+k];
                d2[i*H2+j]=val*(z2[i*H2+j]>0?1.0f:0.0f);
            }
        for(int i=0;i<bs;i++){
            for(int j=0;j<H1;j++)
                for(int k=0;k<H2;k++) dW2[j*H2+k]+=a1[i*H1+j]*d2[i*H2+k];
            for(int k=0;k<H2;k++) db2[k]+=d2[i*H2+k];
        }

        std::vector<float> d1(bs*H1);
        for(int i=0;i<bs;i++)
            for(int j=0;j<H1;j++){
                float val=0;
                for(int k=0;k<H2;k++) val+=d2[i*H2+k]*W2[j*H2+k];
                d1[i*H1+j]=val*(z1[i*H1+j]>0?1.0f:0.0f);
            }
        for(int i=0;i<bs;i++){
            for(int j=0;j<INPUT_DIM;j++)
                for(int k=0;k<H1;k++) dW1[j*H1+k]+=x[i*INPUT_DIM+j]*d1[i*H1+k];
            for(int k=0;k<H1;k++) db1[k]+=d1[i*H1+k];
        }

        for(auto& v:dW1) v*=scale; for(auto& v:db1) v*=scale;
        for(auto& v:dW2) v*=scale; for(auto& v:db2) v*=scale;
        for(auto& v:dW3) v*=scale; for(auto& v:db3) v*=scale;
        for(auto& v:dW4) v*=scale; for(auto& v:db4) v*=scale;
    }

    void sgd_step(float lr) {
        auto upd = [&](std::vector<float>& w, std::vector<float>& dw,
                       std::vector<float>& v){
            for(size_t i=0;i<w.size();i++){
                v[i]=MOMENTUM*v[i]-lr*dw[i];
                w[i]+=v[i];
            }
        };
        upd(W1,dW1,vW1); upd(b1,db1,vb1);
        upd(W2,dW2,vW2); upd(b2,db2,vb2);
        upd(W3,dW3,vW3); upd(b3,db3,vb3);
        upd(W4,dW4,vW4); upd(b4,db4,vb4);
    }

    float accuracy(const float* x, const uint8_t* labels, int n) {
        int correct=0;
        std::vector<float> out(OUTPUT_DIM),h1(H1),h2(H2),h3(H3);
        for(int i=0;i<n;i++){
            const float* xi=x+i*INPUT_DIM;
            for(int j=0;j<H1;j++){
                float v=b1[j];
                for(int k=0;k<INPUT_DIM;k++) v+=xi[k]*W1[k*H1+j];
                h1[j]=std::max(0.0f,v);
            }
            for(int j=0;j<H2;j++){
                float v=b2[j];
                for(int k=0;k<H1;k++) v+=h1[k]*W2[k*H2+j];
                h2[j]=std::max(0.0f,v);
            }
            for(int j=0;j<H3;j++){
                float v=b3[j];
                for(int k=0;k<H2;k++) v+=h2[k]*W3[k*H3+j];
                h3[j]=std::max(0.0f,v);
            }
            for(int j=0;j<OUTPUT_DIM;j++){
                float v=b4[j];
                for(int k=0;k<H3;k++) v+=h3[k]*W4[k*OUTPUT_DIM+j];
                out[j]=v;
            }
            int pred=std::max_element(out.begin(),out.end())-out.begin();
            if(pred==labels[i]) correct++;
        }
        return (float)correct/n;
    }
};

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    srand(42);

    if (rank == 0) std::cout << "Loading MNIST...\n";
    auto train_images = load_float_bin(DATA_DIR+"train_images.bin", TRAIN_N*INPUT_DIM);
    auto train_labels = load_u8_bin(DATA_DIR+"train_labels.bin", TRAIN_N);
    auto test_images  = load_float_bin(DATA_DIR+"test_images.bin", TEST_N*INPUT_DIM);
    auto test_labels  = load_u8_bin(DATA_DIR+"test_labels.bin", TEST_N);
    if (rank == 0) std::cout << "Loaded.\n\n";

    if (BATCH_SIZE % nranks != 0) {
        if (rank == 0)
            std::cerr << "ERROR: BATCH_SIZE (" << BATCH_SIZE
                      << ") must be divisible by nranks (" << nranks << ")\n";
        MPI_Finalize(); return 1;
    }
    int local_bs = BATCH_SIZE / nranks;

    if (rank == 0) {
        std::cout << "=== Ring All-Reduce run ===\n";
        std::cout << "nranks:        " << nranks     << "\n";
        std::cout << "global_batch:  " << BATCH_SIZE << "\n";
        std::cout << "local_batch:   " << local_bs   << "\n";
        std::cout << "epochs:        " << EPOCHS     << "\n";
        std::cout << "lr:            " << LR         << "\n";
        std::cout << "momentum:      " << MOMENTUM   << "\n\n";
    }

    MLP model;

    if (rank == 0) {
        std::cout << "grad_buf_size: " << model.grad_buf_size
                  << " floats = "
                  << (model.grad_buf_size * 4) / (1024.0 * 1024.0)
                  << " MB\n\n";

        // CSV header
        // allreduce_s   = time inside ring_allreduce() — MPI calls only
        // comm_s        = allreduce_s as fraction of epoch (communication overhead)
        // bytes_sent    = theoretical bytes sent per epoch per rank
        std::cout << "epoch,"
                  << "loss,"
                  << "test_acc,"
                  << "fwd_s,"
                  << "bwd_s,"
                  << "allreduce_s,"   // time in ring all-reduce MPI calls
                  << "sgd_s,"
                  << "epoch_s,"
                  << "grad_norm,"
                  << "comm_fraction," // allreduce_s / epoch_s
                  << "bytes_per_rank\n"; // 2*(P-1)/P * grad_buf_size * 4
    }

    std::vector<int> idx(TRAIN_N);
    std::iota(idx.begin(), idx.end(), 0);

    int steps_per_epoch = TRAIN_N / BATCH_SIZE;

    // Theoretical bytes sent per rank per step (alpha-beta model)
    // Ring sends 2*(P-1)/P * m bytes per rank
    double theory_bytes_per_step = 2.0 * (nranks - 1.0) / nranks
                                 * model.grad_buf_size * sizeof(float);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        srand(42 + epoch);
        for (int i = TRAIN_N-1; i > 0; i--) {
            int j = rand() % (i+1);
            std::swap(idx[i], idx[j]);
        }

        float  epoch_loss  = 0.0f;
        double t_fwd       = 0.0;
        double t_bwd       = 0.0;
        double t_allreduce = 0.0;
        double t_sgd       = 0.0;

        std::vector<float>   local_x(local_bs * INPUT_DIM);
        std::vector<uint8_t> local_y(local_bs);

        auto epoch_start = Clock::now();

        for (int step = 0; step < steps_per_epoch; step++) {
            int global_offset = step * BATCH_SIZE + rank * local_bs;
            for (int b = 0; b < local_bs; b++) {
                int s = idx[global_offset + b];
                std::memcpy(local_x.data() + b*INPUT_DIM,
                            train_images.data() + s*INPUT_DIM,
                            INPUT_DIM * sizeof(float));
                local_y[b] = train_labels[s];
            }

            // Forward
            auto t0 = Clock::now();
            float loss = model.forward(local_x.data(), local_y.data(), local_bs);
            auto t1 = Clock::now();

            // Backward
            model.backward(local_x.data(), local_y.data(), local_bs);
            auto t2 = Clock::now();

            // Pack -> Ring All-Reduce -> Unpack
            model.pack_grads();
            double comm_time = ring_allreduce(
                model.grad_buf.data(), model.grad_buf_size, rank, nranks);
            model.unpack_grads();
            auto t3 = Clock::now();

            // SGD
            model.sgd_step(LR);
            auto t4 = Clock::now();

            epoch_loss   += loss;
            t_fwd        += Sec(t1 - t0).count();
            t_bwd        += Sec(t2 - t1).count();
            t_allreduce  += comm_time;                    // MPI time only
            t_sgd        += Sec(t4 - t3).count();
        }

        double epoch_s    = Sec(Clock::now() - epoch_start).count();
        float  train_loss = epoch_loss / steps_per_epoch;

        if (rank == 0) {
            float  test_acc      = model.accuracy(test_images.data(),
                                                   test_labels.data(), TEST_N);
            float  gnorm         = grad_norm(model.grad_buf);
            double comm_fraction = t_allreduce / epoch_s;
            double bytes_sent    = theory_bytes_per_step * steps_per_epoch;

            std::cout << epoch + 1      << ","
                      << train_loss     << ","
                      << test_acc       << ","
                      << t_fwd          << ","
                      << t_bwd          << ","
                      << t_allreduce    << ","
                      << t_sgd          << ","
                      << epoch_s        << ","
                      << gnorm          << ","
                      << comm_fraction  << ","
                      << bytes_sent     << "\n";
            std::cout.flush();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
