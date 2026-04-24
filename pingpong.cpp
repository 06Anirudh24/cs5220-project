// pingpong.cpp — MPI ping-pong benchmark to measure alpha (latency) and beta (inverse bandwidth)
// on Perlmutter's Slingshot-11 network.
//
// How it works:
//   Rank 0 sends a message of size N bytes to rank 1.
//   Rank 1 sends it back to rank 0.
//   One round trip = 2 * (alpha + beta * N)
//   We repeat REPS times and take the median to reduce noise.
//   We do this for many message sizes to fit: T = alpha + beta * N
//
// Compile: mpicxx -O2 -std=c++17 pingpong.cpp -o pingpong
// Run:     srun -n 2 --ntasks-per-node=1 ./pingpong

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <numeric>
#include <iomanip>

// ─── Config ───────────────────────────────────────────────────────────────────
// Number of repeated round trips per message size (more = less noise)
const int REPS = 100;

// Warmup round trips before timing (lets MPI warm up connections)
const int WARMUP = 20;

// Message sizes to test (in bytes)
// We sweep from tiny (1 byte, measures pure latency) to large (64 MB, measures bandwidth)
// Includes the sizes relevant to our gradient buffer (~11.1 MB = ~11,665,160 bytes)
const std::vector<size_t> MSG_SIZES = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024,           // 1 KB
    4096,           // 4 KB
    16384,          // 16 KB
    65536,          // 64 KB
    262144,         // 256 KB
    1048576,        // 1 MB
    2097152,        // 2 MB
    4194304,        // 4 MB
    8388608,        // 8 MB
    11665160,       // ~11.1 MB — matches our actual grad_buf_size * sizeof(float)
    16777216,       // 16 MB
    33554432,       // 32 MB
    67108864        // 64 MB
};

// Output file path
const std::string OUT_PATH = "/pscratch/sd/a/anirudh6/cs5220/project/results/pingpong.csv";

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // This benchmark only uses exactly 2 ranks
    if (nranks != 2) {
        if (rank == 0)
            std::cerr << "ERROR: pingpong requires exactly 2 ranks. "
                      << "Got " << nranks << ".\n"
                      << "Run with: srun -n 2 --ntasks-per-node=1 ./pingpong\n";
        MPI_Finalize();
        return 1;
    }

    // Allocate send/recv buffers large enough for the biggest message
    size_t max_size = MSG_SIZES.back();
    std::vector<char> send_buf(max_size, 0);
    std::vector<char> recv_buf(max_size, 0);

    // Fill send buffer with non-zero data so compiler doesn't optimize it away
    for (size_t i = 0; i < max_size; i++) send_buf[i] = (char)(i % 127);

    // Open output file (rank 0 only)
    std::ofstream outfile;
    if (rank == 0) {
        outfile.open(OUT_PATH);
        if (!outfile) {
            std::cerr << "ERROR: Cannot open output file: " << OUT_PATH << "\n";
            MPI_Finalize();
            return 1;
        }

        std::cout << "Ping-pong benchmark on Perlmutter\n";
        std::cout << "Rank 0 <-> Rank 1 (cross-node, --ntasks-per-node=1)\n";
        std::cout << "REPS=" << REPS << "  WARMUP=" << WARMUP << "\n\n";
        std::cout << "Results will be saved to: " << OUT_PATH << "\n\n";

        // CSV header
        outfile << "bytes,"
                << "half_roundtrip_s,"     // median(round_trip) / 2 — one-way latency+BW
                << "roundtrip_s,"          // median round trip time in seconds
                << "bandwidth_GBps,"       // one-way bandwidth in GB/s
                << "reps\n";

        std::cout << std::left
                  << std::setw(15) << "bytes"
                  << std::setw(20) << "half_rtt_us"
                  << std::setw(20) << "bandwidth_GBps"
                  << "\n";
        std::cout << std::string(55, '-') << "\n";
    }

    for (size_t msg_size : MSG_SIZES) {
        // All ranks must agree on message size
        std::vector<double> times(REPS);

        // Synchronize before starting
        MPI_Barrier(MPI_COMM_WORLD);

        // Warmup — don't time these
        for (int r = 0; r < WARMUP; r++) {
            if (rank == 0) {
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Timed round trips
        for (int r = 0; r < REPS; r++) {
            double t_start = MPI_Wtime();  // use MPI_Wtime for consistency

            if (rank == 0) {
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }

            double t_end = MPI_Wtime();
            times[r] = t_end - t_start;  // full round trip time in seconds
        }

        // Rank 0 computes and prints statistics
        if (rank == 0) {
            // Sort for median (robust against outliers)
            std::vector<double> sorted_times = times;
            std::sort(sorted_times.begin(), sorted_times.end());

            double median_rtt    = sorted_times[REPS / 2];
            double half_rtt      = median_rtt / 2.0;        // one-way time
            double half_rtt_us   = half_rtt * 1e6;          // in microseconds
            double bandwidth     = msg_size / median_rtt / 1e9;  // one-way GB/s

            // Also compute mean and min for reference
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean_rtt = sum / REPS;
            double min_rtt  = sorted_times[0];

            outfile << msg_size         << ","
                    << half_rtt         << ","
                    << median_rtt       << ","
                    << bandwidth        << ","
                    << REPS             << "\n";
            outfile.flush();

            std::cout << std::left
                      << std::setw(15) << msg_size
                      << std::setw(20) << half_rtt_us
                      << std::setw(20) << bandwidth
                      << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        outfile.close();
        std::cout << "\nDone. Results saved to: " << OUT_PATH << "\n";
        std::cout << "\nNext step: run fit_alpha_beta.py to extract alpha and beta.\n";
    }

    MPI_Finalize();
    return 0;
}
