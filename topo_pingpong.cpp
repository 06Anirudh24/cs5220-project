// topo_pingpong.cpp — Multi-pair ping-pong benchmark for Dragonfly topology analysis
//
// Purpose:
//   Measures point-to-point bandwidth and latency between rank pairs at
//   different positions in the job allocation. On Perlmutter's Dragonfly
//   network, pairs that happen to land in the same Dragonfly group see
//   lower latency than pairs that span group boundaries. By running many
//   pairs simultaneously and comparing their RTTs, we can infer whether
//   topology effects are responsible for the gap between our alpha-beta
//   predictions and measured ring all-reduce performance.
//
// How it works:
//   - Requires an EVEN number of ranks (2, 4, 8, 16, 32)
//   - Pairs ranks as: (0,1), (2,3), (4,5), ...
//   - Each pair runs an independent ping-pong benchmark simultaneously
//   - Rank 0 collects and prints all results
//   - We run this at multiple rank counts to see how RTT changes as
//     ranks span more of the Dragonfly fabric
//
// Compile: mpicxx -O2 -std=c++17 topo_pingpong.cpp -o topo_pingpong
// Run:     srun -n <even_ranks> --ntasks-per-node=1 ./topo_pingpong

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <sstream>

// ─── Config ───────────────────────────────────────────────────────────────────
const int REPS   = 200;   // round trips per message size per pair
const int WARMUP = 30;    // warmup round trips before timing

// Message sizes to test — we focus on sizes relevant to our gradient buffer
// Small: measure latency (alpha regime)
// Large: measure bandwidth (beta regime) — 11.1 MB matches our grad_buf
const std::vector<size_t> MSG_SIZES = {
    64,           // tiny — pure latency
    4096,         // 4 KB
    65536,        // 64 KB
    1048576,      // 1 MB
    4194304,      // 4 MB
    11665160,     // ~11.1 MB — matches grad_buf_size * sizeof(float)
    33554432,     // 32 MB — bandwidth saturation check
};

const std::string OUT_DIR = "/pscratch/sd/a/anirudh6/cs5220/project/results/";

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (nranks % 2 != 0) {
        if (rank == 0)
            std::cerr << "ERROR: topo_pingpong requires an even number of ranks.\n"
                      << "Got " << nranks << ". Use 2, 4, 8, 16, or 32.\n";
        MPI_Finalize();
        return 1;
    }

    // ── Pair assignment ────────────────────────────────────────────────────────
    // Pair ranks sequentially: (0,1), (2,3), (4,5), ...
    // Within each pair, the lower rank is the "sender" (initiates ping)
    int pair_id     = rank / 2;           // which pair this rank belongs to
    int pair_role   = rank % 2;           // 0 = sender (even), 1 = receiver (odd)
    int partner     = (pair_role == 0) ? rank + 1 : rank - 1;
    int num_pairs   = nranks / 2;

    // ── Hostname collection ────────────────────────────────────────────────────
    // Collect hostnames so we can see which pairs are on which nodes
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int  hostname_len;
    MPI_Get_processor_name(hostname, &hostname_len);

    // Gather all hostnames to rank 0
    std::vector<char> all_hostnames(nranks * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    // ── Allocate buffers ───────────────────────────────────────────────────────
    size_t max_size = MSG_SIZES.back();
    std::vector<char> send_buf(max_size, 0);
    std::vector<char> recv_buf(max_size, 0);
    for (size_t i = 0; i < max_size; i++) send_buf[i] = (char)(i % 127);

    // Results storage: pair x message_size -> half_rtt in seconds
    // Each sender rank (pair_role==0) will store its own results
    // then we gather to rank 0
    int n_sizes = (int)MSG_SIZES.size();
    std::vector<double> my_results(n_sizes, 0.0);  // half RTT per size
    std::vector<double> my_min_rtts(n_sizes, 0.0);
    std::vector<double> my_stddev(n_sizes, 0.0);

    // ── Output file setup (rank 0 only) ────────────────────────────────────────
    std::ofstream outfile;
    std::string outpath = OUT_DIR + "topo_pingpong_nranks" +
                          std::to_string(nranks) + ".csv";

    if (rank == 0) {
        outfile.open(outpath);
        if (!outfile) {
            std::cerr << "ERROR: Cannot open " << outpath << "\n";
            MPI_Finalize();
            return 1;
        }
        outfile << "nranks,pair_id,rank_a,rank_b,node_a,node_b,"
                << "bytes,half_rtt_s,half_rtt_us,bandwidth_GBps,"
                << "min_rtt_s,stddev_s\n";

        std::cout << "\n=== Topology Ping-Pong Benchmark ===\n";
        std::cout << "nranks:       " << nranks << "\n";
        std::cout << "num_pairs:    " << num_pairs << "\n";
        std::cout << "REPS:         " << REPS << "\n";
        std::cout << "WARMUP:       " << WARMUP << "\n";
        std::cout << "msg_sizes:    " << n_sizes << " sizes from "
                  << MSG_SIZES[0] << "B to " << MSG_SIZES.back()/1e6 << "MB\n\n";

        std::cout << "Rank pairs and nodes:\n";
        for (int p = 0; p < num_pairs; p++) {
            std::string node_a(all_hostnames.data() + (2*p)   * MPI_MAX_PROCESSOR_NAME);
            std::string node_b(all_hostnames.data() + (2*p+1) * MPI_MAX_PROCESSOR_NAME);
            std::cout << "  pair " << p << ": rank " << 2*p
                      << " (" << node_a << ") <-> rank " << 2*p+1
                      << " (" << node_b << ")\n";
        }
        std::cout << "\n";
        std::cout << std::left
                  << std::setw(8)  << "pair"
                  << std::setw(12) << "bytes"
                  << std::setw(16) << "half_rtt_us"
                  << std::setw(16) << "bw_GBps"
                  << std::setw(16) << "stddev_us"
                  << "\n";
        std::cout << std::string(68, '-') << "\n";
    }

    // ── Run benchmark for each message size ────────────────────────────────────
    for (int si = 0; si < n_sizes; si++) {
        size_t msg_size = MSG_SIZES[si];
        std::vector<double> rtts(REPS);

        // All pairs synchronize before each message size
        MPI_Barrier(MPI_COMM_WORLD);

        // Warmup
        for (int r = 0; r < WARMUP; r++) {
            if (pair_role == 0) {
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Timed round trips
        for (int r = 0; r < REPS; r++) {
            double t0 = MPI_Wtime();
            if (pair_role == 0) {
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 1, MPI_COMM_WORLD);
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(recv_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buf.data(), (int)msg_size, MPI_BYTE,
                         partner, 1, MPI_COMM_WORLD);
            }
            rtts[r] = MPI_Wtime() - t0;
        }

        // Sender computes stats
        if (pair_role == 0) {
            std::vector<double> sorted = rtts;
            std::sort(sorted.begin(), sorted.end());

            double median_rtt = sorted[REPS / 2];
            double min_rtt    = sorted[0];
            double half_rtt   = median_rtt / 2.0;

            // Standard deviation
            double mean = 0.0;
            for (double v : rtts) mean += v;
            mean /= REPS;
            double var = 0.0;
            for (double v : rtts) var += (v - mean) * (v - mean);
            double stddev = std::sqrt(var / REPS) / 2.0;  // half-rtt stddev

            my_results[si]   = half_rtt;
            my_min_rtts[si]  = min_rtt / 2.0;
            my_stddev[si]    = stddev;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ── Gather results from all sender ranks to rank 0 ─────────────────────────
    // Only sender ranks (even ranks) have meaningful data
    // Gather half_rtt, min_rtt, stddev arrays from every rank
    std::vector<double> all_results(nranks * n_sizes, 0.0);
    std::vector<double> all_min(nranks * n_sizes, 0.0);
    std::vector<double> all_std(nranks * n_sizes, 0.0);

    MPI_Gather(my_results.data(),  n_sizes, MPI_DOUBLE,
               all_results.data(), n_sizes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(my_min_rtts.data(), n_sizes, MPI_DOUBLE,
               all_min.data(),     n_sizes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(my_stddev.data(),   n_sizes, MPI_DOUBLE,
               all_std.data(),     n_sizes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Rank 0 writes all results ──────────────────────────────────────────────
    if (rank == 0) {
        for (int p = 0; p < num_pairs; p++) {
            int sender_rank = 2 * p;
            std::string node_a(all_hostnames.data() + sender_rank       * MPI_MAX_PROCESSOR_NAME);
            std::string node_b(all_hostnames.data() + (sender_rank + 1) * MPI_MAX_PROCESSOR_NAME);

            for (int si = 0; si < n_sizes; si++) {
                double half_rtt = all_results[sender_rank * n_sizes + si];
                double min_rtt  = all_min[sender_rank * n_sizes + si];
                double stddev   = all_std[sender_rank * n_sizes + si];
                double bw       = MSG_SIZES[si] / (half_rtt * 2.0) / 1e9;
                double half_us  = half_rtt * 1e6;
                double std_us   = stddev * 1e6;

                outfile << nranks        << ","
                        << p            << ","
                        << sender_rank  << ","
                        << sender_rank+1 << ","
                        << node_a       << ","
                        << node_b       << ","
                        << MSG_SIZES[si] << ","
                        << half_rtt     << ","
                        << half_us      << ","
                        << bw           << ","
                        << min_rtt      << ","
                        << stddev       << "\n";
                outfile.flush();

                // Print to stdout for the largest message size (most informative)
                if (si == n_sizes - 2) {  // 11.1 MB row
                    std::cout << std::left
                              << std::setw(8)  << p
                              << std::setw(12) << MSG_SIZES[si]
                              << std::setw(16) << std::fixed << std::setprecision(2) << half_us
                              << std::setw(16) << bw
                              << std::setw(16) << std_us
                              << "\n";
                }
            }
        }

        outfile.close();
        std::cout << "\nFull results saved to: " << outpath << "\n";
        std::cout << "(All message sizes for all pairs in CSV)\n";
    }

    MPI_Finalize();
    return 0;
}
