// pso_ev_clean.cpp
// PSO-based EV charging optimization
// Serial vs OpenMP-parallel with multiple swarm sizes & thread counts
//
// Compile (MinGW-w64 + OpenMP):
//   g++ -O2 -fopenmp pso_ev_clean.cpp -o pso_ev_clean

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::high_resolution_clock;
using dbl = double;

// ----------------- Problem Definition -----------------
struct Problem {
    int M;                  // number of EVs
    int SLOTS;              // time slots
    dbl GRID_MAX;           // grid max kW
    vector<dbl> price;      // price per slot
    vector<dbl> E_req;      // required energy per EV (kWh)
    vector<dbl> max_power;  // max charging rate per EV per slot (kW)
    dbl alpha;              // overload penalty weight
    dbl beta;               // unmet energy penalty weight

    Problem(int M_ = 10)
        : M(M_),
          SLOTS(24),
          GRID_MAX(50.0),
          alpha(10000.0),
          beta(10000.0)
    {
        price.assign(SLOTS, 0.1);
        // sample price curve: cheap at night, more expensive during the day
        for (int t = 0; t < SLOTS; ++t) {
            if (t >= 17 && t <= 20) price[t] = 0.5;       // peak
            else if (t >= 7 && t <= 16) price[t] = 0.2;   // normal
            else price[t] = 0.08;                         // off-peak
        }
        E_req.assign(M, 8.0);       // each EV needs 8 kWh
        max_power.assign(M, 7.0);   // up to 7 kW per hour
    }
};

// ----------------- Random Helpers -----------------
// Thread-safe RNG for serial & OpenMP
static double rand01() {
#ifdef _OPENMP
    static thread_local std::mt19937 rng(1234 + omp_get_thread_num());
#else
    static thread_local std::mt19937 rng(1234);
#endif
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

static double uniform_rand(double a, double b) {
    return a + (b - a) * rand01();
}

// ----------------- Particle Structure -----------------
struct Particle {
    vector<dbl> pos;
    vector<dbl> vel;
    vector<dbl> pbest_pos;
    dbl pbest_val;

    Particle(int dim = 0) {
        pos.assign(dim, 0.0);
        vel.assign(dim, 0.0);
        pbest_pos.assign(dim, 0.0);
        pbest_val = 1e300;
    }
};

// ----------------- Fitness Evaluation -----------------
dbl evaluate_fitness(const vector<dbl>& pos, const Problem& P) {
    int M = P.M, S = P.SLOTS;
    vector<dbl> total_power(S, 0.0);
    vector<dbl> delivered(M, 0.0);

    // Aggregate EV power and delivered energy
    for (int j = 0; j < M; ++j) {
        for (int t = 0; t < S; ++t) {
            dbl p = pos[j * S + t];
            total_power[t] += p;
            delivered[j] += p;
        }
    }

    // Energy cost
    dbl total_cost = 0.0;
    for (int t = 0; t < S; ++t) {
        total_cost += P.price[t] * total_power[t];
    }

    // Grid overload penalty
    dbl overload = 0.0;
    for (int t = 0; t < S; ++t) {
        if (total_power[t] > P.GRID_MAX) {
            dbl diff = total_power[t] - P.GRID_MAX;
            overload += diff * diff;
        }
    }

    // Unmet energy penalty
    dbl unmet = 0.0;
    for (int j = 0; j < M; ++j) {
        if (delivered[j] < P.E_req[j]) {
            dbl d = P.E_req[j] - delivered[j];
            unmet += d * d;
        }
    }

    return total_cost + P.alpha * overload + P.beta * unmet;
}

// ----------------- Constraint Repair -----------------
void repair_particle(vector<dbl>& pos, const Problem& P) {
    int M = P.M, S = P.SLOTS;
    for (int j = 0; j < M; ++j) {
        dbl sum = 0.0;

        // Enforce bounds and sum energy
        for (int t = 0; t < S; ++t) {
            dbl &x = pos[j * S + t];
            if (x < 0.0) x = 0.0;
            if (x > P.max_power[j]) x = P.max_power[j];
            sum += x;
        }

        if (sum > 1e-12) {
            // Scale to match required energy
            dbl scale = P.E_req[j] / sum;
            for (int t = 0; t < S; ++t) {
                dbl &x = pos[j * S + t];
                x *= scale;
                if (x > P.max_power[j]) x = P.max_power[j];
            }
        } else {
            // Distribute required energy if zero or near-zero
            dbl remain = P.E_req[j];
            for (int t = 0; t < S && remain > 1e-12; ++t) {
                dbl add = min(P.max_power[j], remain);
                pos[j * S + t] = add;
                remain -= add;
            }
        }
    }
}

// ----------------- Swarm Initialization -----------------
void initialize_swarm(vector<Particle>& swarm, int swarm_size, const Problem& P) {
    int dim = P.M * P.SLOTS;
    swarm.clear();
    swarm.resize(swarm_size, Particle(dim));

    for (int i = 0; i < swarm_size; ++i) {
        for (int k = 0; k < dim; ++k) {
            int j = k / P.SLOTS;
            swarm[i].pos[k] = uniform_rand(0.0, P.max_power[j]);
            swarm[i].vel[k] = uniform_rand(-P.max_power[j] * 0.1, P.max_power[j] * 0.1);
            swarm[i].pbest_pos[k] = swarm[i].pos[k];
        }
        repair_particle(swarm[i].pos, P);
        swarm[i].pbest_val = evaluate_fitness(swarm[i].pos, P);
        swarm[i].pbest_pos = swarm[i].pos;
    }
}

// ----------------- Serial PSO -----------------
pair<vector<dbl>, dbl> pso_run_serial(int swarm_size, int max_iter, const Problem& P) {
    int dim = P.M * P.SLOTS;
    const double w = 0.729;
    const double c1 = 1.49445;
    const double c2 = 1.49445;

    vector<Particle> swarm;
    initialize_swarm(swarm, swarm_size, P);

    dbl gbest_val = 1e300;
    vector<dbl> gbest_pos(dim, 0.0);

    for (int i = 0; i < swarm_size; ++i) {
        if (swarm[i].pbest_val < gbest_val) {
            gbest_val = swarm[i].pbest_val;
            gbest_pos = swarm[i].pbest_pos;
        }
    }

    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < swarm_size; ++i) {
            dbl val = evaluate_fitness(swarm[i].pos, P);
            if (val < swarm[i].pbest_val) {
                swarm[i].pbest_val = val;
                swarm[i].pbest_pos = swarm[i].pos;
                if (val < gbest_val) {
                    gbest_val = val;
                    gbest_pos = swarm[i].pos;
                }
            }

            // Velocity & position update
            for (int k = 0; k < dim; ++k) {
                double r1 = rand01();
                double r2 = rand01();
                swarm[i].vel[k] = w * swarm[i].vel[k]
                    + c1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k])
                    + c2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);
                swarm[i].pos[k] += swarm[i].vel[k];
            }
            repair_particle(swarm[i].pos, P);
        }
    }

    return { gbest_pos, gbest_val };
}

// ----------------- Parallel PSO (OpenMP) -----------------
pair<vector<dbl>, dbl> pso_run_parallel(int swarm_size, int max_iter, const Problem& P) {
    int dim = P.M * P.SLOTS;
    const double w = 0.729;
    const double c1 = 1.49445;
    const double c2 = 1.49445;

    vector<Particle> swarm;
    initialize_swarm(swarm, swarm_size, P);

    dbl gbest_val = 1e300;
    vector<dbl> gbest_pos(dim, 0.0);

    // Initial global best from pBests
    for (int i = 0; i < swarm_size; ++i) {
        if (swarm[i].pbest_val < gbest_val) {
            gbest_val = swarm[i].pbest_val;
            gbest_pos = swarm[i].pbest_pos;
        }
    }

    for (int it = 0; it < max_iter; ++it) {

        // Full parallel region per iteration
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            // 1) Evaluate fitness & update pBest in parallel
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < swarm_size; ++i) {
                dbl val = evaluate_fitness(swarm[i].pos, P);
                if (val < swarm[i].pbest_val) {
                    swarm[i].pbest_val = val;
                    swarm[i].pbest_pos = swarm[i].pos;
                }
            }

            // 2) Global best update (single thread)
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                for (int i = 0; i < swarm_size; ++i) {
                    if (swarm[i].pbest_val < gbest_val) {
                        gbest_val = swarm[i].pbest_val;
                        gbest_pos = swarm[i].pbest_pos;
                    }
                }
            }

            // 3) Velocity & position update in parallel
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < swarm_size; ++i) {
                for (int k = 0; k < dim; ++k) {
                    double r1 = rand01();
                    double r2 = rand01();
                    swarm[i].vel[k] = w * swarm[i].vel[k]
                        + c1 * r1 * (swarm[i].pbest_pos[k] - swarm[i].pos[k])
                        + c2 * r2 * (gbest_pos[k] - swarm[i].pos[k]);
                    swarm[i].pos[k] += swarm[i].vel[k];
                }
                repair_particle(swarm[i].pos, P);
            }
        } // end parallel region
    }

    return { gbest_pos, gbest_val };
}

// ----------------- MAIN: Sweep Swarms & Threads, CSV Output -----------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Swarm sizes to test
    vector<int> swarm_sizes = {100, 200, 300};
    // Thread counts for parallel PSO
    vector<int> thread_counts = {1, 2, 4, 8};

    int n_runs = 3;      // average over 3 runs
    int max_iter = 100;  // PSO iterations
    Problem P(10);       // 10 EVs

    cout << "PSO EV - Serial vs Parallel (OpenMP)\n";
    cout << "M=" << P.M << " EVs, slots=" << P.SLOTS
         << ", GRID_MAX=" << P.GRID_MAX << "\n";

#ifdef _OPENMP
    cout << "OpenMP detected. Max available threads: "
         << omp_get_max_threads() << "\n\n";
#else
    cout << "OpenMP NOT enabled. Only serial behavior is effective.\n\n";
#endif

    ofstream file("results_threads.csv");
    file << "SwarmSize,Threads,SerialTime,ParallelTime,Speedup\n";

    for (int swarm : swarm_sizes) {
        cout << "==============================\n";
        cout << "Swarm size: " << swarm << "\n";

        // ----- Serial baseline -----
        double t_serial_sum = 0.0;
        dbl serial_best = 0.0;

        for (int r = 0; r < n_runs; ++r) {
            auto t0 = Clock::now();
            auto res = pso_run_serial(swarm, max_iter, P);
            auto t1 = Clock::now();

            double elapsed = chrono::duration<double>(t1 - t0).count();
            t_serial_sum += elapsed;
            if (r == 0) serial_best = res.second;
        }

        double t_serial = t_serial_sum / n_runs;
        cout << "  Serial best fitness: " << serial_best << "\n";
        cout << "  Serial avg time:     " << t_serial << " s\n\n";

        // ----- Parallel runs for each thread count -----
        for (int th : thread_counts) {

#ifdef _OPENMP
            omp_set_num_threads(th);
#endif

            double t_parallel_sum = 0.0;
            dbl parallel_best = 0.0;

            for (int r = 0; r < n_runs; ++r) {
                auto t0 = Clock::now();
                auto res = pso_run_parallel(swarm, max_iter, P);
                auto t1 = Clock::now();

                double elapsed = chrono::duration<double>(t1 - t0).count();
                t_parallel_sum += elapsed;
                if (r == 0) parallel_best = res.second;
            }

            double t_parallel = t_parallel_sum / n_runs;
            double speedup = t_serial / t_parallel;

            cout << "    Threads: " << th << "\n";
            cout << "      Parallel best fitness: " << parallel_best << "\n";
            cout << "      Parallel avg time:     " << t_parallel << " s\n";
            cout << "      Speedup:               " << speedup << "x\n\n";

            file << swarm << ","
                 << th << ","
                 << t_serial << ","
                 << t_parallel << ","
                 << speedup << "\n";
        }
    }

    file.close();
    cout << "\n[+] Detailed results saved to results_threads.csv\n";
    return 0;
}
