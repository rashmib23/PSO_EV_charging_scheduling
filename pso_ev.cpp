// pso_ev_clean.cpp
// Cleaned PSO EV toy (serial vs OpenMP parallel)
// Compile: g++ -O2 -fopenmp pso_ev_clean.cpp -o pso_ev_clean

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

// --- Problem definition ---
struct Problem {
    int M = 10;                  // number of EVs
    int SLOTS = 24;              // time slots
    dbl GRID_MAX = 50.0;         // grid max kW
    vector<dbl> price;           // price per slot
    vector<dbl> E_req;           // required energy per EV (kWh)
    vector<dbl> max_power;       // max charging rate per EV per slot (kW)
    dbl alpha = 10000.0;         // overload penalty weight
    dbl beta  = 10000.0;         // unmet energy penalty weight

    Problem(int M_ = 10) : M(M_) {
        SLOTS = 24;
        price.assign(SLOTS, 0.1);
        // sample price curve: cheap at night (0..6), expensive at 17..20
        for (int t = 0; t < SLOTS; ++t) {
            if (t >= 17 && t <= 20) price[t] = 0.5;
            else if (t >= 7 && t <= 16) price[t] = 0.2;
            else price[t] = 0.08;
        }
        E_req.assign(M, 8.0);       // each needs 8 kWh (example)
        max_power.assign(M, 7.0);   // upto 7 kW per hour
    }
};

// --- Random helpers (thread_local RNG) ---
// Safe per-thread RNG for GCC/MinGW + OpenMP
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


// --- Particle structure ---
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

// --- Fitness evaluation ---
dbl evaluate_fitness(const vector<dbl>& pos, const Problem& P) {
    int M = P.M, S = P.SLOTS;
    vector<dbl> total_power(S, 0.0);
    vector<dbl> delivered(M, 0.0);

    for (int j = 0; j < M; ++j) {
        for (int t = 0; t < S; ++t) {
            dbl p = pos[j * S + t];
            total_power[t] += p;
            delivered[j] += p;
        }
    }

    dbl total_cost = 0.0;
    for (int t = 0; t < S; ++t) total_cost += P.price[t] * total_power[t];

    dbl overload = 0.0;
    for (int t = 0; t < S; ++t) {
        if (total_power[t] > P.GRID_MAX) {
            dbl diff = total_power[t] - P.GRID_MAX;
            overload += diff * diff;
        }
    }

    dbl unmet = 0.0;
    for (int j = 0; j < M; ++j) {
        if (delivered[j] < P.E_req[j]) {
            dbl d = P.E_req[j] - delivered[j];
            unmet += d * d;
        }
    }

    return total_cost + P.alpha * overload + P.beta * unmet;
}

// --- Repair particle to satisfy per-EV energy & bounds ---
void repair_particle(vector<dbl>& pos, const Problem& P) {
    int M = P.M, S = P.SLOTS;
    for (int j = 0; j < M; ++j) {
        dbl sum = 0.0;
        for (int t = 0; t < S; ++t) {
            // enforce bounds [0, max_power[j]]
            dbl &x = pos[j * S + t];
            if (x < 0.0) x = 0.0;
            if (x > P.max_power[j]) x = P.max_power[j];
            sum += x;
        }

        if (sum > 1e-12) {
            dbl scale = P.E_req[j] / sum;
            for (int t = 0; t < S; ++t) {
                dbl &x = pos[j * S + t];
                x *= scale;
                if (x > P.max_power[j]) x = P.max_power[j];
            }
        } else {
            // distribute E_req across slots using max_power caps
            dbl remain = P.E_req[j];
            for (int t = 0; t < S && remain > 1e-12; ++t) {
                dbl add = min(P.max_power[j], remain);
                pos[j * S + t] = add;
                remain -= add;
            }
        }
    }
}

// --- Initialize swarm ---
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

// --- Serial PSO ---
pair<vector<dbl>, dbl> pso_run_serial(int swarm_size, int max_iter, const Problem& P) {
    int dim = P.M * P.SLOTS;
    const double w = 0.729;
    const double c1 = 1.49445, c2 = 1.49445;

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

            // update velocity & position
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
    return {gbest_pos, gbest_val};
}

// --- Parallel PSO (OpenMP) ---
pair<vector<dbl>, dbl> pso_run_parallel(int swarm_size, int max_iter, const Problem& P) {
    int dim = P.M * P.SLOTS;
    const double w = 0.729;
    const double c1 = 1.49445, c2 = 1.49445;

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
        // evaluate & update pbest in parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < swarm_size; ++i) {
            dbl val = evaluate_fitness(swarm[i].pos, P);
            if (val < swarm[i].pbest_val) {
                swarm[i].pbest_val = val;
                swarm[i].pbest_pos = swarm[i].pos;
            }
        }

        // reduction step to find global best (single thread)
        #pragma omp single
        {
            // single must be inside a parallel region; wrap below in parallel region
        }

        // We need a full parallel region to use single properly. Do that:
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < swarm_size; ++i) {
                    if (swarm[i].pbest_val < gbest_val) {
                        gbest_val = swarm[i].pbest_val;
                        gbest_pos = swarm[i].pbest_pos;
                    }
                }
            }

            #pragma omp barrier

            // update velocity & position in parallel
            #pragma omp for schedule(static)
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
            } // omp for
        } // omp parallel
    } // iterations

    return {gbest_pos, gbest_val};
}


//  CSV OUTPUT FUNCTION

void save_results_csv(const vector<int>& swarm_sizes,
                      const vector<double>& serial_times,
                      const vector<double>& parallel_times)
{
    ofstream file("results.csv");
    file << "SwarmSize,SerialTime,ParallelTime,Speedup\n";

    for (int i = 0; i < swarm_sizes.size(); i++) {
        double speedup = serial_times[i] / parallel_times[i];
        file << swarm_sizes[i] << ","
             << serial_times[i] << ","
             << parallel_times[i] << ","
             << speedup << "\n";
    }

    file.close();
    cout << "\n[+] Results saved to results.csv\n";
}


//  MAIN PROGRAM

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int swarm_sizes[3] = {100, 1000, 5000};
    int n_runs = 3;          // average over 3 runs
    int max_iter = 100;      // PSO iterations
    Problem P(10);           // 10 EVs (adjustable)

    vector<double> serial_times;
    vector<double> parallel_times;

    cout << "PSO EV Toy - Serial vs Parallel (OpenMP)\n";
    cout << "M=" << P.M << " EVs, slots=" << P.SLOTS 
         << ", GRID_MAX=" << P.GRID_MAX << "\n";

    // Thread info
    #ifdef _OPENMP
    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
    #else
    cout << "OpenMP NOT enabled. Running only serial.\n\n";
    #endif

    
    //  LOOP OVER SWARM SIZES
    
    for (int s = 0; s < 3; ++s) {
        int swarm = swarm_sizes[s];
        cout << "=== Swarm size: " << swarm << " ===\n";

      
        // Serial runs
        
        double t_serial = 0.0;
        dbl serial_best = 0.0;

        for (int r = 0; r < n_runs; r++) {
            auto t0 = Clock::now();
            auto res = pso_run_serial(swarm, max_iter, P);
            auto t1 = Clock::now();

            double elapsed = chrono::duration<double>(t1 - t0).count();
            t_serial += elapsed;

            if (r == 0)
                serial_best = res.second;
        }
        t_serial /= n_runs;   // average
        serial_times.push_back(t_serial);

        cout << "Serial best fitness: " << serial_best << "\n";
        cout << "Serial avg time:     " << t_serial << " s\n";

        
        // Parallel runs
       
        double t_parallel = 0.0;
        dbl parallel_best = 0.0;

        for (int r = 0; r < n_runs; r++) {
            auto t0 = Clock::now();
            auto res = pso_run_parallel(swarm, max_iter, P);
            auto t1 = Clock::now();

            double elapsed = chrono::duration<double>(t1 - t0).count();
            t_parallel += elapsed;

            if (r == 0)
                parallel_best = res.second;
        }
        t_parallel /= n_runs;
        parallel_times.push_back(t_parallel);

        cout << "Parallel best fitness: " << parallel_best << "\n";
        cout << "Parallel avg time:     " << t_parallel << " s\n";

        // Speedup
        cout << "Speedup (serial/parallel): " 
             << (t_serial / t_parallel) << "\n\n";
    }

    
    //  SAVE RESULTS TO CSV
   
    vector<int> swarm_vec = {100, 1000, 5000};
    save_results_csv(swarm_vec, serial_times, parallel_times);

    return 0;
}