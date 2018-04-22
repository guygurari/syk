////////////////////////////////////////////////////////////
//
// cuBLAS implementation of SYK Lanczos for findind the low
// part of the SYK spectrum.
// 
////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <unistd.h>

// Disable annoying warning on Sherlock
// #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
// #pragma GCC diagnostic push
// #endif
// #pragma GCC diagnostic ignored "-Wuninitialized"

#include <boost/program_options.hpp>

// #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
// #pragma GCC diagnostic pop
// #endif

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <cuda_profiler_api.h>

#include "defs.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "Timer.h"
#include "CudaUtils.h"
#include "CudaHamiltonian.h"
#include "CudaMultiGpuHamiltonian.h"
#include "CudaState.h"
#include "CudaLanczos.h"

namespace po = boost::program_options;

typedef unsigned int uint;

typedef struct command_line_options {
    string run_name;
    string data_dir;
    string checkpoint_dir;
    int N;
    double J;
    double mu;

    // int num_evs;
    // double ev_error_tolerance;

    int num_steps;
    int checkpoint_steps;
    int ev_steps;

    bool resume;
    bool debug;
    bool profile;
    int seed;
    // bool mock_hamiltonian;
    boost::random::mt19937* gen;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\ncheckpoint_dir = " << checkpoint_dir;
        ss << "\nN = " << N;
        ss << "\nJ = " << J;
        // ss << "\nnum_evs = " << num_evs;
        // ss << "\nerror_ntolerance = " << ev_error_tolerance;
        ss << "\nnum_steps = " << num_steps;
        ss << "\ncheckpoint_steps = " << checkpoint_steps;
        ss << "\nev_steps = " << ev_steps;
        ss << "\nmu = " << mu;
        ss << "\nseed = " << seed;
        ss << "\nresume = " << resume;
        ss << "\ndebug = " << debug;
        ss << "\nprofile = " << profile;

        ss << "\n";
        return ss.str();
    }
} command_line_options;

int parse_command_line_options(int argc, char** argv, command_line_options& opts) {
    opts.gen = 0;
    opts.N = 8;
    opts.J = 1.;
    opts.mu = 0.;

    // opts.num_evs = -1;
    // opts.ev_error_tolerance = epsilon;

    opts.num_steps = 0;
    opts.checkpoint_steps = 0;
    opts.ev_steps = 0;

    opts.data_dir = "data/lanczos";
    opts.checkpoint_dir = "data/lanczos-checkpoints";
    opts.resume = false;
    opts.debug = false;
    opts.profile = false;
    // opts.mock_hamiltonian = false;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("run-name", 
         po::value<string>(&opts.run_name)->required(),
         "set run's name (required for fixing a unique seed for each job)")
	    ("data-dir", 
	     po::value<string>(&opts.data_dir),
         "where to save data")
	    ("checkpoint-dir", 
	     po::value<string>(&opts.checkpoint_dir),
         "where to save checkpoints")
	    ("N", 
	     po::value<int>(&opts.N)->required(),
         "number of fermions")
        ("J",
         po::value<double>(&opts.J),
         "coupling")
	    // ("num-evs", 
	    //  po::value<int>(&opts.num_evs)->required(),
        //  "number of eigenvalues at low and high energies with given error tolerance. if not specified, tries to find all of them.")
	    ("num-steps", 
	     po::value<int>(&opts.num_steps)->required(),
         "number of Lanczos steps")
	    ("checkpoint-steps", 
	     po::value<int>(&opts.checkpoint_steps),
         "number of steps between saving the state (default: num-steps)")
	    ("ev-steps", 
	     po::value<int>(&opts.ev_steps),
         "number of steps between computing eigenvalues (must be a multiple of checkpoint-steps; default: checkpoint-steps)")
	    // ("error-tolerance", 
	    //  po::value<double>(&opts.ev_error_tolerance),
        //  "allowed tolerance when deciding which eigenvalues to accept")
        ("mu",
         po::value<double>(&opts.mu),
         "Hamiltonian shift (default = 0)")
        // ("mock", "use mock hamiltonian")
	    ("resume", "try to resume from saved state")
	    ("debug", "show debugging output")
	    ("profile", "turn on CUDA profiling")
	    ("seed",
         po::value<int>(&opts.seed),
	     "random seed")
	    ("help", "produce help message");

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (!vm.count("seed")) {
            opts.seed = get_random_seed(opts.run_name);
        }

        if (opts.N % 2 != 0) {
            cerr << "N must be even (majorana)" << endl;
            exit(1);
        }

        if (vm.count("resume")) {
            opts.resume = true;
        }

        if (vm.count("debug")) {
            opts.debug = true;
        }

        if (!vm.count("checkpoint-steps")) {
            opts.checkpoint_steps = opts.num_steps;
        }

        if (!vm.count("ev-steps")) {
            opts.ev_steps = opts.checkpoint_steps;
        }

        if (opts.ev_steps % opts.checkpoint_steps != 0) {
            cerr << "ev-steps must be a multiple of checkpoint-steps" << endl;
            exit(1);
        }

        if (opts.num_steps % opts.ev_steps != 0) {
            cerr << "num-steps must be multiple of ev-steps" << endl;
            exit(1);
        }

        // if (vm.count("mock")) {
        //     opts.mock_hamiltonian = true;
        // }

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static bool file_exists(string filename) {
    return access(filename.c_str(), F_OK) == 0;
}

static string get_seed_filename(command_line_options& opts) {
    stringstream filename;
    filename << get_base_filename(opts.checkpoint_dir, opts.run_name)
             << "-seed";
    return filename.str();
}

static void save_seed(command_line_options& opts) {
    string filename = get_seed_filename(opts);
    ofstream out(filename.c_str());
    out << opts.seed << "\n";
    out.close();
}

// Try to load the seed from a saved file. Return true iff successful.
static bool load_seed(command_line_options& opts) {
    string filename = get_seed_filename(opts);

    if (file_exists(filename)) {
        ifstream in(filename.c_str());
        in >> opts.seed;
        in.close();
        return true;
    }
    else {
        return false;
    }
}

static string get_lanczos_state_filename(command_line_options& opts) {
    stringstream filename;
    filename << get_base_filename(opts.checkpoint_dir, opts.run_name)
             << "-state";
    return filename.str();
}

static void save_lanczos_results(
    command_line_options& opts,
    const RealVec& evs, const RealVec& errs) {

    string filename =
        get_base_filename(opts.data_dir, opts.run_name) + ".tsv";
    // cout << "Writing results to " << filename << endl;

    ofstream out(filename.c_str());
    out << setprecision(precision);
    out << "# i eigenvalue error-estimate\n";

    for (int i = 0; i < evs.size(); i++) {
        out << i << "\t" << evs(i) << "\t" << errs(i) << "\n";
    }

    out.close();
}

static void count_evs(
    command_line_options& opts,
    FactorizedSpace& space,
    const RealVec& errs) {

    // opts.num_evs < 0 means we are asked to find all the eigenvalues,
    // but just of the even charge sector.
    // int num_evs = opts.num_evs < 0 ? space.D/2 : opts.num_evs;

    // Check if there are enough evs. Low energy and high energy might overlap,
    // we don't care. This allows the all-evs case to work.
    int num_low_found = 0;
    int num_high_found = 0;
    
    for (int i = 0; i < errs.size(); i++) {
        // if (errs(i) > opts.ev_error_tolerance) break;
        if (errs(i) > epsilon) break;
        num_low_found++;
    }

    for (int i = 0; i < errs.size(); i++) {
        // if (errs(errs.size() - 1 - i) > opts.ev_error_tolerance) break;
        if (errs(errs.size() - 1 - i) > epsilon) break;
        num_high_found++;
    }

    cout << "Found " << num_low_found << " good evs at low energy" << endl;
    cout << "Found " << num_high_found << " good evs at high energy" << endl;

    // return (num_low_found >= num_evs) && (num_high_found >= num_evs);
}

void run_lanczos(command_line_options& opts,
                 cublasHandle_t handle,
                 cusparseHandle_t handle_sp) {
    // Jtensor should be the first thing that uses the random number
    // generator, so that we can resume from a known seed.
    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, opts.gen);

    CudaLanczos* lanczos = 0;

    if (opts.resume) {
        lanczos = new CudaLanczos(space,
                                  get_lanczos_state_filename(opts),
                                  opts.num_steps);
        assert(lanczos->max_steps() == opts.num_steps);
        cout << "Resuming from state " << get_lanczos_state_filename(opts)
             << " at step " << lanczos->current_step() << endl;
    }
    else {
        lanczos = new CudaLanczos(
            opts.num_steps,
            FactorizedParityState(space, EVEN_CHARGE, opts.gen));
    }

    if (lanczos->current_step() >= opts.num_steps) {
        cout << "Lanczos already done, no need to compute; quitting." << endl;
        delete lanczos;
        return;
    }

    const size_t mem_overhead = 500000000; // 0.5 GB
    CudaHamiltonianInterface* H;
    Timer timer;

    if (cuda_get_num_devices() == 1) {
        size_t available_memory = cuda_get_device_memory();
        assert(available_memory > 10 * mem_overhead);
        
        size_t hamiltonian_memory =
            available_memory - lanczos->d_alloc_size - mem_overhead;

        double hamiltonian_memory_gb = (double) hamiltonian_memory / pow(2., 30);
        printf("Hamiltonian has %.2f GB available on device\n",
               hamiltonian_memory_gb);

        H = new CudaHamiltonian(space, Jtensor, hamiltonian_memory,
                                false, opts.debug);
        timer.print();
    }
    else {
        assert(cuda_get_num_devices() > 1);
        cout << "Multi-GPU version on " << cuda_get_num_devices() << " devices"
             << endl;
        vector<size_t> mem_per_device;

        // First device should account for lanczos working mem, rest of devices
        // don't need to.
        for (int i = 0; i < cuda_get_num_devices(); i++) {
            size_t hamiltonian_memory = cuda_get_device_memory(i) - mem_overhead;

            if (i == 0) {
                cout << "Subtracting Lanczos overhead: " << lanczos->d_alloc_size
                     << endl;
                hamiltonian_memory -= lanczos->d_alloc_size;
            }

            double hamiltonian_memory_gb = (double) hamiltonian_memory / pow(2., 30);
            printf("Hamiltonian %d has %.2f GB available on device\n",
                   i, hamiltonian_memory_gb);

            mem_per_device.push_back(hamiltonian_memory);
        }
        
        cout << "Creating Hamiltonian..." << endl;

        CudaMultiGpuHamiltonian* multi_H = new CudaMultiGpuHamiltonian(
            space, Jtensor, mem_per_device, false, opts.debug);

        H = multi_H;
        timer.print();
        multi_H->print_all_chunks();
    }

    H->print_memory_allocation();

    while (lanczos->current_step() < opts.num_steps) {
        cout << "\nRunning Lanczos for " << opts.checkpoint_steps
             << " steps" << endl;
        timer.reset();

        if (opts.profile) {
            cudaProfilerStart();
        }
        
        lanczos->compute(handle, handle_sp,
                         *H, opts.mu, opts.checkpoint_steps);

        if (opts.profile) {
            cudaProfilerStop();
        }

        timer.print();

        cout << "Each Lanczos step takes "
             << timer.seconds() / opts.checkpoint_steps
             << " secs" << endl;

        cout << "\n======== Checkpoint at step="
             << lanczos->current_step() << " =============\n";

        // cout << "Saving state to " << get_lanczos_state_filename(opts) << endl;
        lanczos->save_state(get_lanczos_state_filename(opts));

        if (lanczos->current_step() % opts.ev_steps == 0) {
            cout << "Computing eigenvalues\n";
            timer.reset();
            RealVec errs;
            RealVec evs = lanczos->compute_eigenvalues(errs, opts.gen);
            timer.print();

            assert(evs.size() == errs.size());
            save_lanczos_results(opts, evs, errs);
            count_evs(opts, space, errs);
        }
    }

    cout << "Reached " << lanczos->current_step()
         << " steps, quitting." << endl;

    delete lanczos;
    delete H;
}

int main(int argc, char *argv[]) {
    cout << setprecision(precision);
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cuda_print_device_properties();

    cublasHandle_t handle = cublas_init(argc, argv);
    cusparseHandle_t handle_sp = cusparse_init();

    // Try to resume from saved state and seed
    if (opts.resume && load_seed(opts)) {
        if (file_exists(get_lanczos_state_filename(opts))) {
            cout << "Resuming from saved state." << endl;
            opts.resume = true;
        }
        else {
            cout << "Found seed file but not state file, "
                 << "using the seed but not resuming the Lanczos state.\n";
            opts.resume = false;
        }
    }
    else {
        // If we cannot find the seed file we will not resume
        opts.resume = false;
    }

    // Initialize the random number generator once we have
    // the final seed
    opts.gen = new boost::random::mt19937(opts.seed);
    save_seed(opts);

    cout << "Command line options:\n" << opts.to_s() << endl;

    run_lanczos(opts, handle, handle_sp);

    cublas_destroy(handle);
    cusparse_destroy(handle_sp);

    return 0;
}
