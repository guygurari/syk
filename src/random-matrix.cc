#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "defs.h"
#include "RandomMatrix.h"
#include "Timer.h"

namespace po = boost::program_options;

boost::random::mt19937* gen = 0;

typedef enum {
    GUE = 1,
    GOE = 2,
    SparseKlogK = 3
} Ensemble;

string ensemble_name(Ensemble e) {
    if (e == GUE) {
        return "GUE";
    }
    else if (e == GOE) {
        return "GOE";
    }
    else if (e == SparseKlogK) {
        return "SparseKlogK";
    }

    assert(false);
}

typedef struct command_line_options {
    string run_name;
    string data_dir;
    int K;
    double sigma;
    bool sparse_KlogK;
    Ensemble ensemble;
    int seed;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\nK = " << K;
        ss << "\nenesmble = " << ensemble_name(ensemble);
        ss << "\nseed = " << seed;

        ss << "\n";
        return ss.str();
    }
} command_line_options;

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    opts.data_dir = "data";

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            // Basic parameters
            ("run-name", 
             po::value<string>(&opts.run_name)->required(),
             "set run's name (required for fixing a unique seed for each job)")
            ("data-dir", 
             po::value<string>(&opts.data_dir),
             "where to save data")
            ("K", 
             po::value<int>(&opts.K)->required(),
             "The rank of the matrix")
            ("GUE",
             "GUE ensemble")
            ("GOE",
             "GOE ensemble")
            ("SparseKlogK",
             "sparse matrix ensemble with K*log(K) non-zero elements")
            ("seed",
             po::value<int>(&opts.seed),
             "random seed")
            ("help", "produce help message")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("GUE")) {
            opts.ensemble = GUE;
        }
        else if (vm.count("GOE")) {
            opts.ensemble = GOE;
        }
        else if (vm.count("SparseKlogK")) {
            opts.ensemble = SparseKlogK;
        }
        else {
            cerr << "Must specify an ensemble" << endl;
            exit(1);
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (!vm.count("seed")) {
            opts.seed = get_random_seed(opts.run_name);
        }

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cout << setprecision(precision);
    cout << "Command line options:\n" << opts.to_s() << endl;
    gen = new boost::random::mt19937(opts.seed);

    RandomMatrix* matrix = 0;

    cout << "Constructing Hamiltonian..." << endl;
    Timer timer;

    if (opts.ensemble == GUE) {
        matrix = new GUERandomMatrix(opts.K, gen);
    }
    else if (opts.ensemble == GOE) {
        matrix = new GOERandomMatrix(opts.K, gen);
    }
    else if (opts.ensemble == SparseKlogK) {
        int num_nonzeros = (int) (opts.K * log(opts.K));
        matrix = new SparseHermitianRandomMatrix(
                opts.K, num_nonzeros, gen);
    }
    else {
        cerr << "Unknown ensemble " << opts.ensemble << endl;
        exit(1);
    }

    timer.print();

    cout << "Diagonalizing Hamiltonian..." << endl;
    timer.reset();
    RealVec evs = matrix->eigenvalues();
    timer.print();

    string filename = 
        get_base_filename(opts.data_dir, opts.run_name)
        + "-spectrum.tsv";
    cout << "Saving spectrum to " << filename << endl;
    ofstream file;
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "# eigenvalue\n";

    for (int i = 0; i < evs.size(); i++) {
        file << evs[i] << "\n";
    }

    file.close();

    delete matrix;
    return 0;
}



