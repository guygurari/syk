#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "defs.h"
#include "MajoranaDisorderParameter.h"

namespace po = boost::program_options;

typedef struct command_line_options {
    int N;
    double J;
    string seed_file;
} command_line_options;

int parse_command_line_options(
    int argc, char** argv, command_line_options& opts) {

    opts.J = 1.;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
	    ("N", 
	     po::value<int>(&opts.N)->required(),
         "number of fermions")
        ("J",
         po::value<double>(&opts.J),
         "coupling")
        ("seed-file",
         po::value<string>(&opts.seed_file)->required(),
         "File where random seed is saved")
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

        if (opts.N % 2 != 0) {
            cerr << "N must be even (majorana)" << endl;
            exit(1);
        }

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char *argv[]) {
    cout << setprecision(precision);
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    ifstream fin(opts.seed_file.c_str());

    if (!fin.good()) {
        cerr << "Failed to open file " << opts.seed_file << endl;
        return 1;
    }

    int seed;
    fin >> seed;

    if (!fin.good()) {
        cerr << "Failed to read from file " << opts.seed_file << endl;
        return 1;
    }

    // Initialize the random number generator once we have
    // the final seed
    boost::random::mt19937* gen = new boost::random::mt19937(seed);
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, gen);

    double TrH2 = 0.;

    for (int i = 0; i < opts.N; i++) {
        for (int j = i+1; j < opts.N; j++) {
            for (int k = j+1; k < opts.N; k++) {
                for (int l = k+1; l < opts.N; l++) {
                    double Jijkl = Jtensor.elem(i, j, k, l);
                    TrH2 += Jijkl * Jijkl;
                }
            }
        }
    }

    cout << TrH2 << endl;
    return 0;
}
