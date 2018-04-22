#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>

#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/binomial.hpp>

using namespace std;

namespace po = boost::program_options;

typedef unsigned int uint;
typedef unsigned char uchar;

const float epsilon = 1e-6;
const int precision = std::numeric_limits<double>::digits10 + 2;

boost::random::mt19937* gen = 0;

typedef struct command_line_options {
    string run_name;
    string data_dir;
    int N;
    double J;
    int seed;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\nN = " << N;
        ss << "\nJ = " << J;
        ss << "\nseed = " << seed;

        ss << "\n";
        return ss.str();
    }
} command_line_options;

// djb2 by Dan Bernstein
// http://www.cse.yorku.ca/~oz/hash.html
unsigned long djb2_hash(const char *str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str) != 0) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        ++str;
    }

    return hash;
}

// Random seed is a hash of the following string:
// run-name + "!!!" + time-in-seconds + "???" + random-number-from-urandom
int get_random_seed(string run_name) {
    ifstream rand_file("/dev/urandom");
    int dev_urandom_bytes;
	rand_file.read((char*) &dev_urandom_bytes, sizeof(int));
    //seed += dev_urandom_bytes;
    rand_file.close();

    stringstream s;
    s << run_name << "!!!" << time(0) << "???" << dev_urandom_bytes;
    int seed = djb2_hash(s.str().c_str());

    /*int seed = time(0);

    for (unsigned int i = 0; i < run_name.length(); i++) {
        seed += (int) run_name[i];
    }*/

    return seed;
}

int parse_command_line_options(int argc, char** argv, command_line_options& opts) {
    opts.J = 1.;
    opts.data_dir = "data";

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("run-name", 
         po::value<string>(&opts.run_name)->required(),
         "set run's name (required for fixing a unique seed for each job)")
	    ("data-dir", 
	     po::value<string>(&opts.data_dir),
         "where to save data")
	    ("N", 
	     po::value<int>(&opts.N)->required(),
         "number of fermions")
        ("J",
         po::value<double>(&opts.J),
         "coupling")
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

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

int binomial(int n, int k) {
    if (n < 0 || k < 0) {
        return 0;
    }
    if (n < k) {
        return 0;
    }
    else {
        return (int) boost::math::binomial_coefficient<double>(n, k);
    }
}

// void initCouplings(boost::random::mt19937* gen) {
//     float sigma = sqrt(6 * pow(J,2) / pow(N,3));
//     boost::random::normal_distribution<> dist(0, sigma);



//     // Choose random J elements
//     for (int i = 0; i < N; i++) {
//         for (int j = i+1; j < N; j++) {
//             for (int k = j+1; k < N; k++) {
//                 for (int l = k+1; l < N; l++) {
//                     Jelems[i][j][k][l] = dist(*gen);
//                     Jelems[i][j][l][k] = - Jelems[i][j][k][l];
//                     Jelems[i][k][j][l] = - Jelems[i][j][k][l];
//                     Jelems[i][k][l][j] = + Jelems[i][j][k][l];
//                     Jelems[i][l][j][k] = + Jelems[i][j][k][l];
//                     Jelems[i][l][k][j] = - Jelems[i][j][k][l];
//                     Jelems[j][i][k][l] = - Jelems[i][j][k][l];
//                     Jelems[j][i][l][k] = + Jelems[i][j][k][l];
//                     Jelems[j][k][i][l] = + Jelems[i][j][k][l];
//                     Jelems[j][k][l][i] = - Jelems[i][j][k][l];
//                     Jelems[j][l][i][k] = - Jelems[i][j][k][l];
//                     Jelems[j][l][k][i] = + Jelems[i][j][k][l];
//                     Jelems[k][i][j][l] = + Jelems[i][j][k][l];
//                     Jelems[k][i][l][j] = - Jelems[i][j][k][l];
//                     Jelems[k][j][i][l] = - Jelems[i][j][k][l];
//                     Jelems[k][j][l][i] = + Jelems[i][j][k][l];
//                     Jelems[k][l][i][j] = + Jelems[i][j][k][l];
//                     Jelems[k][l][j][i] = - Jelems[i][j][k][l];
//                     Jelems[l][i][j][k] = - Jelems[i][j][k][l];
//                     Jelems[l][i][k][j] = + Jelems[i][j][k][l];
//                     Jelems[l][j][i][k] = + Jelems[i][j][k][l];
//                     Jelems[l][j][k][i] = - Jelems[i][j][k][l];
//                     Jelems[l][k][i][j] = - Jelems[i][j][k][l];
//                     Jelems[l][k][j][i] = + Jelems[i][j][k][l];
//                 }
//             }
//         }
//     }
// }

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cout << setprecision(precision);
    cout << "Command line options:\n" << opts.to_s() << endl;
    gen = new boost::random::mt19937(opts.seed);

    //compute_majorana_spectrum(opts);
    //lanczos(opts);
    return 0;
}
