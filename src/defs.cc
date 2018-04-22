#include <boost/math/special_functions/binomial.hpp>
#include <fstream>
#include <sstream>
#include <limits>
#include "defs.h"

const double epsilon = 1e-10;
const int precision = std::numeric_limits<double>::digits10 + 2;

// total number of fermions in an interaction term
const int q = 4;

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

string get_base_filename(string data_dir, string run_name) {
    return data_dir + "/" + run_name;
}

double relative_err(double expected, double actual) {
    return abs((expected - actual) / expected);
}
