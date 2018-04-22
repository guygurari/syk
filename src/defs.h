#ifndef DEFS_H__ 
#define DEFS_H__

#include <complex>
#include <string>

#define SQR(x) ((x)*(x))

extern const double epsilon;
extern const int precision;
extern const int q;

using namespace std;

typedef std::complex<double> cpx;

int binomial(int n, int k);

// djb2 by Dan Bernstein, a nice hash function useful for making random seeds
// http://www.cse.yorku.ca/~oz/hash.html
unsigned long djb2_hash(const char *str);
int get_random_seed(string run_name);
string get_base_filename(string data_run, string run_name);

double relative_err(double expected, double actual);

typedef enum TIME_TYPE {
    REAL_TIME = 1,
    EUCLIDEAN_TIME = 2
} TIME_TYPE;

#endif // DEFS_H__
