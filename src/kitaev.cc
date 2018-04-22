#include <iostream>
#include <iomanip>
#include <fstream>

#include "kitaev.h"
#include "Correlators.h"

using namespace std;

TwoPointOutput::TwoPointOutput() {}
TwoPointOutput::~TwoPointOutput() {}

TwoPointFileOutput::TwoPointFileOutput(string filename) {
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "# T t Re(<c*(t)c(0)>) Im(<c*(t)c(0)>) |<c*(t)c(0)>|^2\n";
}

TwoPointFileOutput::~TwoPointFileOutput() {
    close();
}

void TwoPointFileOutput::write(double T, double t, cpx correlator) {
    file << T
        << "\t" << t 
        << "\t" << real(correlator) 
        << "\t" << imag(correlator) 
        << "\t" << abs(correlator)*abs(correlator)
        << "\n";
}

void TwoPointFileOutput::close() {
    file.close();
}

TwoPointNullOutput::TwoPointNullOutput() {}
void TwoPointNullOutput::write(
        double T, double t, cpx correlator) {}
void TwoPointNullOutput::close() {}

