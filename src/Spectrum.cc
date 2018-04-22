#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "defs.h"
#include "Spectrum.h"
#include "TSVFile.h"

using namespace std;

Spectrum::Spectrum(KitaevHamiltonian& H) {
    assert(H.is_diagonalized());
    N = H.N;

    for (int Q = 0; Q <= N; Q++) {
        eigenvalues_by_Q.push_back(H.blocks[Q]->eigenvalues());
    }
}

Spectrum::Spectrum(string filename) {
    N = find_N_of_spectrum_file(filename);
    TSVFile file(filename);
    vector<int> idx_by_Q(N+1);

    for (int Q = 0; Q <= N; Q++) {
        int size = binomial(N,Q);
        RealVec evs_Q = RealVec::Zero(size);
        eigenvalues_by_Q.push_back(evs_Q);
        idx_by_Q[Q] = 0;
    }

    while (true) {
        int Q;
        double ev;
        file >> Q >> ev;

        if (file.eof()) {
            break;
        }

        RealVec& evs = eigenvalues_by_Q[Q];
        evs(idx_by_Q[Q]++) = ev;
    }

    file.close();
}

int Spectrum::dim() {
    return pow(2,N);
}

RealVec& Spectrum::eigenvalues(int Q) {
    return eigenvalues_by_Q[Q];
}

RealVec Spectrum::all_eigenvalues() {
    RealVec evs = RealVec(dim());
    int k = 0;

    for (int Q = 0; Q <= N; Q++) {
        int block_dim = eigenvalues_by_Q[Q].size();
        evs.block(k, 0, block_dim, 1) = eigenvalues_by_Q[Q];
        k += block_dim;
    }

    return evs;
}

void Spectrum::save(string filename) {
    ofstream file;
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "# Q eigenvalue\n";

    for (int Q = 0; Q <= N; Q++) {
        RealVec evs = eigenvalues_by_Q[Q];

        for (int i = 0; i < evs.size(); i++) {
            file << Q << "\t" << evs[i] << "\n";
        }
    }

    file.close();
}

int Spectrum::find_N_of_spectrum_file(string filename) {
    TSVFile file(filename);

    int Q;
    double ev;

    while (true) {
        file >> Q >> ev;

        if (file.eof()) {
            break;
        }
    }

    file.close();

    // Last Q is equal to N
    return Q;
}

MajoranaSpectrum::MajoranaSpectrum(string filename) {
    TSVFile file(filename);
    vector<double> vec;

    while (true) {
        int charge_parity;
        double ev;
        file >> charge_parity >> ev;

        if (file.eof()) {
            break;
        }

        vec.push_back(ev);
    }

    evs.resize(vec.size());

    for (unsigned int i = 0; i < vec.size(); i++) {
        evs(i) = vec[i];
    }

    int dirac_N = (int) log2(evs.size());
    majorana_N = 2 * dirac_N;

    file.close();
}

RealVec MajoranaSpectrum::all_eigenvalues() {
    return evs;
}
