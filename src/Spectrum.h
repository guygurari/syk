#ifndef SPECTRUM_H__ 
#define SPECTRUM_H__

#include <vector>
#include <string>
#include "eigen_utils.h"
#include "KitaevHamiltonian.h"

using namespace std;

/*
 * The spectrum of a Hamiltonian.
 */
class Spectrum {
public:
    // Construct the spectrum from the given diagonalized Hamiltonian
    Spectrum(KitaevHamiltonian& H);

    // Load the spectrum from the given file
    Spectrum(string filename);

    // Hilbert space dimension
    int dim();

    // All eigenvalues
    RealVec all_eigenvalues();

    // Eigenvalues for given Q sector
    RealVec& eigenvalues(int Q);

    // Save the spectrum to the given file
    void save(string filename);

    int find_N_of_spectrum_file(string filename);

    int N;
    vector<RealVec> eigenvalues_by_Q;
};

/*
 * The spectrum of a Majorana Hamiltonian.
 */
class MajoranaSpectrum {
public:
    // Load the spectrum from the given file
    MajoranaSpectrum(string filename);

    // All eigenvalues
    RealVec all_eigenvalues();

    int majorana_N;
    RealVec evs;
};

#endif // SPECTRUM_H__
