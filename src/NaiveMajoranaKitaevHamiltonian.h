#ifndef NAIVE_MAJORANA_KITAEV_HAMILTONIAN_H__ 
#define NAIVE_MAJORANA_KITAEV_HAMILTONIAN_H__

#include "defs.h"
#include "eigen_utils.h"
#include "BasisState.h"
#include "MajoranaDisorderParameter.h"

// Represents a complete Hamiltonian for Majorana fermions using a naive
// and slow implementation.
class NaiveMajoranaKitaevHamiltonian {
public:
    // N must be even
    NaiveMajoranaKitaevHamiltonian(int N, MajoranaDisorderParameter* J);

    // Returns the dimension of the Hilbert space
    int dim();

    // Diagonalize the Hamiltonian, finding only the eigenvalues.
    void diagonalize();

    bool is_diagonalized();

    // Returns the spectrum
    RealVec all_eigenvalues();

    Mat matrix;
    const int N;
    const int dirac_N;
    
private:
    void add_majorana_term_contributions(
            int ket_n,
            int a, int b, int c, int d, 
            MajoranaDisorderParameter* J);
    cpx act_with_dirac_operator(
            BasisState& state, int majorana_index, int c_or_cstar);

    bool diagonalized;
    RealVec evs;
};

#endif // NAIVE_MAJORANA_KITAEV_HAMILTONIAN_H__

