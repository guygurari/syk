#ifndef KITAEV_HAMILTONIAN_H__ 
#define KITAEV_HAMILTONIAN_H__

#include <Eigen/Core>
#include <Eigen/Dense>
#include "defs.h"
#include "DisorderParameter.h"
#include "KitaevHamiltonianBlock.h"

using namespace Eigen;

// Represents a complete Hamiltonian as a list of (N+1) blocks sorted by Q,
// Q = 0,...,N.
class KitaevHamiltonian {
public:
    KitaevHamiltonian(int N, DisorderParameter* J);
    ~KitaevHamiltonian();

    // Returns the dimension of the Hilbert space
    int dim();

    // Diagonalize the Hamiltonian by blocks.
    //
    // Flag indicates whether to do full diagonalization, namely
    // also compute the eigenvectors.
    void diagonalize(
            bool full_diagonalization = true,
            bool print_progress = false);

    bool is_diagonalized();

    // Returns the spectrum, ordered by Q
    RealVec eigenvalues();

    // Return the full matrix form of the Hamiltonian. 
    // Blocks are sorted by Q.
    Mat as_matrix();

    // Blocks ordered by Q
    vector<KitaevHamiltonianBlock*> blocks;

    const int N;
};

#endif // KITAEV_HAMILTONIAN_H__
