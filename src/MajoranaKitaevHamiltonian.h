#ifndef MAJORANA_KITAEV_HAMILTONIAN_H__ 
#define MAJORANA_KITAEV_HAMILTONIAN_H__

#include "defs.h"
#include "eigen_utils.h"
#include "BasisState.h"
#include "MajoranaDisorderParameter.h"

using namespace Eigen;

// Represents a complete Hamiltonian for Majorana fermions.
class MajoranaKitaevHamiltonian {
public:
    // mN (Majorana N) must be even
    MajoranaKitaevHamiltonian(int mN, MajoranaDisorderParameter* J);

    // Returns the dimension of the Hilbert space
    int dim();

    // Diagonalize the Hamiltonian, finding only the eigenvalues.
    void diagonalize(bool print_progress = false);

    // Diagonalize the Hamiltonian, finding eigenvalues and eigenvectors.
    void diagonalize_full(bool print_progress = false);

    bool is_diagonalized();
    bool is_fully_diagonalized();

    // Returns the spectrum, even parity first, odd parity second
    RealVec all_eigenvalues();

    // Returns the spectrum, sorted
    RealVec all_eigenvalues_sorted();

    // Return the Hamiltonian as a dense matrix
    Mat dense_matrix();

    void prepare_charge_parity_blocks(
        SpMat& even_block,
        SpMat& odd_block);

    RealVec diagonalize_block(SpMat& block, bool print_progress);

    // Convert the given matrix to the energy eigenbasis
    Mat to_energy_basis(SpMat& M);
    Mat to_energy_basis(Mat& M);

    // TODO bad API, overlaps with all_eigenvalues.
    // Returns the eigenvalues.. should be called only if fully-diagonalized
    RealVec eigenvalues();

    SpMat matrix;
    const int mN;
    const int dirac_N;

    RealVec even_charge_parity_evs;
    RealVec odd_charge_parity_evs;

    // After full diagonalization,
    // matrix = V D V^{-1}
    // D = diag(evs)
    RealVec evs;
    Mat V;

    // The \chi matrices
    vector<SpMat> chi;
    
private:
    int charge(int global_state_number);

    bool diagonalized;
    bool fully_diagonalized;
};

#endif // MAJORANA_KITAEV_HAMILTONIAN_H__
