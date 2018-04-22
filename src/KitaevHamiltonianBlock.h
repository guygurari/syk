#ifndef KITAEV_HAMILTONIAN_BLOCK__ 
#define KITAEV_HAMILTONIAN_BLOCK__

#include "defs.h"
#include "eigen_utils.h"
#include "BasisState.h"
#include "DisorderParameter.h"

/*
 * A block of the Kitaev Hamiltonian with a given U(1) charge Q and a given
 * choice of disorder parameter.
 * The full Hamiltonian is given by:
 *
 * H = (1/2N)^3/2 \sum_{i!=j, k!=l} J_{ijkl} c^\dag_i c^\dag_j c_k c_l
 */
class KitaevHamiltonianBlock {
public:
    KitaevHamiltonianBlock(int N, int Q, DisorderParameter* J);

    // Construct from a given matrix
    KitaevHamiltonianBlock(int N, int Q, Mat block);

    virtual ~KitaevHamiltonianBlock();

    BasisState act_with_c_operators(
        int i, int j, int k, int l, BasisState ket);

    // Returns the dimension of this block
    int dim();

    // Returns matrix elements (n,m) of this Hamiltonian
    cpx operator () (int n, int m);

    // Diagonalize the Hamiltonian.
    //
    // If full_diagonalization = true, set U,D such that:
    // H = U.D.U^\dagger
    //
    // Otherwise set just D to be the eigenvalues.
    void diagonalize(bool full_diagonalization = true);

    // To have a consistent interface with KitaevHamiltonian
    bool is_diagonalized();

    RealVec eigenvalues();

    // The diagonal D matrix (after diagonalization)
    Mat D_matrix();

    // The unitary U matrix (after diagonalization)
    Mat U_matrix();

    const int N;
    const int Q;
    Mat matrix;

    bool diagonalized;
    Mat U;
    RealVec evs;

protected:
    // Initialize a zero black
    KitaevHamiltonianBlock(int N, int Q);

    void initialize_block_matrix(DisorderParameter* J);
    void add_hamiltonian_term_contribution(
        DisorderParameter* J, int i, int j, int k, int l);
    void add_term_and_state_contribution(
        DisorderParameter* J, int i, int j, int k, int l,
        list<int>& indices);
    void insert_index_and_shift(list<int>& indices, int i);
    void shift_starting_at_index(list<int>& indices, int i);
};

/*
 * Naive implementation of Kitaev Hamiltonian block.
 */
class NaiveKitaevHamiltonianBlock : public KitaevHamiltonianBlock {
public:
    NaiveKitaevHamiltonianBlock(int N, int Q, DisorderParameter* J);
    virtual ~NaiveKitaevHamiltonianBlock();

protected:
    void initialize_block_matrix_naive_implementation(DisorderParameter* J);
};

#endif // KITAEV_HAMILTONIAN_BLOCK__
