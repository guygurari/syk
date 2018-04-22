#include "defs.h"
#include "BasisState.h"
#include "KitaevHamiltonianBlock.h"
#include <Eigen/Eigenvalues> 
#include <iostream>

using namespace std;

KitaevHamiltonianBlock::~KitaevHamiltonianBlock() {}
NaiveKitaevHamiltonianBlock::~NaiveKitaevHamiltonianBlock() {}

KitaevHamiltonianBlock::KitaevHamiltonianBlock(
        int _N, int _Q, DisorderParameter* J) 
    : N(_N), Q(_Q), diagonalized(false) {
    initialize_block_matrix(J);
}

KitaevHamiltonianBlock::KitaevHamiltonianBlock(int _N, int _Q) 
    : N(_N), Q(_Q), diagonalized(false) {
    matrix = Mat::Zero(dim(), dim());
}

KitaevHamiltonianBlock::KitaevHamiltonianBlock(int _N, int _Q, Mat block)
    : N(_N), Q(_Q), diagonalized(false) {
    matrix = block;
}

NaiveKitaevHamiltonianBlock::NaiveKitaevHamiltonianBlock(
        int _N, int _Q, DisorderParameter* J) 
    : KitaevHamiltonianBlock(_N,_Q) {
    initialize_block_matrix_naive_implementation(J);
}

// Initialize the matrix. Only go over states that don't trivially vanish.
void KitaevHamiltonianBlock::initialize_block_matrix(DisorderParameter* J) {
    matrix = Mat::Zero(dim(), dim());

    // Overall Hamiltonian coefficient.
    // Factor of 4 is because we are only summing over i<j and k<l
    double coefficient = 1./pow(2. * N, 3./2.) * 4.;

    // Loop over disorder elements
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    add_hamiltonian_term_contribution(J, i, j, k, l);
                }
            }
        }
    }

    matrix *= coefficient;
}

void KitaevHamiltonianBlock::add_term_and_state_contribution(
        DisorderParameter* J, 
        int i, int j, int k, int l,
        list<int>& indices) {
    BasisState ket(indices);
    BasisState bra = act_with_c_operators(i, j, k, l, ket);

    if (bra.is_zero()) {
        return;
    }

    if (bra.charge() != Q) {
        cout << i << "," << j << "," << k << "," << l << endl;
        cout << "ket.charge=" << ket.charge() << " ";
        cout << "bra.charge=" << bra.charge() << " Q=" << Q << endl;
    }
    assert(bra.charge() == Q);
    int ket_n = ket.get_state_number();
    int bra_n = bra.get_state_number();

    assert(bra_n >= 0);
    assert(bra_n < dim());
    matrix(bra_n, ket_n) += J->elem(i,j,k,l) * cpx(bra.coefficient, 0);
}

void KitaevHamiltonianBlock::insert_index_and_shift(
        list<int>& indices, int i) {
    list<int>::iterator iter;
    bool inserted = false;

    for (iter = indices.begin(); iter != indices.end(); ++iter) {
        if (*iter >= i) {
            if (!inserted) {
                indices.insert(iter, i);
                inserted = true;
            }
            (*iter)++;
        }
    }

    if (!inserted) {
        // Reached the end without inserting, so insert it here
        indices.insert(iter, i);
    }
}

void KitaevHamiltonianBlock::shift_starting_at_index(
        list<int>& indices, int i) {
    list<int>::iterator iter;

    for (iter = indices.begin(); iter != indices.end(); ++iter) {
        if (*iter >= i) {
            (*iter)++;
        }
    }
}

void KitaevHamiltonianBlock::add_hamiltonian_term_contribution(
        DisorderParameter* J, int i, int j, int k, int l) {
    assert(i < j);
    assert(k < l);

    if (i == k && j == l) {
        // Loop over the allowed ket states.
        // The two indices k,l on (c_k c_l) must appear in the ket,
        // and then the (c*_i c^*j) restore the same indices.
        // In the ket we turn on k,l, and we need to choose the
        // remaining Q-2 amond the remaining N-2 locations.
        // So we loop over (N-2) choose (Q-2) combinations, and then
        // insert k,l indices in the appropriate places.
        for (int n = 0; n < binomial(N-2, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);
            insert_index_and_shift(indices, k);
            insert_index_and_shift(indices, l);
            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
    else if (i == k) {
        // For the state not to be annihilated by this term, 
        // it must include k,l and it must not include j.
        for (int n = 0; n < binomial(N-3, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);
            insert_index_and_shift(indices, k);

            if (j < l) {
                shift_starting_at_index(indices, j);
                insert_index_and_shift(indices, l);
            }
            else {
                assert(l < j);
                insert_index_and_shift(indices, l);
                shift_starting_at_index(indices, j);
            }

            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
    else if (j == l) {
        // For the state not to be annihilated by this term, 
        // it must include k,l and it must not include i.
        for (int n = 0; n < binomial(N-3, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);

            if (i < k) {
                shift_starting_at_index(indices, i);
                insert_index_and_shift(indices, k);
            }
            else {
                assert(k < i);
                insert_index_and_shift(indices, k);
                shift_starting_at_index(indices, i);
            }

            insert_index_and_shift(indices, l);
            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
    else if (i == l) {
        // Order is k < i=l < j
        for (int n = 0; n < binomial(N-3, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);
            insert_index_and_shift(indices, k);
            insert_index_and_shift(indices, l);
            shift_starting_at_index(indices, j);
            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
    else if (j == k) {
        // Order is i < j=k < l
        for (int n = 0; n < binomial(N-3, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);
            shift_starting_at_index(indices, i);
            insert_index_and_shift(indices, k);
            insert_index_and_shift(indices, l);
            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
    else {
        assert(i != k);
        assert(i != l);
        assert(j != k);
        assert(j != l);

        // All i,j,k,l are different.
        // For the state not to be annihilated by this term, 
        // it must include k,l and it must not include i,j.
        for (int n = 0; n < binomial(N-4, Q-2); n++) {
            list<int> indices = state_number_to_occupations(n, Q-2);

            // All allowed permutations (i<j and k<l):
            // i j k l (i.e. i < j < k < l)
            // i k j l
            // k i j l
            // i k l j
            // k i l j
            // k l i j

            if (i < j && j < k && k < l) {
                shift_starting_at_index(indices, i);
                shift_starting_at_index(indices, j);
                insert_index_and_shift(indices, k);
                insert_index_and_shift(indices, l);
            }
            else if (i < k && k < j && j < l) {
                shift_starting_at_index(indices, i);
                insert_index_and_shift(indices, k);
                shift_starting_at_index(indices, j);
                insert_index_and_shift(indices, l);
            }
            else if (k < i && i < j && j < l) {
                insert_index_and_shift(indices, k);
                shift_starting_at_index(indices, i);
                shift_starting_at_index(indices, j);
                insert_index_and_shift(indices, l);
            }
            else if (i < k && k < l && l < j) {
                shift_starting_at_index(indices, i);
                insert_index_and_shift(indices, k);
                insert_index_and_shift(indices, l);
                shift_starting_at_index(indices, j);
            }
            else if (k < i && i < l && l < j) {
                insert_index_and_shift(indices, k);
                shift_starting_at_index(indices, i);
                insert_index_and_shift(indices, l);
                shift_starting_at_index(indices, j);
            }
            else if (k < l && l < i && i < j) {
                insert_index_and_shift(indices, k);
                insert_index_and_shift(indices, l);
                shift_starting_at_index(indices, i);
                shift_starting_at_index(indices, j);
            }

            add_term_and_state_contribution(J, i, j, k, l, indices);
        }
    }
}

// A slow and straightforward implementation
void NaiveKitaevHamiltonianBlock::initialize_block_matrix_naive_implementation(
        DisorderParameter* J) {
    matrix = Mat::Zero(dim(), dim());

    // Overall Hamiltonian coefficient.
    // Factor of 4 is because we are only summing over i<j and k<l
    double coefficient = 1./pow(2. * N, 3./2.) * 4.;

    // n is the state number of the ket
    for (int ket_n = 0; ket_n < dim(); ket_n++) {
        BasisState ket(ket_n, Q);

        // Act with all the Hamiltonian terms
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = k+1; l < N; l++) {
                        //cout << i << "," << j << "," << k << "," << l << endl;
                        BasisState bra = act_with_c_operators(
                                i, j, k, l, ket);

                        if (bra.is_zero()) {
                            continue;
                        }

                        assert(bra.charge() == Q);
                        int bra_n = bra.get_state_number();

                        assert(bra_n >= 0);
                        assert(bra_n < dim());
                        matrix(bra_n, ket_n) += 
                            J->elem(i,j,k,l) * cpx(bra.coefficient, 0);
                    }
                }
            }
        }
    }

    matrix *= coefficient;
}

// Compute: 
// c^\dagger_i c^\dagger_j c_k c_l |ket>
BasisState KitaevHamiltonianBlock::act_with_c_operators(
        int i, int j, int k, int l, BasisState ket) {
    ket.annihilate(l);
    ket.annihilate(k);
    ket.create(j);
    ket.create(i);
    return ket;
}

int KitaevHamiltonianBlock::dim() {
    return binomial(N, Q);
}

cpx KitaevHamiltonianBlock::operator()(int n, int m) {
    return matrix(n, m);
}

void KitaevHamiltonianBlock::diagonalize(bool full_diagonalization) {
    if (diagonalized) {
        return;
    }
    
    SelfAdjointEigenSolver<Mat> solver;

    if (full_diagonalization) {
        solver.compute(matrix);
        evs = solver.eigenvalues();
        U = solver.eigenvectors();
    }
    else {
        // Only compute the eigenvalues
        solver.compute(matrix, EigenvaluesOnly);
        evs = solver.eigenvalues();
    }

    diagonalized = true;
}

bool KitaevHamiltonianBlock::is_diagonalized() {
    return diagonalized;
}

RealVec KitaevHamiltonianBlock::eigenvalues() {
    assert(diagonalized);
    return evs;
}

Mat KitaevHamiltonianBlock::D_matrix() {
    assert(diagonalized);
    Mat D = Mat::Zero(dim(), dim());

    for (int i = 0; i < dim(); i++) {
        D(i,i) = evs(i);
    }

    return D;
}

Mat KitaevHamiltonianBlock::U_matrix() {
    assert(diagonalized);
    return U;
}

