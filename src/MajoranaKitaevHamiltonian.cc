#include <Eigen/Eigenvalues> 
#include <iostream>

#include "MajoranaKitaevHamiltonian.h"
#include "FockSpaceUtils.h"
#include "Timer.h"

using namespace std;

typedef pair<int,int> int_pair;

MajoranaKitaevHamiltonian::MajoranaKitaevHamiltonian(
        int _mN, MajoranaDisorderParameter* J) 
    : mN(_mN), dirac_N(_mN/2), 
    diagonalized(false), 
    fully_diagonalized(false) {

    assert(mN % 2 == 0);

    matrix.resize(dim(), dim());

    chi = compute_chi_matrices(mN);

    map<int_pair, SpMat> chi_prods;

    for (int a = 0; a < mN; a++) {
        for (int b = a+1; b < mN; b++) {
            chi_prods[int_pair(a,b)] = chi[a] * chi[b];
        }
    }
    
    for (int a = 0; a < mN; a++) {
        for (int b = a+1; b < mN; b++) {
            SpMat prods_cd(dim(), dim());

            for (int c = b+1; c < mN; c++) {
                for (int d = c+1; d < mN; d++) {
                    prods_cd += chi_prods[int_pair(c,d)] * J->elem(a,b,c,d);

                    // Slower version
                    /*matrix += 
                          chi[a] * chi[b] 
                          * chi[c] * chi[d] * J->elem(a,b,c,d);*/

                    // Another slower version
                    /*matrix +=
                        chi_prods[int_pair(a,b)] 
                        * chi_prods[int_pair(c,d)]
                        * J->elem(a,b,c,d);*/
                } 
            }

            matrix += chi_prods[int_pair(a,b)] * prods_cd;
        }
    }

    // This implementation is a factor of 3 faster than the naive one
    // (for constructing the Hamiltonian)
    /*for (int a = 0; a < mN; a++) {
        SpMat chi_a = compute_chi_matrix(a, mN);

        for (int b = a+1; b < mN; b++) {
            SpMat chi_b = compute_chi_matrix(b, mN);

            for (int c = b+1; c < mN; c++) {
                SpMat chi_c = compute_chi_matrix(c, mN);

                for (int d = c+1; d < mN; d++) {
                    SpMat chi_d = compute_chi_matrix(d, mN);
                    matrix += 
                        chi_a * chi_b * chi_c * chi_d * J->elem(a,b,c,d);
                } 
            }
        }
    }*/
}

int MajoranaKitaevHamiltonian::dim() {
    return pow(2,dirac_N);
}

int MajoranaKitaevHamiltonian::charge(int global_state_number) {
    return BasisState(global_state_number).charge();
    //return __builtin_popcount(global_state_number);
}

void MajoranaKitaevHamiltonian::prepare_charge_parity_blocks(
        SpMat& even_block,
        SpMat& odd_block) {

    // Prepare lookup table of charge parities and index in
    // charge-parity sector, indexed by the Hilbert space index.
    vector<int> charge_parity(dim());
    vector<int> index_in_sector(dim());
    int even_a = 0;
    int odd_a = 0;

    for (int a = 0; a < dim(); a++) {
        charge_parity[a] = charge(a) % 2;

        if (charge_parity[a] == 0) {
            index_in_sector[a] = even_a;
            even_a++;
        }
        else {
            index_in_sector[a] = odd_a;
            odd_a++;
        }
    }

    vector<CpxTriplet> even_block_triplets;
    vector<CpxTriplet> odd_block_triplets;

    // Compute the two blocks: iterate over the non-zero elements of
    // the matrix
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (SpMat::InnerIterator it(matrix,k); it; ++it) {
            int row_parity = charge_parity[it.row()];
            int col_parity = charge_parity[it.col()];
            int sector_row = index_in_sector[it.row()];
            int sector_col = index_in_sector[it.col()];

            if (row_parity % 2 == 0 && col_parity % 2 == 0) {
                even_block_triplets.push_back(CpxTriplet(
                            sector_row,
                            sector_col,
                            it.value()));
            }
            else if (row_parity % 2 == 1 && col_parity % 2 == 1) {
                odd_block_triplets.push_back(CpxTriplet(
                            sector_row,
                            sector_col,
                            it.value()));
            }
            else {
                assert(false);
            }
        }
    }

    even_block.setFromTriplets(
        even_block_triplets.begin(), 
        even_block_triplets.end());

    odd_block.setFromTriplets(
        odd_block_triplets.begin(), 
        odd_block_triplets.end());
}

RealVec MajoranaKitaevHamiltonian::diagonalize_block(SpMat& block, bool print_progress) {
    Timer timer;
    SelfAdjointEigenSolver<Mat> solver;
    solver.compute(block, EigenvaluesOnly);

    if (print_progress) {
        cout << "took " << timer.seconds() << " seconds" << endl;
    }

    return solver.eigenvalues();
}

Mat MajoranaKitaevHamiltonian::to_energy_basis(SpMat& M) {
    return V.adjoint() * M * V;
}

Mat MajoranaKitaevHamiltonian::to_energy_basis(Mat& M) {
    return V.adjoint() * M * V;
}

RealVec MajoranaKitaevHamiltonian::eigenvalues() {
    assert(is_fully_diagonalized());
    return evs;
}

void MajoranaKitaevHamiltonian::diagonalize_full(bool print_progress) {
    if (fully_diagonalized) {
        return;
    }

    Timer timer;
    SelfAdjointEigenSolver<Mat> solver;
    solver.compute(matrix, ComputeEigenvectors);

    if (print_progress) {
        cout << "took " << timer.seconds() << " seconds" << endl;
    }

    evs = solver.eigenvalues();
    V = solver.eigenvectors();

    fully_diagonalized = true;
}

void MajoranaKitaevHamiltonian::diagonalize(bool print_progress) {
    if (diagonalized) {
        return;
    }

    if (print_progress) {
        cout << "Preparing charge parity blocks ... ";
        cout.flush();
    }

    // The theory has conserved charge parity, so we can diagonalize
    // the Q-even and Q-odd blocks seperately. Start by constructing 
    // each block.
    SpMat even_block(dim()/2, dim()/2);
    SpMat odd_block(dim()/2, dim()/2);

    Timer timer;
    prepare_charge_parity_blocks(even_block, odd_block);

    if (print_progress) {
        cout << "took " << timer.seconds() << " seconds" << endl;
    }

    // Now diagonalize the blocks...

    // Diagonalize charge-even block
    if (print_progress) {
        cout << "Diagonalizing charge-even ... ";
        cout.flush();
    }

    // Call a function to diagonalize each block.
    // This way the solver memory gets freed and we only use
    // the memory for one block at a time.
    even_charge_parity_evs = diagonalize_block(even_block, print_progress);

    // When N_dirac is odd, there is degeneracy between the charge parity
    // even and odd blocks so we only need to diagonalize one of them.
    if (dirac_N % 2 == 0) {
        // Diagonalize charge-odd block
        if (print_progress) {
            cout << "Diagonalizing charge-odd ... ";
            cout.flush();
        }

        odd_charge_parity_evs = diagonalize_block(odd_block, print_progress);
    }
    else {
        if (print_progress) {
            cout << "Charge-odd block is degenerate with charge even." << endl;
        }

        odd_charge_parity_evs = even_charge_parity_evs;
    }

    diagonalized = true;
}

bool MajoranaKitaevHamiltonian::is_diagonalized() {
    return diagonalized;
}

bool MajoranaKitaevHamiltonian::is_fully_diagonalized() {
    return fully_diagonalized;
}

RealVec MajoranaKitaevHamiltonian::all_eigenvalues() {
    assert(is_diagonalized());

    // concatenate the two vectors of eigenvalues
    RealVec evs = RealVec::Zero(dim());
    evs.block(0,0,dim()/2,1) = even_charge_parity_evs;
    evs.block(dim()/2,0,dim()/2,1) = odd_charge_parity_evs;
    return evs;
}

RealVec MajoranaKitaevHamiltonian::all_eigenvalues_sorted() {
    RealVec evs = all_eigenvalues();
    std::sort(evs.data(), evs.data() + evs.size());
    return evs;
}

Mat MajoranaKitaevHamiltonian::dense_matrix() {
    Mat dense = matrix;
    return dense;
}

