#include <iostream>
#include <Eigen/Eigenvalues> 
#include "NaiveMajoranaKitaevHamiltonian.h"

using namespace std;

NaiveMajoranaKitaevHamiltonian::NaiveMajoranaKitaevHamiltonian(
        int _N, MajoranaDisorderParameter* J) 
    : N(_N), dirac_N(_N/2), diagonalized(false) {
    assert(N % 2 == 0);

    matrix = Mat::Zero(dim(), dim());
    double coefficient = 1.;

    for (int ket_n = 0; ket_n < dim(); ket_n++) {
        // Loop over the four Majorana indices
        for (int a = 0; a < N; a++) {
            for (int b = a+1; b < N; b++) {
                for (int c = b+1; c < N; c++) {
                    for (int d = c+1; d < N; d++) {
                        add_majorana_term_contributions(
                                ket_n, a, b, c, d, J);
                    }
                }
            }
        }
    }

    matrix *= coefficient;
}

void NaiveMajoranaKitaevHamiltonian::add_majorana_term_contributions(
        int ket_n,
        int a, int b, int c, int d, 
        MajoranaDisorderParameter* J) {
    // Loop over the 2^4 different terms when expanding \chi ~ c +- c*.
    //
    // a_c=0 means act with c in chi_a
    // a_c=1 means act with c* in chi_a
    // b_c=0 mean ...       c  in chi_b
    // etc.
    for (int a_c = 0; a_c < 2; a_c++) {
        for (int b_c = 0; b_c < 2; b_c++) {
            for (int c_c = 0; c_c < 2; c_c++) {
                for (int d_c = 0; d_c < 2; d_c++) {
                    cpx op_coefficient = 1.;
                    BasisState bra(ket_n);

                    op_coefficient *= act_with_dirac_operator(bra, d, d_c);
                    if (bra.coefficient == 0) continue;

                    op_coefficient *= act_with_dirac_operator(bra, c, c_c);
                    if (bra.coefficient == 0) continue;

                    op_coefficient *= act_with_dirac_operator(bra, b, b_c);
                    if (bra.coefficient == 0) continue;

                    op_coefficient *= act_with_dirac_operator(bra, a, a_c);
                    if (bra.coefficient == 0) continue;

                    int bra_n = bra.get_global_state_number();

                    // The 1/4 comes from the definitions of chi:
                    // chi = (c +- cd) / \sqrt{2}
                    // So from chi^4 we have a factor of (1/sqrt{2})^4 = 1/4.
                    matrix(bra_n, ket_n) +=
                        (J->elem(a,b,c,d)/4.) *
                        op_coefficient * 
                        cpx(bra.coefficient,0);
                }
            }
        }
    }
}

cpx NaiveMajoranaKitaevHamiltonian::act_with_dirac_operator(
        BasisState& state, int majorana_index, int c_or_cstar) {
    int dirac_index = majorana_index / 2;

    cpx coeff;

    if (c_or_cstar == 0) {
        // Act with c
        coeff = (majorana_index % 2 == 0) ? cpx(1,0) : cpx(0,1);
        state.annihilate(dirac_index);
    }
    else {
        // Act with c*
        coeff = (majorana_index % 2 == 0) ? cpx(1,0) : cpx(0,-1);
        state.create(dirac_index);
    }

    return coeff;
}

int NaiveMajoranaKitaevHamiltonian::dim() {
    return pow(2,dirac_N);
}


void NaiveMajoranaKitaevHamiltonian::diagonalize() {
    if (diagonalized) {
        return;
    }

    SelfAdjointEigenSolver<Mat> solver;
    solver.compute(matrix, EigenvaluesOnly);
    //cout << "eigenvalues:\n" << solver.eigenvalues() << endl;
    evs = solver.eigenvalues();
    diagonalized = true;
}

bool NaiveMajoranaKitaevHamiltonian::is_diagonalized() {
    return diagonalized;
}

RealVec NaiveMajoranaKitaevHamiltonian::all_eigenvalues() {
    assert(is_diagonalized());
    return evs;
}

