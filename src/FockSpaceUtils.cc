#include "BasisState.h"
#include "FockSpaceUtils.h"

int dim(int N) {
    return pow(2,N);
}

int Q_sector_dim(int N, int Q) {
    return binomial(N, Q);
}

// Compute the matrix representation of c_i between the Q and Q-1 sectors.
Mat compute_c_matrix_dense(int i, int N, int Q) {
    assert(Q > 0);
    Mat c_i = Mat::Zero(Q_sector_dim(N,Q-1), Q_sector_dim(N,Q));

    for (int ket_state_num = 0; 
         ket_state_num < Q_sector_dim(N,Q);
         ket_state_num++) {
        BasisState state(ket_state_num, Q);
        state.annihilate(i);

        if (!state.is_zero()) {
            int bra_state_num = state.get_state_number();
            c_i(bra_state_num, ket_state_num) = state.coefficient;
        }
    }

    return c_i;
}

// Compute the matrix representation of cdag_i between the Q and 
// Q+1 sectors.
Mat compute_cdagger_matrix_dense(int i, int N, int Q) {
    assert(Q < N);
    Mat cdag_i = Mat::Zero(Q_sector_dim(N,Q+1), Q_sector_dim(N,Q));

    for (int ket_state_num = 0; 
         ket_state_num < Q_sector_dim(N,Q);
         ket_state_num++) {
        BasisState state(ket_state_num, Q);
        state.create(i);

        if (!state.is_zero()) {
            int bra_state_num = state.get_state_number();
            cdag_i(bra_state_num, ket_state_num) = state.coefficient;
        }
    }

    return cdag_i;
}

SpMat compute_c_matrix(int i, int N, int Q) {
    assert(Q > 0);
    vector<CpxTriplet> triplets;

    for (int ket_state_num = 0; 
         ket_state_num < Q_sector_dim(N,Q);
         ket_state_num++) {
        BasisState state(ket_state_num, Q);
        state.annihilate(i);

        if (!state.is_zero()) {
            int bra_state_num = state.get_state_number();
            triplets.push_back(CpxTriplet(
                        bra_state_num,
                        ket_state_num,
                        cpx(state.coefficient,0.)));
        }
    }

    SpMat c_i(Q_sector_dim(N,Q-1), Q_sector_dim(N,Q));
    c_i.setFromTriplets(triplets.begin(), triplets.end());
    return c_i;
}

SpMat compute_cdagger_matrix(int i, int N, int Q) {
    assert(Q < N);
    vector<CpxTriplet> triplets;

    for (int ket_state_num = 0; 
         ket_state_num < Q_sector_dim(N,Q);
         ket_state_num++) {
        BasisState state(ket_state_num, Q);
        state.create(i);

        if (!state.is_zero()) {
            int bra_state_num = state.get_state_number();
            triplets.push_back(CpxTriplet(
                        bra_state_num,
                        ket_state_num,
                        cpx(state.coefficient,0.)));
        }
    }

    SpMat cdag_i(Q_sector_dim(N,Q+1), Q_sector_dim(N,Q));
    cdag_i.setFromTriplets(triplets.begin(), triplets.end());
    return cdag_i;
}

SpMat compute_c_matrix(int i, int N) {
    vector<CpxTriplet> triplets;

    for (int ket_global_num = 0; 
         ket_global_num < dim(N);
         ket_global_num++) {
        BasisState state(ket_global_num);
        state.annihilate(i);

        if (!state.is_zero()) {
            int bra_global_num = state.get_global_state_number();
            triplets.push_back(CpxTriplet(
                        bra_global_num,
                        ket_global_num,
                        cpx(state.coefficient,0.)));
        }
    }

    SpMat c_i(dim(N), dim(N));
    c_i.setFromTriplets(triplets.begin(), triplets.end());
    return c_i;
}

SpMat compute_cdagger_matrix(int i, int N) {
    vector<CpxTriplet> triplets;

    for (int ket_global_num = 0; 
         ket_global_num < dim(N);
         ket_global_num++) {
        BasisState state(ket_global_num);
        state.create(i);

        if (!state.is_zero()) {
            int bra_global_num = state.get_global_state_number();
            triplets.push_back(CpxTriplet(
                        bra_global_num,
                        ket_global_num,
                        cpx(state.coefficient,0.)));
        }
    }

    SpMat c_i(dim(N), dim(N));
    c_i.setFromTriplets(triplets.begin(), triplets.end());
    return c_i;
}

vector<SpMat> compute_chi_matrices(int mN) {
    assert(mN % 2 == 0);
    int dirac_N = mN / 2;
    vector<SpMat> chi(mN);

    // Re-use c,cd matrices
    for (int a = 0; a < mN; a += 2) {
        int i = a/2;
        SpMat c_i = compute_c_matrix(i, dirac_N);
        SpMat cd_i = compute_cdagger_matrix(i, dirac_N);

        chi[a] = (c_i + cd_i) * cpx(1./sqrt(2.), 0.);
        chi[a+1] = (c_i - cd_i) * cpx(0.,1./sqrt(2.));
    }

    return chi;
}

SpMat compute_chi_matrix(int a, int mN) {
    assert(mN % 2 == 0);
    assert(a >= 0);
    assert(a < mN);

    int dirac_N = mN / 2;
    SpMat chi(dim(dirac_N), dim(dirac_N));

    if (a % 2 == 0) {
        int i = a/2;
        chi = compute_c_matrix(i, dirac_N) 
            + compute_cdagger_matrix(i, dirac_N);
        chi /= sqrt(2.);
    }
    else {
        int i = (a-1) / 2;
        chi = cpx(0.,1./sqrt(2.)) * (
                compute_c_matrix(i, dirac_N) 
                - compute_cdagger_matrix(i, dirac_N)
                );
    }

    return chi;
}

Space::Space(const Space& other) {
    N = other.N;
    Nd = other.Nd;
    D = other.D;
}

Space& Space::operator=(const Space& other) {
    N = other.N;
    Nd = other.Nd;
    D = other.D;
    return *this;
}

Space Space::from_majorana(int N) {
    assert(N % 2 == 0);
    
    Space space;
    space.init_dirac(N / 2);
    return space;
}

Space Space::from_dirac(int Nd) {
    Space space;
    space.init_dirac(Nd);
    return space;
}

Space::Space() {
    N = -1;
    Nd = -1;
    D = -1;
}

void Space::init_dirac(int _Nd) {
    Nd = _Nd;
    N = 2 * Nd;
    D = dim(Nd);
}

