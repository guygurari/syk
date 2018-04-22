//
// This program is used to test and benchmark new ideas.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Eigenvalues>

#include "defs.h"
#include "eigen_utils.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "Timer.h"

//#include <lapacke.h>

// Definitions from lapacke.h, which we cannot include for some reason
#define lapack_int int
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

extern "C" {
lapack_int LAPACKE_dsteqr( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_dstedc( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
void dstedc_(
    char* compz, int* N, double* D, double* E, double* Z,
    int* LDZ, double* WORK, int* LWORK, int* IWORK, int* LIWORK, int* INFO);
}

void benchmark_sparse_implementations(FactorizedSpace space,
                                      boost::random::mt19937* gen) {
    
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    cout << "\n\nCreating Hamiltonians with Nd=" << space.Nd
         << " Nd_left=" << space.left.Nd
         << endl;
    FactorizedHamiltonian dense_H(space, Jtensor);
    SparseFactorizedHamiltonian sparse_H(space, Jtensor);
    HalfSparseFactorizedHamiltonian half_sparse_H(space, Jtensor);

    Mat state = get_factorized_random_state(space, gen);
        Mat::Zero(space.left.D, space.right.D);
    Mat output = Mat::Zero(space.left.D, space.right.D);

    int iterations = 10;
    Timer timer;

    // cout << "Running dense implementation\n";
    // timer.reset();
    // for (int i = 0; i < iterations; i++) {
    //     dense_H.act_even(output, state);
    // }
    // timer.print_msec();

    // cout << "\nRunning sparse implementation\n";
    // timer.reset();
    // for (int i = 0; i < iterations; i++) {
    //     sparse_H.act_even(output, state);
    // }
    // timer.print_msec();

    cout << "Running half sparse implementation\n";
    timer.reset();
    for (int i = 0; i < iterations; i++) {
        half_sparse_H.act_even(output, state);
    }
    timer.print_msec();
}

void benchmark_tridiagonal_eig(boost::random::mt19937* gen) {
    const int D = 2000;

    cout << "Diagonalizing tridiagonal matrix of dimension " << D
         << " including eigenvectors ..." << endl;

    RealVec diag = get_random_real_vector(D, gen);
    RealVec subdiag = get_random_real_vector(D-1, gen);

    // SelfAdjointEigenSolver<RealMat> solver;
    // Timer timer;
    // solver.computeFromTridiagonal(diag, subdiag, EigenvaluesOnly);
    // solver.computeFromTridiagonal(diag, subdiag, ComputeEigenvectors);
    // timer.print();

    vector<RealTriplet> triplets;

    for (int i = 0; i < D; i++) {
        triplets.push_back(RealTriplet(i, i, diag(i)));

        if (i < D-1) {
            triplets.push_back(RealTriplet(i, i+1, subdiag(i)));
            triplets.push_back(RealTriplet(i+1, i, subdiag(i)));
        }
    }

    RealSpMat mat(D, D);
    mat.setFromTriplets(triplets.begin(), triplets.end());

    SelfAdjointEigenSolver<RealSpMat> solver;

    Timer timer;
    solver.compute(mat, ComputeEigenvectors);
    timer.print();
}

void benchmark_dense_tridiagonal_eig(boost::random::mt19937* gen) {
    const int D = 5000;

    cout << "Diagonalizing dense tridiagonal matrix of dimension " << D
         << " including eigenvectors" << endl;

    RealVec diag = get_random_real_vector(D, gen);
    RealVec subdiag = get_random_real_vector(D-1, gen);
    RealMat mat = RealMat::Zero(D, D);

    for (int i = 0; i < D; i++) {
        mat(i, i) = diag(i);

        if (i < D-1) {
            mat(i, i+1) = subdiag(i);
            mat(i+1, i) = subdiag(i);
        }
    }

    SelfAdjointEigenSolver<RealMat> solver;

    Timer timer;
    // solver.compute(mat, EigenvaluesOnly);
    solver.compute(mat, ComputeEigenvectors);
    timer.print();
}

void benchmark_lapack_tridiagonal_eig(
    boost::random::mt19937* gen, int N) {

    cout << "Diagonalizing tridiagonal matrix of dimension " << N
         << " including eigenvectors with LAPACK" << endl;

    RealVec diag = get_random_real_vector(N, gen);
    RealVec subdiag = get_random_real_vector(N-1, gen);

    // int ldz = N;

    RealMat Z = RealMat::Identity(N, N);

    // int result = LAPACKE_dsteqr(LAPACK_COL_MAJOR, 'V', N,
    //                             diag.data(), subdiag.data(), Z.data(),
    //                             ldz);

    // int result = LAPACKE_dstedc(LAPACK_COL_MAJOR, 'V', N,
    //                             diag.data(), subdiag.data(), Z.data(),
    //                             ldz);

    int result;
    int LWORK = -1;
    int LIWORK = -1;
    char compz = 'I';

    double WORK_size;
    int IWORK_size;

    dstedc_(&compz, &N, diag.data(), subdiag.data(), Z.data(),
            &N, &WORK_size, &LWORK, &IWORK_size, &LIWORK, &result);

    LWORK = (int) WORK_size;
    LIWORK = IWORK_size;

    cout << "LWORK = " << ((double) LWORK) / 1e6 << " MB" << endl;
    cout << "LIWORK = " << ((double) LIWORK) / 1e6 << " MB" << endl;

    double* WORK = (double*) malloc(sizeof(double) * LWORK);
    int* IWORK = (int*) malloc(sizeof(int) * LIWORK);

    assert(WORK != 0);
    assert(IWORK != 0);

    Timer timer;
    dstedc_(&compz, &N, diag.data(), subdiag.data(), Z.data(),
            &N, WORK, &LWORK, IWORK, &LIWORK, &result);
    timer.print();

    cout << "result = " << result << endl;

    free(WORK);
    free(IWORK);
}

int main(int argc, char *argv[]) {
    int seed = get_random_seed("");
    boost::random::mt19937* gen =
        new boost::random::mt19937(seed);

    if (argc != 2) {
        cerr << "N must be provided" << endl;
        return 1;
    }

    int N = atoi(argv[1]);

    // benchmark_tridiagonal_eig(gen);
    // benchmark_dense_tridiagonal_eig(gen);
    benchmark_lapack_tridiagonal_eig(gen, N);

    // benchmark_sparse_implementations(
    //     FactorizedSpace::from_dirac(14, 6), gen);
    // benchmark_sparse_implementations(
    //     FactorizedSpace::from_dirac(14, 7), gen);
    // benchmark_sparse_implementations(
    //     FactorizedSpace::from_dirac(14, 8), gen);
    // benchmark_sparse_implementations(
    //     FactorizedSpace::from_dirac(14, 8), gen);
    // benchmark_sparse_implementations(
    //     FactorizedSpace::from_dirac(14, 9), gen);

    // for (int N = 26; N <= 30; N += 2) {
    //     FactorizedSpace space = FactorizedSpace::from_majorana(N);
    //     benchmark_sparse_implementations(space, gen);
    // }
}
