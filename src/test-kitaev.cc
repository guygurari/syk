/////////////////////////////////////////////////////////////////////////
//////////// Old tests moved to archived-tests.cc to improve compilation
//////////// times.
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "MajoranaDisorderParameter.h"
#include "MajoranaKitaevHamiltonian.h"
#include "FockSpaceUtils.h"
#include "FactorizedSpaceUtils.h"
#include "FactorizedHamiltonian.h"
#include "Lanczos.h"
#include "kitaev.h"
#include "TestUtils.h"

#include <boost/random/normal_distribution.hpp>

const bool run_all_slow_tests = false;

boost::random::mt19937* gen = new boost::random::mt19937(time(0));

// Make sure the created matrix is column-major, which is the default
// for eigen. This is important to match the cuBLAS conventions.
TEST(LanczosTest, get_factorized_random_state_column_major) {
    FactorizedSpace space = FactorizedSpace::from_dirac(4);
    assert(space.left.D == 4);
    Mat state = get_factorized_random_state(space, gen);

    cpx elem = state(1,0);
    cpx actual_elem = *(state.data() + 1);
    assert_cpx_equal(elem, actual_elem);

    elem = state(0,1);
    actual_elem = *(state.data() + space.left.D);
    assert_cpx_equal(elem, actual_elem);
}

TEST(LanczosTest, factorized_lanczos_even_Nd_sanity) {
    int N = 12;
    int m = 10;
    double mu = 1.;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    FactorizedHamiltonianGenericParity H(space, Jtensor);

    RealVec alpha;
    RealVec beta;

    factorized_lanczos(H, m, mu, gen, alpha, beta); 
}

TEST(LanczosTest, factorized_lanczos_odd_Nd_sanity) {
    int N = 10;
    int m = 200;
    double mu = 1.;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    FactorizedHamiltonianGenericParity H(space, Jtensor);

    RealVec alpha;
    RealVec beta;

    factorized_lanczos(H, m, mu, gen, alpha, beta); 
}

void factorized_lanczos_test(int N, int m) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    double mu = 1.;
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    Vec initial_state = get_random_state(space, gen);

    // Reference Lanczos
    MajoranaKitaevHamiltonian H(N, &Jtensor);

    RealVec alpha;
    RealVec beta;

    reference_lanczos(H, mu, m, alpha, beta, initial_state);

    // Factorized Lanczos
    FactorizedHamiltonianGenericParity factH(space, Jtensor);

    RealVec fact_alpha; 
    RealVec fact_beta;

    Mat fact_initial_state = get_factorized_state(space,
                                                  initial_state);

    factorized_lanczos(factH, mu, m, fact_initial_state,
                        fact_alpha, fact_beta); 

    // Compare
    assert_equal_vectors(alpha, fact_alpha);
    assert_equal_vectors(beta, fact_beta);
}

TEST(LanczosTest, factorized_lanczos_odd_Nd) {
    factorized_lanczos_test(10, 10);
}

TEST(LanczosTest, factorized_lanczos_even_Nd) {
    factorized_lanczos_test(12, 10);
}

void full_lanczos_test(int N, double mu) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    // This is 2 times bigger than the space, but we need it to get
    // all the eigenvalues
    int m = 2 * space.D;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    // Reference diagonzliation
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize();
    //RealVec unique_evs = H.even_charge_parity_evs;
    RealVec unique_evs = H.all_eigenvalues_sorted();
    unique_evs += RealVec::Constant(unique_evs.size(), mu);
    unique_evs = get_unique_elements(unique_evs, epsilon);

    // Lanczos
    FactorizedHamiltonianGenericParity factH(space, Jtensor);

    RealVec alpha;
    RealVec beta;
    Mat initial_state = get_factorized_random_state(space, gen);
    factorized_lanczos(factH, mu, m, initial_state, alpha, beta); 

    RealVec lanczos_evs = find_good_lanczos_evs(alpha, beta);

    // print_lanczos_results(unique_evs, lanczos_evs);

    // Make sure it claims we found all the eigenvalues, then make
    // sure we really did
    ASSERT_EQ(lanczos_evs.size(), unique_evs.size());

    // Compare eigenvalues
    int j = 0;
    bool matched_prev = false;

    // needed to be bigger than epsilon for N=16
    double tolerance = 1e-8; 

    for (int i = 0; i < unique_evs.size(); i++) {
        if (relative_err(lanczos_evs(j), unique_evs(i)) < tolerance) {
            j++;
            matched_prev = true;
        }
        else {
            // Make sure there are no gaps in the matches: once we
            // start matching, we need to match all subsequent
            // eigenvalues
            ASSERT_FALSE(matched_prev) << "i=" << i << " j=" << j;
        }

        if (j == lanczos_evs.size()) {
            // Make sure we got to the last eigenvalue, namely that
            // we matched the largest evs
            ASSERT_EQ(i, unique_evs.size() - 1);
        }
    }

    // Make sure claimed good eigenvalues match
    ASSERT_EQ(j, lanczos_evs.size());
}

TEST(LanczosTest, full_lanczos_process_10_mu_0) {
    full_lanczos_test(10, 0.);
}

TEST(LanczosTest, full_lanczos_process_12_mu_0) {
    full_lanczos_test(12, 0.);
}

TEST(LanczosTest, full_lanczos_process_14_mu_0) {
    full_lanczos_test(14, 0.);
}

TEST(LanczosTest, full_lanczos_process_16_mu_0) {
    full_lanczos_test(16, 0.);
}

TEST(LanczosTest, full_lanczos_process_10_mu_1) {
    full_lanczos_test(10, 1.);
}

TEST(LanczosTest, full_lanczos_process_12_mu_1) {
    full_lanczos_test(12, 1.);
}

TEST(LanczosTest, full_lanczos_process_14_mu_1) {
    full_lanczos_test(14, 1.);
}

void error_estimates_test(int N, double mu) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    // Reference diagonzliation
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize();

    RealVec unique_evs = H.all_eigenvalues_sorted();
    unique_evs += RealVec::Constant(unique_evs.size(), mu);
    unique_evs = get_unique_elements(unique_evs, epsilon);

    // Lanczos
    FactorizedHamiltonianGenericParity factH(space, Jtensor);

    for (int m = 10; m < space.D * 2; m += 10) {
        // cout << "\n\n***************  m = " << m << "  *****************\n";
        RealVec alpha;
        RealVec extended_beta;
        Mat initial_state = get_factorized_random_state(space, gen);
        factorized_lanczos(factH, mu, m, initial_state,
                           alpha, extended_beta, true); 

        RealVec error_estimates;
        RealVec lanczos_evs = find_good_lanczos_evs_and_errs(
            alpha, extended_beta, error_estimates, gen);

        ASSERT_EQ(error_estimates.size(), lanczos_evs.size());

        map<int, int> nearest = find_nearest_true_ev(unique_evs,
                                                     lanczos_evs);

        // Verify the error estimates over-estimate the true errors
        for (map<int,int>::iterator iter = nearest.begin();
             iter != nearest.end();
             ++iter) {

            double true_ev = unique_evs(iter->first);
            double lanczos_ev = lanczos_evs(iter->second);
            double err_estimate = error_estimates(iter->second);
            double true_err = abs(true_ev - lanczos_ev);

            if (err_estimate < epsilon) {
                // If the error estimate is tiny, make sure the true
                // error is tiny (it doesn't make sense to compare them
                // because they are both essentially zero)
                ASSERT_LT(true_err, epsilon);
            }
            else {
                // Errors are larger, so make sure the estimate gives an
                // upper bound
                ASSERT_LT(true_err, err_estimate);
            }
        }

        // print_lanczos_results(unique_evs, lanczos_evs, error_estimates);
    }
}

TEST(LanczosTest, error_estimates) {
    error_estimates_test(12, 0.);
}

TEST(LanczosTest, is_eigenvector) {
    int N = 400;

    RealVec alpha = get_random_real_vector(N, gen);
    RealVec beta = get_random_real_vector(N-1, gen);
    RealMat mat = get_tridiagonal_matrix(alpha, beta);

    RealVec rand = get_random_real_vector(N, gen);

    SelfAdjointEigenSolver<RealMat> solver;
    solver.compute(mat, ComputeEigenvectors);
    RealVec evs = solver.eigenvalues();
    RealMat evecs = solver.eigenvectors();

    for (int i = 0; i < N; i++) {
        RealVec evec = RealVec(evecs.block(0, i, N, 1));
        ASSERT_FALSE(is_eigenvector(alpha, beta, evs(i), rand));
        ASSERT_TRUE(is_eigenvector(alpha, beta, evs(i), evec));
    }
}

TEST(LanczosTest, find_eigenvector_for_ev) {
    int N = 400;
    RealVec alpha = get_random_real_vector(N, gen);
    RealVec beta = get_random_real_vector(N-1, gen);
    RealMat mat = get_tridiagonal_matrix(alpha, beta);

    SelfAdjointEigenSolver<RealMat> solver;
    solver.compute(mat, ComputeEigenvectors);
    RealVec evs = solver.eigenvalues();
    RealMat evecs = solver.eigenvectors();

    for (int i = 0; i < N; i++) {
        double ev = evs(i);
        RealVec expected = RealVec(evecs.block(0, i, N, 1));
        RealVec evec = find_eigenvector_for_ev(alpha, beta, ev, gen);

        ASSERT_TRUE(abs(evec.norm() - 1) < epsilon);
        ASSERT_TRUE(is_eigenvector(alpha, beta, ev, evec));

        double dot_prod = evec.dot(expected);

        // Dot product should be close to 1 or -1
        ASSERT_TRUE(abs(abs(dot_prod) - 1) < epsilon)
            << "i=" << i << " ev=" << ev
            << " dot_prod=" << dot_prod << endl;

        // Element-by-element comparison is not perfect, so tolerance
        // can't be too tight
        double elem_tolerance = 1e-6;

        // Can only compare elements up to a sign
        for (int i = 0; i < N; i++) {
            ASSERT_TRUE(abs(abs(evec(i)) - abs(expected(i)))
                        < elem_tolerance)
                << evec(i) << " vs. " << expected(i)
                << " (tolerance=" << elem_tolerance 
                << ", if they don't look too different, consider "
                << "increasing the tolerance. this test asks for "
                << "precise agreement which is not necessary.)"
                << endl;
        }
    }
}

class TestProcessor : public HamiltonianTermProcessor {
public:
    TestProcessor() {}
    virtual ~TestProcessor() {}

    virtual void process(int left_idx, FactorizedOperatorPair& ops) {
        if (num_ops.count(left_idx) == 0) {
            num_ops[left_idx] = 0;
        }

        num_ops[left_idx]++;
    }

    map<int, int> num_ops;
};

TEST(FactorizedHamiltonianTest,
     num_factorized_hamiltonian_operator_pairs) {

    int N = 10;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    TestProcessor p;

    generate_factorized_hamiltonian_terms(space, Jtensor, false, p);

    for (int i = 0; i <= q; i++) {
        ASSERT_EQ(p.num_ops[i], 
                  num_factorized_hamiltonian_operator_pairs(space, i))
            << "left_idx=" << i << endl;
    }
}

TEST(EigenUtilsTest, add_value_triplets) {
    int N = 100;
    Mat mat = get_random_matrix(N, N, gen);

    boost::random::uniform_01<> uni;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (uni(*gen) < 0.5) {
                mat(i, j) = cpx(0., 0.);
            }
        }
    }

    SpMat expected = mat.sparseView();

    vector<CpxTriplet> triplets;
    add_nonzeros_to_triplets(triplets, mat);

    SpMat actual(N, N);
    actual.setFromTriplets(triplets.begin(), triplets.end());

    assert_equal_matrices(Mat(expected), Mat(actual));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

