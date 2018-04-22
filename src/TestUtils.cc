#include "TestUtils.h"

void assert_matrix_is_zero(
    Mat matrix, 
    string description,
    double _epsilon) {
    double diff = sqrt(abs((matrix.adjoint() * matrix).trace()));
    ASSERT_NEAR(0., diff, _epsilon) << description;
}

void assert_equal_matrices(
    Mat expected, 
    Mat actual, 
    string description,
    double _epsilon) {
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    assert_matrix_is_zero(expected - actual, description, _epsilon);
}

void assert_equal_vectors(RealVec& expected,
                          RealVec& actual,
                          string description,
                          double _epsilon) {
    ASSERT_EQ(expected.size(), actual.size());
    double diff = (expected - actual).norm();

    if (diff > _epsilon) {
        int num_printed = 0;
        
        for (int i = 0; i < expected.size(); i++) {
            if (abs(expected(i) - actual(i)) > _epsilon) {
                if (num_printed > 3) {
                    cerr << "...\n";
                    break;
                }
                
                cerr << "Difference at index " << i << ":\n";
                cerr << "\texpected(i) = " << expected(i) << "\n";
                cerr << "\tactual(i) = " << actual(i) << "\n";
                num_printed++;
            }
        }
    }
    
    ASSERT_NEAR(0., diff, _epsilon) << "That was: " << description;
}

void assert_equal_vectors(Vec& expected,
                          Vec& actual,
                          string description,
                          double _epsilon) {
    ASSERT_EQ(expected.size(), actual.size());
    double diff = (expected - actual).norm();
    ASSERT_NEAR(0., diff, _epsilon) << description;
}
                              
void assert_cpx_equal(cpx a, cpx b, string description) {
    ASSERT_NEAR(real(a), real(b), epsilon) << description + " (real)";
    ASSERT_NEAR(imag(a), imag(b), epsilon) << description + " (imaginary)";
}

void verify_same_evs(const RealVec& true_evs,
                     const RealVec& actual_evs) {
    // Make sure it claims we found all the eigenvalues, then make
    // sure we really did
    ASSERT_EQ(actual_evs.size(), true_evs.size());

    // Compare eigenvalues
    int j = 0;
    bool matched_prev = false;

    // needed to be bigger than epsilon for N=16
    double tolerance = 1e-8; 

    for (int i = 0; i < true_evs.size(); i++) {
        if (relative_err(actual_evs(j), true_evs(i)) < tolerance) {
            j++;
            matched_prev = true;
        }
        else {
            // Make sure there are no gaps in the matches: once we
            // start matching, we need to match all subsequent
            // eigenvalues
            ASSERT_FALSE(matched_prev) << "i=" << i << " j=" << j;
        }

        if (j == actual_evs.size()) {
            // Make sure we got to the last eigenvalue, namely that
            // we matched the largest evs
            ASSERT_EQ(i, true_evs.size() - 1);
        }
    }

    // Make sure claimed good eigenvalues match
    ASSERT_EQ(j, actual_evs.size());
}
