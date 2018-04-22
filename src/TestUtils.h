#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include "gtest/gtest.h"

#include "defs.h"
#include "eigen_utils.h"

extern const bool run_all_slow_tests;

void assert_matrix_is_zero(
    Mat matrix, 
    string description = "",
    double _epsilon = epsilon);

void assert_equal_matrices(
    Mat expected, 
    Mat actual, 
    string description = "",
    double _epsilon = epsilon);

void assert_equal_vectors(RealVec& expected,
                          RealVec& actual,
                          string description = "",
                          double _epsilon = epsilon);

void assert_equal_vectors(Vec& expected,
                          Vec& actual,
                          string description = "",
                          double _epsilon = epsilon);

void assert_cpx_equal(cpx a, cpx b, string description = "");

void verify_same_evs(const RealVec& true_evs,
                     const RealVec& actual_evs);

#endif
