#ifndef RANDOM_MATRIX_H__
#define RANDOM_MATRIX_H__

#include <boost/random/mersenne_twister.hpp>
#include "defs.h"
#include "eigen_utils.h"

class RandomMatrix {
public:
    RandomMatrix();
    virtual ~RandomMatrix();

    // Compute the eigenvalues and return them
    virtual RealVec eigenvalues() = 0;
};

// 
// GUE: Random Hermitian matrix with Gaussian-distributed elements.
//
class GUERandomMatrix : public RandomMatrix {
public:
    // Matrix elements have Gaussian distributions with mean=0.
    // K = matrix rank
    //
    // Ensemble weighting is:
    // weighting is exp(-K/2 * Tr(M^2))
    //
    GUERandomMatrix(int K, boost::random::mt19937* gen);

    virtual ~GUERandomMatrix();

    // Compute the eigenvalues and return them
    virtual RealVec eigenvalues();

private:
    Mat matrix;
    bool diagonalized;
    RealVec evs;
};

// 
// GOE: Random real-symmetic matrix with Gaussian-distributed elements.
//
class GOERandomMatrix : public RandomMatrix {
public:
    // Matrix elements have Gaussian distributions with mean=0.
    // K = matrix rank
    //
    // Ensemble weighting is:
    // weighting is exp(-K/4 * Tr(M^2))
    //
    GOERandomMatrix(int K, boost::random::mt19937* gen);

    virtual ~GOERandomMatrix();

    // Compute the eigenvalues and return them
    virtual RealVec eigenvalues();

private:
    RealMat matrix;
    bool diagonalized;
    RealVec evs;
};

typedef pair<int,int> int_pair;

class SparseHermitianRandomMatrix : public RandomMatrix {
public:
    SparseHermitianRandomMatrix(
            int K, int num_nonzeros, boost::random::mt19937* gen);
    virtual ~SparseHermitianRandomMatrix();

    // Compute the eigenvalues and return them
    virtual RealVec eigenvalues();

    SpMat matrix;

private:
    vector<int_pair> get_random_int_pairs(
        int K, int num_nonzeros, boost::random::mt19937* gen);

    bool diagonalized;
    RealVec evs;
};


#endif // RANDOM_MATRIX_H__
