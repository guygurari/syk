#include <map>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <Eigen/Eigenvalues> 
#include "RandomMatrix.h"

RandomMatrix::RandomMatrix() {}
RandomMatrix::~RandomMatrix() {}
GUERandomMatrix::~GUERandomMatrix() {}
GOERandomMatrix::~GOERandomMatrix() {}

GUERandomMatrix::GUERandomMatrix(int K, boost::random::mt19937* gen) {
    diagonalized = false;
    matrix = Mat::Zero(K,K);

    // Different variance for diagonal and (real and imaginary parts of)
    // off-diagonal elements
    boost::random::normal_distribution<> dist_diag(0, sqrt(1./K));

    // Off-diagonal elements are complex
    boost::random::normal_distribution<> dist_off_diag(0, sqrt(1./(2.*K)));

    for (int i = 0; i < K; i++) {
        matrix(i,i) = cpx(dist_diag(*gen), 0.);

        for (int j = i+1; j < K; j++) {
            matrix(i,j) = cpx(
                    dist_off_diag(*gen), 
                    dist_off_diag(*gen));
            matrix(j,i) = conj(matrix(i,j));
        }
    }
}

RealVec GUERandomMatrix::eigenvalues() {
    if (!diagonalized) {
        diagonalized = true;
        SelfAdjointEigenSolver<Mat> solver;
        solver.compute(matrix, EigenvaluesOnly);
        evs = solver.eigenvalues();
    }

    return evs;
}

GOERandomMatrix::GOERandomMatrix(int K, boost::random::mt19937* gen) {
    diagonalized = false;
    matrix = RealMat::Zero(K,K);
    
    // Different variances for diagonal and off-diagonal elements
    boost::random::normal_distribution<> dist_diag(0, sqrt(2./K));
    boost::random::normal_distribution<> dist_off_diag(0, sqrt(1./K));

    for (int i = 0; i < K; i++) {
        matrix(i,i) = dist_diag(*gen);

        for (int j = i+1; j < K; j++) {
            matrix(i,j) = dist_off_diag(*gen);
            matrix(j,i) = matrix(i,j);
        }
    }
}

RealVec GOERandomMatrix::eigenvalues() {
    if (!diagonalized) {
        diagonalized = true;
        SelfAdjointEigenSolver<RealMat> solver;
        solver.compute(matrix, EigenvaluesOnly);
        evs = solver.eigenvalues();
    }

    return evs;
}

vector<int_pair> SparseHermitianRandomMatrix::get_random_int_pairs(
        int K, int num_nonzeros, boost::random::mt19937* gen) {
    assert(num_nonzeros <= ((double) K)*((double) K));
    map<int_pair,int> positions_map;
    vector<int_pair> positions;
    boost::random::uniform_int_distribution<> dist(0,K-1);

    while (positions_map.size() < num_nonzeros - 1) {
        pair<int,int> pos(dist(*gen), dist(*gen));

        // If it doesn't already exist, add it
        if (positions_map.find(pos) == positions_map.end()) {
            positions_map[pos] = 1;
            positions.push_back(pos);

            // For the off-diagonal element, 
            // also add the transpose to the map so we don't add it again
            if (pos.first != pos.second) {
                pair<int,int> transpose_pos(pos.second, pos.first);
                positions_map[transpose_pos] = 1;
            }
        }
    }

    while (positions_map.size() < num_nonzeros) {
        // Only one element left, so it must be diagonal
        assert(positions_map.size() == num_nonzeros - 1);
        int x = dist(*gen);
        pair<int,int> pos(x, x);

        // If it doesn't already exist, add it
        if (positions_map.find(pos) == positions_map.end()) {
            positions_map[pos] = 1;
            positions.push_back(pos);
        }
    }

    return positions;
}

SparseHermitianRandomMatrix::SparseHermitianRandomMatrix(
        int K, int num_nonzeros, boost::random::mt19937* gen) :
    matrix(K,K), diagonalized(false) {
    // Use the distributions from GUE
    boost::random::normal_distribution<> dist_diag(0, sqrt(2./K));
    boost::random::normal_distribution<> dist_off_diag(0, sqrt(1./K));

    vector<CpxTriplet> triplets;

    // This function needs to make sure the pairs don't collide under
    // transpose
    vector<int_pair> positions = get_random_int_pairs(
            K, num_nonzeros, gen);

    for (int i = 0; i < positions.size(); i++) {
        int row = positions[i].first;
        int col = positions[i].second;

        if (row == col) {
            double value = dist_diag(*gen);
            triplets.push_back(CpxTriplet(
                        row,
                        col,
                        value));
        }
        else {
            cpx value = cpx(dist_off_diag(*gen), dist_off_diag(*gen));
            triplets.push_back(CpxTriplet(row, col, value));
            triplets.push_back(CpxTriplet(col, row, conj(value)));
        }
    }

    assert(triplets.size() == num_nonzeros);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}

SparseHermitianRandomMatrix::~SparseHermitianRandomMatrix() {}

RealVec SparseHermitianRandomMatrix::eigenvalues() {
    if (!diagonalized) {
        diagonalized = true;
        SelfAdjointEigenSolver<Mat> solver;
        solver.compute(matrix, EigenvaluesOnly);
        evs = solver.eigenvalues();
    }

    return evs;
}

