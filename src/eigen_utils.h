#ifndef EIGEN_DEFS_H__ 
#define EIGEN_DEFS_H__

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <boost/random/mersenne_twister.hpp>
#include "defs.h"

using namespace Eigen;
using namespace std;

typedef Matrix< cpx, Dynamic, Dynamic > Mat;
typedef Matrix< double, Dynamic, Dynamic > RealMat;
typedef Matrix< double, Dynamic, 1 > RealVec;
typedef Matrix< cpx, Dynamic, 1 > Vec;
typedef SparseMatrix< cpx > SpMat;
typedef SparseMatrix< double > RealSpMat;
typedef SparseMatrix< cpx, RowMajor > RowMajorSpMat; // for cuSPARSE

typedef Eigen::Triplet<cpx> CpxTriplet;
typedef Eigen::Triplet<double> RealTriplet;

Vec get_random_vector(int size, boost::random::mt19937* gen);
RealVec get_random_real_vector(int size, boost::random::mt19937* gen);
Mat get_random_matrix(int rows, int cols, boost::random::mt19937* gen);

void write_matrix_binary(ofstream& out, const Mat& matrix);
Mat read_matrix_binary(ifstream& in);

void write_vector_binary(ofstream& out, const Vec& vec);
Vec read_vector_binary(ifstream& in);

void write_real_vector_binary(ofstream& out, const RealVec& vec);
RealVec read_real_vector_binary(ifstream& in);

// Returns the first n elements of v
RealVec get_head(const RealVec& v, int n);

// v should be sorted. returns the vector of elements that are different
// up to unique_eps.
RealVec get_unique_elements(RealVec& v, double unique_eps);

// Add the non-zero elements of mat to the triplets vector.
// rows and cols are offset by row_offset and col_offset.
void add_nonzeros_to_triplets(vector<CpxTriplet>& triplets, const Mat& mat,
                              int row_offset = 0, int col_offset = 0);

#endif // EIGEN_DEFS_H__
