#ifndef FOCK_SPACE_UTILS_H__ 
#define FOCK_SPACE_UTILS_H__

#include <vector>
#include "defs.h"
#include "eigen_utils.h"

using namespace std;

// Hilbert space dimension with N Dirac fermions
int dim(int N);

// Dimension of the given Q sector in the Dirac model
int Q_sector_dim(int N, int Q);

// Compute the dense matrix representation of c_i between the Q and Q-1 sectors.
Mat compute_c_matrix_dense(int i, int N, int Q);

// Compute the dense matrix representation of cdagger_i between the Q and 
// Q+1 sectors.
Mat compute_cdagger_matrix_dense(int i, int N, int Q);

// Compute the sparse matrix representation of c_i between the Q and Q-1 
// sectors.
SpMat compute_c_matrix(int i, int N, int Q);

// Compute the sparse matrix representation of cdagger_i between the Q and 
// Q+1 sectors.
SpMat compute_cdagger_matrix(int i, int N, int Q);

// Compute the sparse matrix representation of c_i.
// The states are labelled using global numbers.
SpMat compute_c_matrix(int i, int N);

// Compute the sparse matrix representation of cdagger_i.
// The states are labelled using global numbers.
SpMat compute_cdagger_matrix(int i, int N);

// Compute the sparse matrix representation of the Majorana fermion chi_a.
// Here a is the Majorana label and mN is the number of Majorana fermions
// (Dirac N = mN / 2).
SpMat compute_chi_matrix(int a, int mN);

// Compute all chi_a matrices, a=0,...,mN-1
vector<SpMat> compute_chi_matrices(int mN);

// Hilbert space of N Majorana fermions
class Space {
public:
    Space();
    Space(const Space& other);
    virtual Space& operator=(const Space& other);
    
    static Space from_majorana(int N);
    static Space from_dirac(int Nd);
    
    int N;  // num Majorana fermions
    int Nd; // num Dirac fermions
    int D;  // dimension

 protected:
    void init_dirac(int _Nd);
};

#endif // FOCK_SPACE_UTILS_H__
