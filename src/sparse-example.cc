#include <Eigen/SparseCore>
typedef SparseMatrix< cpx > SpMat;

int main(int argc, const char *argv[]) {
    // Create a sparse matrix
    vector<CpxTriplet> triplets;

    int col = 10;
    double x = 1.;

    for (int row = 0; row < N; row++) {
        triplets.push_back(CpxTriplet(i, c, x));
    }

    SpMat mat(N,N);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    
    return 0;
}

