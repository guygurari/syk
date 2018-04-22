#include <sstream>
#include <boost/random/normal_distribution.hpp>
#include "MajoranaDisorderParameter.h"

MajoranaDisorderParameter::~MajoranaDisorderParameter() {}
MajoranaKitaevDisorderParameter::~MajoranaKitaevDisorderParameter() {}
MockMajoranaDisorderParameter::~MockMajoranaDisorderParameter() {}
MajoranaKitaevDisorderParameterWithoutNeighbors::~MajoranaKitaevDisorderParameterWithoutNeighbors() {}

// Whether \chi_i and \chi_j correspond to the same Dirac fermion.
// Must have i < j.
bool is_same_dirac_fermion(int i, int j) {
    assert(i < j);
    return (i % 2 == 0) && (j == i + 1);
}

MajoranaKitaevDisorderParameterWithoutNeighbors::MajoranaKitaevDisorderParameterWithoutNeighbors(int _N, double J, boost::random::mt19937* gen) :
    MajoranaKitaevDisorderParameter(_N, J, gen) {

    // Zero out the J elements that correspond to neighboring Majoranas
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    if (is_same_dirac_fermion(i,j) ||
                        is_same_dirac_fermion(j,k) ||
                        is_same_dirac_fermion(k,l)) {

                        Jelems[i][j][k][l] = 0.;
                    }
                }
            }
        }
    }

    antisymmetrize();
}

// Assume J[i,j,k,l] is set for i<j<k<l, and set all other components
// using antisymmetry.
void MajoranaKitaevDisorderParameter::antisymmetrize() {
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    Jelems[i][j][l][k] = - Jelems[i][j][k][l];
                    Jelems[i][k][j][l] = - Jelems[i][j][k][l];
                    Jelems[i][k][l][j] = + Jelems[i][j][k][l];
                    Jelems[i][l][j][k] = + Jelems[i][j][k][l];
                    Jelems[i][l][k][j] = - Jelems[i][j][k][l];
                    Jelems[j][i][k][l] = - Jelems[i][j][k][l];
                    Jelems[j][i][l][k] = + Jelems[i][j][k][l];
                    Jelems[j][k][i][l] = + Jelems[i][j][k][l];
                    Jelems[j][k][l][i] = - Jelems[i][j][k][l];
                    Jelems[j][l][i][k] = - Jelems[i][j][k][l];
                    Jelems[j][l][k][i] = + Jelems[i][j][k][l];
                    Jelems[k][i][j][l] = + Jelems[i][j][k][l];
                    Jelems[k][i][l][j] = - Jelems[i][j][k][l];
                    Jelems[k][j][i][l] = - Jelems[i][j][k][l];
                    Jelems[k][j][l][i] = + Jelems[i][j][k][l];
                    Jelems[k][l][i][j] = + Jelems[i][j][k][l];
                    Jelems[k][l][j][i] = - Jelems[i][j][k][l];
                    Jelems[l][i][j][k] = - Jelems[i][j][k][l];
                    Jelems[l][i][k][j] = + Jelems[i][j][k][l];
                    Jelems[l][j][i][k] = + Jelems[i][j][k][l];
                    Jelems[l][j][k][i] = - Jelems[i][j][k][l];
                    Jelems[l][k][i][j] = - Jelems[i][j][k][l];
                    Jelems[l][k][j][i] = + Jelems[i][j][k][l];
                }
            }
        }
    }
}

MajoranaKitaevDisorderParameter::MajoranaKitaevDisorderParameter(
        int _N, double J, boost::random::mt19937* gen) :
    N(_N), Jelems(boost::extents[N][N][N][N]) {

    double sigma = sqrt(6 * pow(J,2) / pow(N,3));
    boost::random::normal_distribution<> dist(0, sigma);

    // Choose random J elements
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    Jelems[i][j][k][l] = dist(*gen);
                }
            }
        }
    }

    antisymmetrize();
}

MajoranaKitaevDisorderParameter::MajoranaKitaevDisorderParameter(int _N) :
    N(_N), Jelems(boost::extents[N][N][N][N]) {

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    Jelems[i][j][k][l] = 0.;
                }
            }
        }
    }

    antisymmetrize();
}

double MajoranaKitaevDisorderParameter::elem(int i, int j, int k, int l) {
    return Jelems[i][j][k][l];
}

string MajoranaKitaevDisorderParameter::to_string() {
    stringstream s;

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    s << i << "," << j << "," << k << "," << l << ": "
                        << elem(i,j,k,l) << "\n";
                }
            }
        }
    }

    return s.str();
}

double MockMajoranaDisorderParameter::elem(int i, int j, int k, int l) {
    return i - j + k - l;
}

