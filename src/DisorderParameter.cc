#include <boost/random/normal_distribution.hpp>
#include "DisorderParameter.h"

DisorderParameter::~DisorderParameter() {}
KitaevDisorderParameter::~KitaevDisorderParameter() {}
MockDisorderParameter::~MockDisorderParameter() {}

KitaevDisorderParameter::KitaevDisorderParameter(
        int N, double J, boost::random::mt19937* gen, 
        bool complex_elements) :
    Jelems(boost::extents[N][N][N][N]) {

    boost::random::normal_distribution<> dist_diag(0, J);
    boost::random::normal_distribution<> dist_off_diag(0, J/sqrt(2.));

    // Choose random J elements
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    if (k == i && l == j) {
                        // Diagonal element is real
                        Jelems[i][j][i][j] = cpx(dist_diag(*gen), 0.);
                    }
                    else {
                        if (complex_elements) {
                            // Off-diagonal element is complex
                            Jelems[i][j][k][l] = cpx(
                                    dist_off_diag(*gen), 
                                    dist_off_diag(*gen));
                        }
                        else {
                            Jelems[i][j][k][l] = cpx(dist_off_diag(*gen), 0.);
                        }
                    }

                    // Apply reality condition
                    Jelems[k][l][i][j] = conj( Jelems[i][j][k][l] );

                    // Anti-symmetrize (both Jijkl and Jklij)
                    Jelems[j][i][k][l] = - Jelems[i][j][k][l];
                    Jelems[i][j][l][k] = - Jelems[i][j][k][l];
                    Jelems[j][i][l][k] = + Jelems[i][j][k][l];

                    Jelems[l][k][i][j] = - Jelems[k][l][i][j];
                    Jelems[k][l][j][i] = - Jelems[k][l][i][j];
                    Jelems[l][k][j][i] = + Jelems[k][l][i][j];
                }
            }
        }
    }
}

cpx KitaevDisorderParameter::elem(int i, int j, int k, int l) {
    return Jelems[i][j][k][l];
}

cpx MockDisorderParameter::elem(int i, int j, int k, int l) {
    return i - j + k - l;
}


