#include <iostream>
#include <ctime>
#include "KitaevHamiltonian.h"
#include "Timer.h"

using namespace std;

KitaevHamiltonian::KitaevHamiltonian(
        int _N, DisorderParameter* J) : N(_N) {
    for (int Q = 0; Q <= N; Q++) {
        blocks.push_back(new KitaevHamiltonianBlock(N, Q, J));
    }
}

KitaevHamiltonian::~KitaevHamiltonian() {
    for (int Q = 0; Q <= N; Q++) {
        if (blocks[Q] != 0) {
            delete blocks[Q];
            blocks[Q] = 0;
        }
    }
}

int KitaevHamiltonian::dim() {
    return pow(2,N);
}

void KitaevHamiltonian::diagonalize(
        bool full_diagonalization,
        bool print_progress) {
    for (int Q = 0; Q <= N; Q++) {
        if (print_progress) {
            cout << "Diagonalizing Q=" << Q << " ... ";
            cout.flush();
        }

        Timer timer;
        blocks[Q]->diagonalize(full_diagonalization);

        if (print_progress) {
            cout << "took " << timer.seconds() << " seconds" << endl;
        }
    }

    if (print_progress) {
        cout << "Diagonalization complete." << endl;
    }
}

bool KitaevHamiltonian::is_diagonalized() {
    return blocks[0]->diagonalized;
}

RealVec KitaevHamiltonian::eigenvalues() {
    assert(is_diagonalized());
    RealVec evs = RealVec(dim());

    int k = 0;

    for (int Q = 0; Q <= N; Q++) {
        int block_dim = blocks[Q]->dim();
        evs.block(k, 0, block_dim, 1) = blocks[Q]->eigenvalues();
        k += block_dim;
    }

    return evs;
}

Mat KitaevHamiltonian::as_matrix() {
    Mat H = Mat::Zero(dim(), dim());

    int block_row = 0;
    int block_col = 0;

    for (int Q = 0; Q <= N; Q++) {
        int block_dim = blocks[Q]->dim();
        H.block(block_row, block_col, block_dim, block_dim) =
            blocks[Q]->matrix;
        block_row += block_dim;
        block_col += block_dim;
    }

    return H;
}

