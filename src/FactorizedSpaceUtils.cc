/////////////////////////////////////////////////////////////////////
//
// Utilities to work with the factorized-space representation of
// SYK. For N Majorana fermions, with Hilbert space dimension
// D = 2^{N/2}, we split the space to left and right, both with
// dimension \sqrt{D} (assume for simplicity N is divisible by 4).
// An operator on the total space can be written as a sum of O_L . O_R
// where . means tensor product. The state vector V is written as a
// \sqrt{D} x \sqrt{D} matrix, and the action of the operator becomes:
//     O_L v O_R^T .
// For both spaces, our basis ordering puts even charge states first,
// followed by odd charge states. This makes our O_L and O_R operators
// have block-diagonal structure. The state is block-diagonal for even
// charge states, and off-block-diagonal for odd charge states.
//
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include <boost/random/normal_distribution.hpp>
#include "FactorizedSpaceUtils.h"
#include "FockSpaceUtils.h"
#include "BasisState.h"

Mat get_factorized_state(const FactorizedSpace& space,
                         const Vec& state) {
    // First, create the factorized state where the left/right
    // spaces follow the usual global ordering (each within
    // its respective space).
    Mat unordered_state(space.left.D, space.right.D);

    // Left space is in the LSBs of the global state
    for (int i = 0; i < space.left.D; i++) {
        for (int j = 0; j < space.right.D; j++) {
            int global_state = i + (j << space.left.Nd);
            unordered_state(i, j) = state(global_state);
        }
    }

    // Now, reorder each axis to follow charge parity ordering
    Mat result(space.left.D, space.right.D);

    for (GlobalStateIterator left_iter(space.left);
         !left_iter.done();
         left_iter.next()) {

        for (GlobalStateIterator right_iter(space.right);
             !right_iter.done();
             right_iter.next()) {

            result(left_iter.parity_ordered_state,
                   right_iter.parity_ordered_state) =
                unordered_state(left_iter.global_state,
                                right_iter.global_state);
        }
    }

    return result;
}

Vec get_unfactorized_state(const FactorizedSpace& space,
                           const Mat& factorized_state) {
    assert(factorized_state.rows() == space.left.D);
    assert(factorized_state.cols() == space.right.D);

    // First, create the factorized state where the left/right
    // spaces follow the usual global ordering (each within
    // its respective space).
    Mat unordered_state(space.left.D, space.right.D);

    for (GlobalStateIterator left_iter(space.left);
         !left_iter.done();
         left_iter.next()) {

        for (GlobalStateIterator right_iter(space.right);
             !right_iter.done();
             right_iter.next()) {
            
            unordered_state(left_iter.global_state,
                            right_iter.global_state) = 
                factorized_state(left_iter.parity_ordered_state,
                                 right_iter.parity_ordered_state);
        }
    }

    Vec result(space.D);

    // Now, flatten the matrix to the vector
    // (Left space is in the LSBs of the global state)
    for (int i = 0; i < space.left.D; i++) {
        for (int j = 0; j < space.right.D; j++) {
            int global_state = i + (j << space.left.Nd);
            result(global_state) = unordered_state(i, j);
        }
    }

    return result;
}

FactorizedSpace::FactorizedSpace(const FactorizedSpace& other) :
    Space(other), left(other.left), right(other.right) {}

FactorizedSpace& FactorizedSpace::operator=(const FactorizedSpace& other) {
    Space& parent = *this;
    parent = (Space&) other;
    
    left = other.left;
    right = other.right;

    return *this;
}

FactorizedSpace FactorizedSpace::from_majorana(int _N) {
    assert(_N % 2 == 0);

    FactorizedSpace space;
    space.fact_init_dirac(_N / 2);
    return space;
}

FactorizedSpace FactorizedSpace::from_majorana(int _N, int _N_left) {
    assert(_N % 2 == 0);
    assert(_N_left % 2 == 0);

    FactorizedSpace space;
    space.fact_init_dirac(_N / 2, _N_left / 2);
    return space;
}

FactorizedSpace FactorizedSpace::from_dirac(int _Nd) {
    FactorizedSpace space;
    space.fact_init_dirac(_Nd);
    return space;
}

FactorizedSpace FactorizedSpace::from_dirac(int _Nd, int _Nd_left) {
    FactorizedSpace space;
    space.fact_init_dirac(_Nd, _Nd_left);
    return space;
}

FactorizedSpace::FactorizedSpace() {}

void FactorizedSpace::fact_init_dirac(int _Nd) {
    fact_init_dirac(_Nd, _Nd / 2);
}

void FactorizedSpace::fact_init_dirac(int _Nd, int _Nd_left) {
    init_dirac(_Nd);

    int Nd_left = _Nd_left;
    int Nd_right = Nd - Nd_left;
    assert(Nd_left + Nd_right == Nd);

    left = Space::from_dirac(Nd_left);
    right = Space::from_dirac(Nd_right);
}

int FactorizedSpace::state_size() {
    return 2 * state_block_size();
}

size_t FactorizedSpace::state_alloc_size() {
    return sizeof(cpx) * state_size();
}

int FactorizedSpace::state_block_size() {
    return state_block_rows() * state_block_cols();
}

int FactorizedSpace::state_block_rows() {
    return left.D / 2;
}

int FactorizedSpace::state_block_cols() {
    return right.D / 2;
}

int FactorizedSpace::left_block_rows() {
    return left.D / 2;
}

int FactorizedSpace::left_block_cols() {
    return left.D / 2;
}

int FactorizedSpace::left_block_size() {
    return left_block_rows() * left_block_cols();
}

int FactorizedSpace::right_block_rows() {
    return right.D / 2;
}

int FactorizedSpace::right_block_cols() {
    return right.D / 2;
}

int FactorizedSpace::right_block_size() {
    return right_block_rows() * right_block_cols();
}

FactorizedParityState::FactorizedParityState(FactorizedSpace& _space,
                                 ChargeParity _charge_parity) {
    space = _space;
    charge_parity = _charge_parity;
    matrix = Mat::Zero(space.left.D, space.right.D);
}

FactorizedParityState::FactorizedParityState(FactorizedSpace& _space,
                                 ChargeParity _charge_parity,
                                 boost::random::mt19937* gen) {
    space = _space;
    charge_parity = _charge_parity;
    matrix = get_factorized_random_state(space, charge_parity, gen);
}

FactorizedParityState::FactorizedParityState(
    const FactorizedParityState& other) {

    space = other.space;
    charge_parity = other.charge_parity;
    matrix = other.matrix;
}

FactorizedParityState& FactorizedParityState::operator= (
    const FactorizedParityState& other) {

    space = other.space;
    charge_parity = other.charge_parity;
    matrix = other.matrix;
    return *this;
}

int FactorizedParityState::size() {
    return space.left.D * space.right.D / 2;
}

Mat get_factorized_random_state(const FactorizedSpace& space,
                                boost::random::mt19937* gen) {
    if (gen == 0) {
        return Mat::Random(space.left.D, space.right.D);
    }
    
    Mat state = Mat(space.left.D, space.right.D);
    boost::random::normal_distribution<> dist(0., 1.);

    for (int i = 0; i < space.left.D; i++) {
        for (int j = 0; j < space.right.D; j++) {
            state(i,j) = cpx(dist(*gen), dist(*gen));
        }
    }

    state = state / state.norm();

    return state;
}

Mat get_factorized_random_state(const FactorizedSpace& space,
                                ChargeParity charge_parity,
                                boost::random::mt19937* gen) {

    Mat state = get_factorized_random_state(space, gen);
        
    if (charge_parity == EVEN_CHARGE) {
        // Even parity: matrix should be block-diagonal, so set off-diagonal
        // blocks to zero.
        state.block(0, space.right.D/2,
                    space.left.D/2, space.right.D/2).setZero();
        state.block(space.left.D/2, 0,
                    space.left.D/2, space.right.D/2).setZero();
    }
    else {
        // Odd parity: matrix should be block-off-diagonal
        state.block(0, 0,
                    space.left.D/2, space.right.D/2).setZero();
        state.block(space.left.D/2, space.right.D/2,
                    space.left.D/2, space.right.D/2).setZero();
    }

    state = state / state.norm();

    return state;
}

Vec get_random_state(const Space& space, boost::random::mt19937* gen) {
    Vec state = get_random_vector(space.D, gen);
    state = state / state.norm();
    return state;
}

