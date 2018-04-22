#include <sstream>
#include <iostream>
#include <assert.h>
#include "BasisState.h"
#include "FockSpaceUtils.h"

// Constructs the object with the vacuum state
BasisState::BasisState() {
    coefficient = 1;
}

BasisState::BasisState(list<int> _indices, int _coefficient) :
    indices(_indices), coefficient(_coefficient) {}

BasisState::BasisState(
        int* _indices, int _num_indices, int _coefficient) {
    indices = list<int>(_indices, _indices + _num_indices);
    coefficient = _coefficient;
}

BasisState::BasisState(int state_number, int Q) : coefficient(1) {
    indices = state_number_to_occupations(state_number, Q);
}

BasisState::BasisState(int global_state_number) : coefficient(1) {
    assert(global_state_number >= 0);
    int i = 0;

    while (global_state_number != 0) {
        int bit = global_state_number % 2;
        global_state_number /= 2;

        if (bit) {
            indices.push_back(i);
        }

        i++;
    }
}

// Act with c_i
void BasisState::annihilate(int i) {
    if (coefficient == 0) {
        return;
    }

    bool index_found = false;
    list<int>::iterator iter = indices.begin();

    while (iter != indices.end()) {
        if (*iter == i) {
            // Found the position
            index_found = true;
            break;
        }
        else if (*iter < i) {
            // We need to drag c_i across this, so we flip the sign
            coefficient *= -1;
        }
        else {
            // No need to go any further, index not found in state
            // The index i doesn't appear in the state, so c_i kills it
            break;
        }

        ++iter;
    }

    if (index_found) {
        indices.erase(iter);
    }
    else {
        coefficient = 0;
    }
}

// Act with c^\dagger_i
void BasisState::create(int i) {
    if (coefficient == 0) {
        return;
    }

    list<int>::iterator iter = indices.begin();

    while (iter != indices.end()) {
        if (*iter == i) {
            // Acting twice with same creation operator kills the state
            coefficient = 0;
            return;
        }
        else if (*iter < i) {
            // We need to drag c^\dagger_i across this, so we flip the sign
            coefficient *= -1;
        }
        else {
            // We found the position
            break;
        }

        ++iter;
    }

    indices.insert(iter, i);
}

int BasisState::get_state_number() {
    return occupations_to_state_number(indices);
}

int BasisState::get_global_state_number() {
    int num = 0;

    for (list<int>::iterator iter = indices.begin();
         iter != indices.end();
         ++iter) {
        num += pow(2, *iter);
    }

    return num;
}

int BasisState::charge() {
    assert(coefficient != 0);
    return indices.size();
}

bool BasisState::is_zero() {
    return coefficient == 0;
}

BasisState BasisState::next_state() {
    return BasisState(
            get_state_number() + 1,
            charge()
            );
}

BasisState& BasisState::operator = (const BasisState& other) {
    if (this != &other) {
        indices = other.indices;
        coefficient = other.coefficient;
    }

    return *this;
}

bool BasisState::operator == (const BasisState& other) {
    return (indices == other.indices) && (coefficient == other.coefficient);
}

bool BasisState::operator != (const BasisState& other) {
    return !((*this) == other);
}

string BasisState::to_string() {
    if (coefficient == 0) {
        return "0";
    }
    else {
        stringstream s;

        if (coefficient == -1) {
            s << "-";
        }
        else if (coefficient != 1) {
            s << coefficient << "*";
        }

        s << "|";

        list<int>::iterator iter = indices.begin();
        while (iter != indices.end()) {
            s << *iter;
            ++iter;
            if (iter != indices.end()) {
                s << ",";
            }
        }

        s << ">";
        return s.str();
    }
}

// Find maximal c_k such that (c_k choose k) <= N
static int find_maximal_ck(int state_number, int k) {
    int ck = 0;

    while (binomial(ck, k) <= state_number) {
        ck++;
    }

    return ck - 1;
}

list<int> state_number_to_occupations(int state_number, int Q) {
    list<int> indices;
    int k = Q;

    while (k > 0) {
        int ck = find_maximal_ck(state_number, k);
        indices.push_front(ck); // we all use 0-based indices
        state_number -= binomial(ck, k);
        k--;
    }

    assert(state_number == 0);
    return indices;
}

int occupations_to_state_number(const list<int>& indices) {
    list<int>::const_iterator iter = indices.begin();
    int k = 1;
    int result = 0;

    while (iter != indices.end()) {
        // The formula uses 0-based indices, just like us
        int i = *iter;
        result += binomial(i, k);
        ++iter;
        k++;
    }

    return result;
}

GlobalStateIterator::GlobalStateIterator(const Space& _space) {
    space = _space;
    global_state = 0;
    parity_ordered_state = 0;

    for (int Q = 0; Q <= space.Nd; Q++) {
        seen_states_by_charge[Q] = 0;
    }
}

bool GlobalStateIterator::done() {
    return global_state >= space.D;
}

void GlobalStateIterator::next() {
    if (done()) return;
    global_state++;
    if (done()) return;

    int state_Q = __builtin_popcount(global_state);

    // Offset within charge sector
    parity_ordered_state = seen_states_by_charge.at(state_Q);

    if (state_Q % 2 == 0) {
        // Add all previous even sectors
        for (int Q = 0; Q < state_Q; Q += 2) {
            parity_ordered_state += Q_sector_dim(space.Nd, Q);
        }
    }
    else {
        // For odd charges, add all even sectors, and all previous
        // odd sectors
        for (int Q = 0; Q <= space.Nd; Q += 2) {
            parity_ordered_state += Q_sector_dim(space.Nd, Q);
        }

        for (int Q = 1; Q < state_Q; Q += 2) {
            parity_ordered_state += Q_sector_dim(space.Nd, Q);
        }
    }

    assert(parity_ordered_state >= 0);
    assert(parity_ordered_state < space.D);

    seen_states_by_charge.at(state_Q)++;
}
