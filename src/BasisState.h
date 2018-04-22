#ifndef BASIS_STATE_H__ 
#define BASIS_STATE_H__

#include <list>
#include <string>
#include "defs.h"
#include "FockSpaceUtils.h"

// Convert state number (according to the combinatorial number system)
// to a set of indices denoting the fermion occupations at charge Q
list<int> state_number_to_occupations(int state_number, int Q);

// Convert the given fermion occupation indices to a state number
int occupations_to_state_number(const list<int>& indices);

/*
 * A Fock space state with an integer coefficient.
 * The convention is:
 *
 * |i_1,...,i_Q> = c^\dagger_{i_1} ... c^\dagger_{i_Q} |0>
 * 0 <= i_1 < ... < i_Q
 *
 * Note that indices are 0-based.
 */
class BasisState {
public:
    // Constructs the object with the vacuum state
    BasisState();

    // Constructs the object with the given indices
    BasisState(list<int> _indices, int _coefficient = 1);

    // Constructs the object with the given indices
    BasisState(int* _indices, int _num_indices, int _coefficient = 1);

    // Constructs the object from the given state number and charge 
    // (Q = number of creation operators). See:
    // https://en.wikipedia.org/wiki/Combinatorial_number_system
    BasisState(int state_number, int Q);

    // Consructs the object from the given state number, which labels
    // states in the full Hilbert space.
    BasisState(int global_state_number);

    // Act with c_i
    void annihilate(int i);

    // Act with c^\dagger_i
    void create(int i);

    // Returns the number that corresponds to this Fock space with charge Q.
    // The number is unique to each state without gaps, and allows 
    // ordering of states within each Q sector. See:
    // https://en.wikipedia.org/wiki/Combinatorial_number_system
    int get_state_number();

    // Returns the state number over the whole Hilbert space (all charges).
    int get_global_state_number();

    // Returns the charge of this state, i.e. the number of creation
    // operators
    int charge();

    // Whether this is a zero state
    bool is_zero();

    // Returns the next state in the get_state_number() ordering
    BasisState next_state();

    BasisState& operator = (const BasisState& other);
    bool operator == (const BasisState& other);
    bool operator != (const BasisState& other);

    string to_string();

    int find_maximal_ck(int state_number, int k);

    list<int> indices;
    int coefficient;
};

//
// Iterates over global states, and for each state computes a state
// number where even charge states appear before odd charge ones.
// 
class GlobalStateIterator {
public:
    GlobalStateIterator(const Space& space);

    bool done();
    void next();

    int global_state;
    int parity_ordered_state;

private:
    Space space;
    map<int, int> seen_states_by_charge;
};

#endif // BASIS_STATE_H__
