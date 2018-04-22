/////////////////////////////////////////////////////////////////////
//
// Majorana Hamiltonian in factorized space.
// 
/////////////////////////////////////////////////////////////////////

#ifndef __FACTORIZED_HAMILTONIAN_H__
#define __FACTORIZED_HAMILTONIAN_H__

#include <vector>
#include "FactorizedSpaceUtils.h"

class FactorizedOperatorPair {
public:
    FactorizedOperatorPair();

    FactorizedOperatorPair(const Mat& _O_left,
                           const Mat& _O_right,
                           ChargeParity _charge_parity);

    FactorizedOperatorPair(const FactorizedOperatorPair& other);
                           
    Mat O_left;
    Mat O_right;
    ChargeParity charge_parity;
};

class HamiltonianTermProcessor {
public:
    HamiltonianTermProcessor();
    virtual ~HamiltonianTermProcessor();

    // Process a pair of O_L, O_R operators with left_idx indices
    // in the left space.
    virtual void process(int left_idx, FactorizedOperatorPair&) = 0;
};

//
// Generates terms one by one and sends them to a callback. This way we
// can process a Hamiltonian without storing multiple copies of it in
// memory.
// 
void generate_factorized_hamiltonian_terms(
    FactorizedSpace& space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian,
    HamiltonianTermProcessor& callback);

// Return the number of operator pairs with left_idx left-space indices.
int num_factorized_hamiltonian_operator_pairs(
    FactorizedSpace& space, int left_idx);

//
// Majorana Hamiltonian in the factorized representation.
// No special handling of charge parity.
// 
class FactorizedHamiltonianGenericParity : public HamiltonianTermProcessor {
public:
    FactorizedHamiltonianGenericParity(
        FactorizedSpace _space,
        MajoranaDisorderParameter& Jtensor,
        bool mock_hamiltonian = false);

    virtual ~FactorizedHamiltonianGenericParity();

    // output += H * state
    void act(Mat& output, const Mat& state);

    FactorizedSpace space;

    // First vector is indexed by the number of left indices.
    // e.g. operators[0] has the left operators with 4 left indices,
    // and the right operators with 0 right indices.
    vector< vector<FactorizedOperatorPair> > operators;

    // Used when initializing the terms
    virtual void process(int left_idx, FactorizedOperatorPair&);

private:
    void init_operators(MajoranaDisorderParameter& Jtensor,
                        bool mock_hamiltonian);
};

typedef enum {
    BLOCK_DIAGONAL, BLOCK_OFF_DIAGONAL
} BlockShape;

// An operator that is either 2x2 block diagonal or block off diagonal.
class BlockOperator2x2 {
public:
    /* BlockOperator2x2(); */
    BlockOperator2x2(const Mat& op, BlockShape shape);
    BlockOperator2x2(const BlockOperator2x2& other);
    BlockOperator2x2& operator= (const BlockOperator2x2& other);
    
    // If block diagonal, top-left block.
    // If block off-diagonal, top-right block.
    Mat top_block;

    // If block diagonal, bottom-right block.
    // If block off-diagonal, bottom-left block.
    Mat bottom_block;

    BlockShape shape;
};

BlockShape get_block_shape(int left_idx);

//
// A Hamiltonian optimized to take advantage of charge parity
// conservation. Stores the operators in separate blocks.
//
class FactorizedHamiltonian {
public:
    FactorizedHamiltonian(
        FactorizedSpace _space,
        MajoranaDisorderParameter& Jtensor,
        bool mock_hamiltonian = false);

    // output += H * state
    // state has even charge parity
    void act_even(Mat& output, const Mat& state);

    // output += H * state
    // state has odd charge parity
    void act_odd(Mat& output, const Mat& state);

    // Individual block version
    void act_even(Mat& output_tl,
                  Mat& output_br,
                  const Mat& state_tl,
                  const Mat& state_br);

    // Individual block version
    void act_odd(Mat& output_tr,
                 Mat& output_bl,
                 const Mat& state_tr,
                 const Mat& state_bl);

    // First vector is indexed by the number of left indices.
    // e.g. left_operators[0] has the operators with 0 left indices.
    // right_operators[0] has the operators with 4 right indices.
    vector< vector<BlockOperator2x2> > left_operators;
    vector< vector<BlockOperator2x2> > right_operators;

    FactorizedSpace space;
    int state_block_alloc_size; // memory a single state block takes

private:
    void init_operators(MajoranaDisorderParameter& Jtensor);
};

// An operator that is either 2x2 block diagonal or block off diagonal.
// Uses sparse matrices.
class SparseBlockOperator2x2 {
public:
    /* SparseBlockOperator2x2(); */
    SparseBlockOperator2x2(const Mat& op, BlockShape shape);
    
    // If block diagonal, top-left block.
    // If block off-diagonal, top-right block.
    SpMat top_block;

    // If block diagonal, bottom-right block.
    // If block off-diagonal, bottom-left block.
    SpMat bottom_block;

    BlockShape shape;
};

//
// Same as FactorizedHamiltonian but uses sparse matrices for the
// operators.
//
class SparseFactorizedHamiltonian {
public:
    SparseFactorizedHamiltonian(
        FactorizedSpace _space,
        MajoranaDisorderParameter& Jtensor,
        bool mock_hamiltonian = false);

    // output += H * state
    // state has even charge parity
    void act_even(Mat& output, const Mat& state);

    // output += H * state
    // state has odd charge parity
    void act_odd(Mat& output, const Mat& state);

    // Individual block version
    void act_even(Mat& output_tl,
                  Mat& output_br,
                  const Mat& state_tl,
                  const Mat& state_br);

    // Individual block version
    void act_odd(Mat& output_tr,
                 Mat& output_bl,
                 const Mat& state_tr,
                 const Mat& state_bl);

    // First vector is indexed by the number of left/right indices.
    // e.g. left_operators[0] has the operators with 0 left indices.
    // right_operators[3] has the operators with 3 right indices.
    vector< vector<SparseBlockOperator2x2> > left_operators;
    vector< vector<SparseBlockOperator2x2> > right_operators;

    FactorizedSpace space;
    int state_block_alloc_size; // memory a single state block takes

private:
    void init_operators(MajoranaDisorderParameter& Jtensor);
};

//
// Same as FactorizedHamiltonian but uses some sparse matrices for the
// operators.
//
class HalfSparseFactorizedHamiltonian {
public:
    HalfSparseFactorizedHamiltonian(
        FactorizedSpace _space,
        MajoranaDisorderParameter& Jtensor,
        bool mock_hamiltonian = false);

    // output += H * state
    // state has even charge parity
    void act_even(Mat& output, const Mat& state);

    // output += H * state
    // state has odd charge parity
    void act_odd(Mat& output, const Mat& state);

    // Individual block version
    void act_even(Mat& output_tl,
                  Mat& output_br,
                  const Mat& state_tl,
                  const Mat& state_br);

    // Individual block version
    void act_odd(Mat& output_tr,
                 Mat& output_bl,
                 const Mat& state_tr,
                 const Mat& state_bl);

    // Maps are indexed by the number of indices (left indices for left,
    // right indices for right).
    // e.g. left_X_operators[0] has the operators with 0 left indices.
    // right_X_operators[1] has the operators with 1 right index.
    map< int, vector<SparseBlockOperator2x2> > left_sparse_operators;
    map< int, vector<BlockOperator2x2> > left_dense_operators;

    map< int, vector<SparseBlockOperator2x2> > right_sparse_operators;
    map< int, vector<BlockOperator2x2> > right_dense_operators;

    FactorizedSpace space;
    int state_block_alloc_size; // memory a single state block takes

private:
    void init_operators(MajoranaDisorderParameter& Jtensor);
    BlockShape get_shape(int num_left_indices);
    void add_left_sparse_right_dense(
        FactorizedHamiltonianGenericParity& H, int num_left_indices);
    void add_left_dense_right_sparse(
        FactorizedHamiltonianGenericParity& H, int num_left_indices);
};

Mat psi_in_charge_parity_ordering(const Space& space,
                                  const Mat& psi_fock_ordering);

vector<SpMat> compute_psi_matrices(const Space& space);

#endif // __FACTORIZED_HAMILTONIAN_H__
