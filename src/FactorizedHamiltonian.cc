#include <iostream>
#include "FactorizedHamiltonian.h"
#include "FockSpaceUtils.h"
#include "BasisState.h"
#include "Timer.h"

//typedef pair<Mat,Mat> Mat_pair;
typedef pair<int,int> int_pair;

typedef enum {
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT
} BlockPosition;

// Returns the requested block in a matrix
static Mat get_block(const Mat& mat, BlockPosition pos) {
    assert(mat.rows() % 2 == 0);
    assert(mat.cols() % 2 == 0);
    int Drow = mat.rows() / 2;
    int Dcol = mat.cols() / 2;
    
    switch (pos) {
    case TOP_LEFT:     return mat.block(0,    0,    Drow, Dcol);
    case TOP_RIGHT:    return mat.block(0,    Dcol, Drow, Dcol);
    case BOTTOM_LEFT:  return mat.block(Drow, 0,    Drow, Dcol);
    case BOTTOM_RIGHT: return mat.block(Drow, Dcol, Drow, Dcol);
    }

    cerr << "Unreachable code" << endl;
    exit(1);
}

Mat psi_in_charge_parity_ordering(const Space& space,
                                  const Mat& psi_fock_ordering) {
    Mat psi = Mat::Zero(space.D, space.D);

    for (GlobalStateIterator row_iter(space);
         !row_iter.done();
         row_iter.next()) {

        for (GlobalStateIterator col_iter(space);
            !col_iter.done();
            col_iter.next()) {

            psi(row_iter.parity_ordered_state,
                col_iter.parity_ordered_state) =
                psi_fock_ordering(row_iter.global_state,
                                  col_iter.global_state);
        }
    }

    return psi;
}

// Compute sparse \psi_a matrices for the given Majorana N value,
// in charge parity ordering.
vector<SpMat> compute_psi_matrices(const Space& space,
                                   bool mock) {
    vector<SpMat> orig_psis = compute_chi_matrices(space.N);
    assert(orig_psis.size() == (size_t) space.N);

    vector<SpMat> psis;

    for (size_t a = 0; a < orig_psis.size(); a++) {
        Mat psi_fock_ordering = Mat(orig_psis[a]);
        Mat psi = psi_in_charge_parity_ordering(space,
                                                psi_fock_ordering);
        psis.push_back(psi.sparseView());
    }

    assert(psis.size() == (size_t) space.N);
    return psis;
}

static void compute_psi_prods(const Space& space,
                              vector<SpMat>& psi,
                              map<int_pair, SpMat>& psi_prods) {
    for (int i = 0; i < space.N; i++) {
        for (int j = i+1; j < space.N; j++) {
            psi_prods[int_pair(i,j)] = psi[i] * psi[j];
        }
    }

    assert(psi_prods.size() == (size_t) binomial(space.N, 2));
}


// Operators acting on the left/right part of Hilbert space
class FactorizedPsiData {
public:
    // _Nd = number of Dirac fermions on this space
    FactorizedPsiData(const Space& _space, bool mock=false)
        : space(_space) {
        psi = compute_psi_matrices(space, mock);
        compute_psi_prods(space, psi, psi_prods);
    }

    // psi_i * psi_j
    SpMat& psi_prod(int i, int j) {
        return psi_prods[int_pair(i,j)];
    }

    // psi_i * psi_j * psi_k
    SpMat psi_prod(int i, int j, int k) {
        return psi_prods[int_pair(i,j)] * psi[k];
    }

    // psi_i * psi_j * psi_k * psi_l
    SpMat psi_prod(int i, int j, int k, int l) {
        return psi_prods[int_pair(i,j)] * psi_prods[int_pair(k,l)];
    }

    // zero matrix for this space
    Mat zero() {
        return Mat::Zero(space.D, space.D);
    }

    // identity matrix for this space
    Mat identity() {
        return Mat::Identity(space.D, space.D);
    }

    Space space;
    vector<SpMat> psi; // psi_i operators
    map<int_pair, SpMat> psi_prods; // psi_i * psi_j operators, i<j
};

FactorizedOperatorPair::FactorizedOperatorPair(const Mat& _O_left,
                                               const Mat& _O_right,
                                               ChargeParity _charge_parity) :
        O_left(_O_left),
        O_right(_O_right),
        charge_parity(_charge_parity) {}

FactorizedOperatorPair::FactorizedOperatorPair(
    const FactorizedOperatorPair& other) :
        O_left(other.O_left),
        O_right(other.O_right),
        charge_parity(other.charge_parity) {}

FactorizedOperatorPair::FactorizedOperatorPair() {}
                           
FactorizedHamiltonianGenericParity::FactorizedHamiltonianGenericParity(
    FactorizedSpace _space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian) {
    space = _space;
    init_operators(Jtensor, mock_hamiltonian);
}

// Returns the matrix that needs to be multiplied against the factorized
// state to flip the sign of states with odd charge parity.
static Mat get_odd_parity_sign_flip_matrix(const Space& left_space) {
    Mat flipper = Mat::Zero(left_space.D, left_space.D);

    for (int i = 0; i < left_space.D/2; i++) {
        flipper(i,i) = 1.;
    }

    for (int i = left_space.D/2; i < left_space.D; i++) {
        flipper(i,i) = -1.;
    }

    return flipper;
}

HamiltonianTermProcessor::HamiltonianTermProcessor() {}
HamiltonianTermProcessor::~HamiltonianTermProcessor() {}

void generate_factorized_hamiltonian_terms(
    FactorizedSpace& space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian,
    HamiltonianTermProcessor& callback) {

    FactorizedPsiData left(space.left, mock_hamiltonian);
    FactorizedPsiData right(space.right, mock_hamiltonian);

    Mat sign_flip_left = get_odd_parity_sign_flip_matrix(space.left);

    // Go over possibilities for \psi_i \psi_j \psi_k \psi_l :
    // which of i<j<k<l refer to left/right spaces.
    // We can sum over one of the sides, and we choose the side with
    // the most elements.

    //// 0 left indices, 4 right indices ////
    Mat psi4_right = right.zero();

    if (!mock_hamiltonian) {
        for (int i = 0; i < right.space.N; i++) {
            for (int j = i+1; j < right.space.N; j++) {
                for (int k = j+1; k < right.space.N; k++) {
                    for (int l = k+1; l < right.space.N; l++) {
                        psi4_right +=
                            Jtensor.elem(i + left.space.N,
                                         j + left.space.N,
                                         k + left.space.N,
                                         l + left.space.N) *
                            right.psi_prod(i,j,k,l);
                    }
                }
            }
        }
    }

    FactorizedOperatorPair pair0(left.identity(),
                                 psi4_right.transpose(),
                                 EVEN_CHARGE);
    callback.process(0, pair0);

    //// 1 left, 3 right ////
    for (int i = 0; i < left.space.N; i++) {
        Mat psi3_right = right.zero();

        if (!mock_hamiltonian) {
            for (int j = 0; j < right.space.N; j++) {
                for (int k = j+1; k < right.space.N; k++) {
                    for (int l = k+1; l < right.space.N; l++) {
                        psi3_right += 
                            Jtensor.elem(i,
                                         j + left.space.N,
                                         k + left.space.N,
                                         l + left.space.N) *
                            right.psi_prod(j,k,l);
                    }
                }
            }
        }

        // When the right matrix has odd parity, we need to include
        // the effect of anti-commuting it through the left fermions.
        // This incurs an extra minus sign for states with an odd
        // number of fermions on the left.
        FactorizedOperatorPair pair1(left.psi[i] * sign_flip_left,
                                     psi3_right.transpose(),
                                     ODD_CHARGE);
        callback.process(1, pair1);
    }

    //// 2 left, 2 right ////
    // sum over the right, because Nr > Nl when Nd is odd.
    for (int i = 0; i < left.space.N; i++) {
        for (int j = i+1; j < left.space.N; j++) {
            Mat psi2_right = right.zero();

            if (!mock_hamiltonian) {
                for (int k = 0; k < right.space.N; k++) {
                    for (int l = k+1; l < right.space.N; l++) {
                        psi2_right +=
                            Jtensor.elem(i,
                                         j,
                                         k + left.space.N,
                                         l + left.space.N) *
                            right.psi_prod(k,l);
                    }
                }
            }

            FactorizedOperatorPair pair2(left.psi_prod(i,j),
                                         psi2_right.transpose(),
                                         EVEN_CHARGE);
            callback.process(2, pair2);
        }
    }

    //// 3 indices on the left, 1 on the right ////
    for (int l = 0; l < right.space.N; l++) {
        Mat psi3_left = left.zero();

        if (!mock_hamiltonian) {
            for (int i = 0; i < left.space.N; i++) {
                for (int j = i+1; j < left.space.N; j++) {
                    for (int k = j+1; k < left.space.N; k++) {
                        psi3_left +=
                            Jtensor.elem(i, j, k, l + left.space.N) *
                            left.psi_prod(i,j,k);
                    }
                }
            }
        }

        // When the right matrix has odd parity, we need to include
        // the effect of anti-commuting it through the left fermions.
        // This incurs an extra minus sign for states with an odd
        // number of fermions on the left.
        FactorizedOperatorPair pair3(psi3_left * sign_flip_left,
                                     right.psi[l].transpose(),
                                     ODD_CHARGE);
        callback.process(3, pair3);
    }

    //// All 4 indices on the left (identity on the right) ////
    Mat psi4_left = left.zero();

    if (!mock_hamiltonian) {
        for (int i = 0; i < left.space.N; i++) {
            for (int j = i+1; j < left.space.N; j++) {
                for (int k = j+1; k < left.space.N; k++) {
                    for (int l = k+1; l < left.space.N; l++) {
                        psi4_left +=
                            Jtensor.elem(i,j,k,l) *
                            left.psi_prod(i,j,k,l);
                    }
                }
            }
        }
    }

    FactorizedOperatorPair pair4(psi4_left, right.identity(), EVEN_CHARGE);
    callback.process(4, pair4);
}

int num_factorized_hamiltonian_operator_pairs(FactorizedSpace& space,
                                              int left_idx) {
    if (left_idx <= q/2) {
        return binomial(space.left.N, left_idx);
    }
    else {
        return binomial(space.right.N, q-left_idx);
    }
}

FactorizedHamiltonianGenericParity::~FactorizedHamiltonianGenericParity() {}

void FactorizedHamiltonianGenericParity::init_operators(
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian) {

    // Initialize some empty vectors to hold the operators
    for (int i = 0; i <= q; i++) {
        operators.push_back(vector<FactorizedOperatorPair>());
    }

    generate_factorized_hamiltonian_terms(
        space, Jtensor, mock_hamiltonian, *this);
}

void FactorizedHamiltonianGenericParity::process(
    int left_idx, FactorizedOperatorPair& ops) {

    operators.at(left_idx).push_back(FactorizedOperatorPair(ops));
}

void FactorizedHamiltonianGenericParity::act(
    Mat& output, const Mat& state) {

    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    for (int left_ind = 0; left_ind < operators.size(); left_ind++) {
        for (int n = 0; n < operators[left_ind].size(); n++) {
            // The O_R transpose was taken when it was created
            FactorizedOperatorPair& op_pair = operators[left_ind][n];
            output += op_pair.O_left * state * op_pair.O_right;
        }
    }
}

BlockOperator2x2::BlockOperator2x2(const Mat& op, BlockShape _shape) {
    shape = _shape;

    if (shape == BLOCK_DIAGONAL) {
        top_block = get_block(op, TOP_LEFT);
        bottom_block = get_block(op, BOTTOM_RIGHT);
    }
    else {
        top_block = get_block(op, TOP_RIGHT);
        bottom_block = get_block(op, BOTTOM_LEFT);
    }
}

BlockOperator2x2::BlockOperator2x2(const BlockOperator2x2& other) {
    top_block = other.top_block;
    bottom_block = other.bottom_block;
    shape = other.shape;
}

BlockOperator2x2& BlockOperator2x2::operator= (
    const BlockOperator2x2& other) {

    top_block = other.top_block;
    bottom_block = other.bottom_block;
    shape = other.shape;
    return *this;
}

BlockShape get_block_shape(int left_idx) {
    return (left_idx % 2 == 0) ? BLOCK_DIAGONAL : BLOCK_OFF_DIAGONAL;
}

FactorizedHamiltonian::FactorizedHamiltonian(
    FactorizedSpace _space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian) {

    space = _space;

    state_block_alloc_size =
        sizeof(cpx) * (space.left.D / 2) * (space.right.D / 2);
    
    FactorizedHamiltonianGenericParity H(space,
                                         Jtensor,
                                         mock_hamiltonian);

    for (size_t left_ind = 0; left_ind < H.operators.size(); left_ind++) {
        left_operators.push_back(vector<BlockOperator2x2>());
        right_operators.push_back(vector<BlockOperator2x2>());
    }

    for (size_t left_ind = 0; left_ind < H.operators.size(); left_ind++) {
        BlockShape shape = get_block_shape(left_ind);

        for (size_t n = 0; n < H.operators.at(left_ind).size(); n++) {
            FactorizedOperatorPair& op_pair = H.operators.at(left_ind).at(n);

            assert(left_operators.size() > left_ind);
            assert(right_operators.size() > q - left_ind);

            left_operators.at(left_ind).push_back(
                BlockOperator2x2(op_pair.O_left, shape)); 

            right_operators.at(q - left_ind).push_back(
                BlockOperator2x2(op_pair.O_right, shape)); 
        }
    }
}

void FactorizedHamiltonian::act_even(Mat& output_tl,
                                     Mat& output_br,
                                     const Mat& state_tl,
                                     const Mat& state_br) {
    assert(left_operators.size() == q+1);
    assert(right_operators.size() == q+1);

    for (size_t left_ind = 0;
         left_ind < left_operators.size();
         left_ind++) {

        assert(right_operators[q-left_ind].size() ==
               left_operators[left_ind].size());

        for (size_t n = 0; n < left_operators[left_ind].size(); n++) {
            BlockOperator2x2& O_L = left_operators[left_ind][n];
            BlockOperator2x2& O_R = right_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   ( tl  0  )   ( Rtop 0    )    
                // (    0 Lbot ) . (  0  br ) . (    0 Rbot ) = 
                // 
                //    ( Ltop.tl.Rtop  0            )
                //    (            0  Lbot.br.Rbot )

                output_tl += O_L.top_block    *state_tl * O_R.top_block;
                output_br += O_L.bottom_block *state_br *O_R.bottom_block;
            }
            else {
                // Odd operators
                //
                // ( 0     Ltop )   ( tl  0  )   ( 0    Rtop )   
                // ( Lbot  0    ) . (  0  br ) . ( Rbot 0    ) = 
                // 
                //    ( Ltop.br.Rbot  0            )
                //    (            0  Lbot.tl.Rtop )

                output_tl += O_L.top_block * state_br * O_R.bottom_block;
                output_br += O_L.bottom_block * state_tl * O_R.top_block;
            }
        }
    }
}

void FactorizedHamiltonian::act_even(Mat& output, const Mat& state) {
    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tl = get_block(state, TOP_LEFT);
    Mat state_br = get_block(state, BOTTOM_RIGHT);

    Mat output_tl = Mat::Zero(state_tl.rows(), state_tl.cols());
    Mat output_br = Mat::Zero(state_br.rows(), state_br.cols());

    act_even(output_tl, output_br, state_tl, state_br);

    // Write the blocks to the output
    output.block(0,
                 0,
                 output_tl.rows(),
                 output_tl.cols()) += output_tl;

    output.block(output_tl.rows(),
                 output_tl.cols(),
                 output_br.rows(),
                 output_br.cols()) += output_br;
}

void FactorizedHamiltonian::act_odd(Mat& output_tr,
                                    Mat& output_bl,
                                    const Mat& state_tr,
                                    const Mat& state_bl) {
    for (size_t left_ind = 0;
         left_ind < left_operators.size();
         left_ind++) {

        for (size_t n = 0; n < left_operators[left_ind].size(); n++) {
            BlockOperator2x2& O_L = left_operators[left_ind][n];
            BlockOperator2x2& O_R = right_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   (  0  tr )   ( Rtop 0    )    
                // (    0 Lbot ) . ( bl   0 ) . (    0 Rbot ) = 
                // 
                //    (            0  Ltop.tr.Rbot )
                //    ( Lbot.bl.Rtop  0            )

                output_tr += O_L.top_block * state_tr * O_R.bottom_block;
                output_bl += O_L.bottom_block * state_bl * O_R.top_block;
            }
            else {
                //
                // ( 0     Ltop )   (  0  tr )   ( 0    Rtop )   
                // ( Lbot  0    ) . ( bl   0 ) . ( Rbot 0    ) = 
                // 
                //    (            0 Ltop.bl.Rtop )
                //    ( Lbot.tr.Rbot            0 )

                output_tr += O_L.top_block * state_bl * O_R.top_block;
                output_bl += O_L.bottom_block * state_tr *O_R.bottom_block;
            }
        }
    }
}

void FactorizedHamiltonian::act_odd(Mat& output, const Mat& state) {
    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tr = get_block(state, TOP_RIGHT);
    Mat state_bl = get_block(state, BOTTOM_LEFT);

    Mat output_tr = Mat::Zero(state_tr.rows(), state_tr.cols());
    Mat output_bl = Mat::Zero(state_bl.rows(), state_bl.cols());

    act_odd(output_tr, output_bl, state_tr, state_bl);

    // Write the blocks to the output
    output.block(0,
                 output_bl.cols(),
                 output_tr.rows(),
                 output_tr.cols()) += output_tr;

    output.block(output_tr.rows(),
                 0,
                 output_bl.rows(),
                 output_bl.cols()) += output_bl;
}

SparseBlockOperator2x2::SparseBlockOperator2x2(const Mat& op,
                                               BlockShape _shape) {
    shape = _shape;

    if (shape == BLOCK_DIAGONAL) {
        top_block = SpMat(get_block(op, TOP_LEFT).sparseView());
        bottom_block = SpMat(get_block(op, BOTTOM_RIGHT).sparseView());
    }
    else {
        top_block = SpMat(get_block(op, TOP_RIGHT).sparseView());
        bottom_block = SpMat(get_block(op, BOTTOM_LEFT).sparseView());
    }
}

SparseFactorizedHamiltonian::SparseFactorizedHamiltonian(
    FactorizedSpace _space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian) {

    space = _space;

    state_block_alloc_size =
        sizeof(cpx) * (space.left.D / 2) * (space.right.D / 2);
    
    FactorizedHamiltonianGenericParity H(space,
                                         Jtensor,
                                         mock_hamiltonian);

    for (size_t left_ind = 0; left_ind < H.operators.size(); left_ind++) {
        left_operators.push_back(vector<SparseBlockOperator2x2>());
        right_operators.push_back(vector<SparseBlockOperator2x2>());
    }

    for (size_t left_ind = 0; left_ind < H.operators.size(); left_ind++) {
        BlockShape shape =
            (left_ind % 2 == 0) ? BLOCK_DIAGONAL : BLOCK_OFF_DIAGONAL;

        for (size_t n = 0; n < H.operators[left_ind].size(); n++) {
            FactorizedOperatorPair& op_pair = H.operators[left_ind][n];

            left_operators[left_ind].push_back(
                SparseBlockOperator2x2(op_pair.O_left, shape)); 

            right_operators[q - left_ind].push_back(
                SparseBlockOperator2x2(op_pair.O_right, shape)); 
        }
    }
}

void SparseFactorizedHamiltonian::act_even(Mat& output_tl,
                                           Mat& output_br,
                                           const Mat& state_tl,
                                           const Mat& state_br) {
    for (size_t left_ind = 0;
         left_ind < left_operators.size();
         left_ind++) {

        for (size_t n = 0; n < left_operators[left_ind].size(); n++) {
            SparseBlockOperator2x2& O_L = left_operators[left_ind][n];
            SparseBlockOperator2x2& O_R = right_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   ( tl  0  )   ( Rtop 0    )    
                // (    0 Lbot ) . (  0  br ) . (    0 Rbot ) = 
                // 
                //    ( Ltop.tl.Rtop  0            )
                //    (            0  Lbot.br.Rbot )

                output_tl += O_L.top_block    * state_tl * O_R.top_block;
                output_br += O_L.bottom_block * state_br *O_R.bottom_block;
            }
            else {
                // Odd operators
                //
                // ( 0     Ltop )   ( tl  0  )   ( 0    Rtop )   
                // ( Lbot  0    ) . (  0  br ) . ( Rbot 0    ) = 
                // 
                //    ( Ltop.br.Rbot  0            )
                //    (            0  Lbot.tl.Rtop )

                output_tl += O_L.top_block * state_br * O_R.bottom_block;
                output_br += O_L.bottom_block * state_tl * O_R.top_block;
            }
        }
    }
}

void SparseFactorizedHamiltonian::act_even(Mat& output, const Mat& state) {
    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tl = get_block(state, TOP_LEFT);
    Mat state_br = get_block(state, BOTTOM_RIGHT);

    Mat output_tl = Mat::Zero(state_tl.rows(), state_tl.cols());
    Mat output_br = Mat::Zero(state_br.rows(), state_br.cols());

    act_even(output_tl, output_br, state_tl, state_br);

    // Write the blocks to the output
    output.block(0,
                 0,
                 output_tl.rows(),
                 output_tl.cols()) += output_tl;

    output.block(output_tl.rows(),
                 output_tl.cols(),
                 output_br.rows(),
                 output_br.cols()) += output_br;
}

void SparseFactorizedHamiltonian::act_odd(Mat& output_tr,
                                    Mat& output_bl,
                                    const Mat& state_tr,
                                    const Mat& state_bl) {
    for (size_t left_ind = 0;
         left_ind < left_operators.size();
         left_ind++) {

        for (size_t n = 0; n < left_operators[left_ind].size(); n++) {
            SparseBlockOperator2x2& O_L = left_operators[left_ind][n];
            SparseBlockOperator2x2& O_R = right_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   (  0  tr )   ( Rtop 0    )    
                // (    0 Lbot ) . ( bl   0 ) . (    0 Rbot ) = 
                // 
                //    (            0  Ltop.tr.Rbot )
                //    ( Lbot.bl.Rtop  0            )

                output_tr += O_L.top_block * state_tr * O_R.bottom_block;
                output_bl += O_L.bottom_block * state_bl * O_R.top_block;
            }
            else {
                //
                // ( 0     Ltop )   (  0  tr )   ( 0    Rtop )   
                // ( Lbot  0    ) . ( bl   0 ) . ( Rbot 0    ) = 
                // 
                //    (            0 Ltop.bl.Rtop )
                //    ( Lbot.tr.Rbot            0 )

                output_tr += O_L.top_block * state_bl * O_R.top_block;
                output_bl += O_L.bottom_block * state_tr *O_R.bottom_block;
            }
        }
    }
}

void SparseFactorizedHamiltonian::act_odd(Mat& output, const Mat& state) {
    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tr = get_block(state, TOP_RIGHT);
    Mat state_bl = get_block(state, BOTTOM_LEFT);

    Mat output_tr = Mat::Zero(state_tr.rows(), state_tr.cols());
    Mat output_bl = Mat::Zero(state_bl.rows(), state_bl.cols());

    act_odd(output_tr, output_bl, state_tr, state_bl);

    // Write the blocks to the output
    output.block(0,
                 output_bl.cols(),
                 output_tr.rows(),
                 output_tr.cols()) += output_tr;

    output.block(output_tr.rows(),
                 0,
                 output_bl.rows(),
                 output_bl.cols()) += output_bl;
}

BlockShape HalfSparseFactorizedHamiltonian::get_shape(
    int num_left_indices) {

    return (num_left_indices % 2 == 0) ?
        BLOCK_DIAGONAL : BLOCK_OFF_DIAGONAL;
}
    

void HalfSparseFactorizedHamiltonian::add_left_sparse_right_dense(
    FactorizedHamiltonianGenericParity& H, int num_left_indices) {

    for (size_t n = 0; n < H.operators[num_left_indices].size(); n++) {
        FactorizedOperatorPair& op_pair = H.operators[num_left_indices][n];

        left_sparse_operators[num_left_indices].push_back(
            SparseBlockOperator2x2(op_pair.O_left,
                                   get_shape(num_left_indices))); 

        right_dense_operators[q - num_left_indices].push_back(
            BlockOperator2x2(op_pair.O_right,
                             get_shape(num_left_indices))); 
    }
}

void HalfSparseFactorizedHamiltonian::add_left_dense_right_sparse(
    FactorizedHamiltonianGenericParity& H, int num_left_indices) {

    for (size_t n = 0; n < H.operators[num_left_indices].size(); n++) {
        FactorizedOperatorPair& op_pair = H.operators[num_left_indices][n];

        left_dense_operators[num_left_indices].push_back(
            BlockOperator2x2(op_pair.O_left,
                             get_shape(num_left_indices))); 

        right_sparse_operators[q - num_left_indices].push_back(
            SparseBlockOperator2x2(op_pair.O_right,
                                   get_shape(num_left_indices))); 
    }
}

HalfSparseFactorizedHamiltonian::HalfSparseFactorizedHamiltonian(
    FactorizedSpace _space,
    MajoranaDisorderParameter& Jtensor,
    bool mock_hamiltonian) {

    space = _space;

    state_block_alloc_size =
        sizeof(cpx) * (space.left.D / 2) * (space.right.D / 2);
    
    FactorizedHamiltonianGenericParity H(space,
                                         Jtensor,
                                         mock_hamiltonian);

    left_sparse_operators[0] = vector<SparseBlockOperator2x2>();
    left_sparse_operators[1] = vector<SparseBlockOperator2x2>();
    left_sparse_operators[2] = vector<SparseBlockOperator2x2>();

    left_dense_operators[3] = vector<BlockOperator2x2>();
    left_dense_operators[4] = vector<BlockOperator2x2>();

    right_sparse_operators[0] = vector<SparseBlockOperator2x2>();
    right_sparse_operators[1] = vector<SparseBlockOperator2x2>();

    right_dense_operators[2] = vector<BlockOperator2x2>();
    right_dense_operators[3] = vector<BlockOperator2x2>();
    right_dense_operators[4] = vector<BlockOperator2x2>();

    add_left_sparse_right_dense(H, 0);
    add_left_sparse_right_dense(H, 1);
    add_left_sparse_right_dense(H, 2);
    add_left_dense_right_sparse(H, 3);
    add_left_dense_right_sparse(H, 4);
}

void HalfSparseFactorizedHamiltonian::act_even(Mat& output_tl,
                                           Mat& output_br,
                                           const Mat& state_tl,
                                           const Mat& state_br) {
    // Left sparse
    for (size_t left_ind = 0; left_ind <= 2; left_ind++) {
        for (size_t n = 0;
             n < left_sparse_operators[left_ind].size();
             n++) {

            SparseBlockOperator2x2& O_L =
                left_sparse_operators[left_ind][n];
            BlockOperator2x2& O_R =
                right_dense_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   ( tl  0  )   ( Rtop 0    )    
                // (    0 Lbot ) . (  0  br ) . (    0 Rbot ) = 
                // 
                //    ( Ltop.tl.Rtop  0            )
                //    (            0  Lbot.br.Rbot )

                output_tl += O_L.top_block    * state_tl * O_R.top_block;
                output_br += O_L.bottom_block * state_br *O_R.bottom_block;
            }
            else {
                // Odd operators
                //
                // ( 0     Ltop )   ( tl  0  )   ( 0    Rtop )   
                // ( Lbot  0    ) . (  0  br ) . ( Rbot 0    ) = 
                // 
                //    ( Ltop.br.Rbot  0            )
                //    (            0  Lbot.tl.Rtop )

                output_tl += O_L.top_block * state_br * O_R.bottom_block;
                output_br += O_L.bottom_block * state_tl * O_R.top_block;
            }
        }
    }

    // Left dense
    for (size_t left_ind = 3; left_ind <= 4; left_ind++) {
        for (size_t n = 0;
             n < left_dense_operators[left_ind].size();
             n++) {

            BlockOperator2x2& O_L =
                left_dense_operators[left_ind][n];
            SparseBlockOperator2x2& O_R =
                right_sparse_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                output_tl += O_L.top_block    * state_tl * O_R.top_block;
                output_br += O_L.bottom_block * state_br *O_R.bottom_block;
            }
            else {
                output_tl += O_L.top_block * state_br * O_R.bottom_block;
                output_br += O_L.bottom_block * state_tl * O_R.top_block;
            }
        }
    }
}

void HalfSparseFactorizedHamiltonian::act_even(
    Mat& output, const Mat& state) {

    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tl = get_block(state, TOP_LEFT);
    Mat state_br = get_block(state, BOTTOM_RIGHT);

    Mat output_tl = Mat::Zero(state_tl.rows(), state_tl.cols());
    Mat output_br = Mat::Zero(state_br.rows(), state_br.cols());

    act_even(output_tl, output_br, state_tl, state_br);

    // Write the blocks to the output
    output.block(0,
                 0,
                 output_tl.rows(),
                 output_tl.cols()) += output_tl;

    output.block(output_tl.rows(),
                 output_tl.cols(),
                 output_br.rows(),
                 output_br.cols()) += output_br;
}

void HalfSparseFactorizedHamiltonian::act_odd(Mat& output_tr,
                                              Mat& output_bl,
                                              const Mat& state_tr,
                                              const Mat& state_bl) {

    // Left sparse
    for (size_t left_ind = 0; left_ind <= 2; left_ind++) {
        for (size_t n = 0;
             n < left_sparse_operators[left_ind].size();
             n++) {

            SparseBlockOperator2x2& O_L =
                left_sparse_operators[left_ind][n];
            BlockOperator2x2& O_R =
                right_dense_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                // Even operators
                //
                // ( Ltop 0    )   (  0  tr )   ( Rtop 0    )    
                // (    0 Lbot ) . ( bl   0 ) . (    0 Rbot ) = 
                // 
                //    (            0  Ltop.tr.Rbot )
                //    ( Lbot.bl.Rtop  0            )

                output_tr += O_L.top_block * state_tr * O_R.bottom_block;
                output_bl += O_L.bottom_block * state_bl * O_R.top_block;
            }
            else {
                //
                // ( 0     Ltop )   (  0  tr )   ( 0    Rtop )   
                // ( Lbot  0    ) . ( bl   0 ) . ( Rbot 0    ) = 
                // 
                //    (            0 Ltop.bl.Rtop )
                //    ( Lbot.tr.Rbot            0 )

                output_tr += O_L.top_block * state_bl * O_R.top_block;
                output_bl += O_L.bottom_block * state_tr *O_R.bottom_block;
            }
        }
    }

    // Left dense
    for (size_t left_ind = 3; left_ind <= 4; left_ind++) {
        for (size_t n = 0;
             n < left_dense_operators[left_ind].size();
             n++) {

            BlockOperator2x2& O_L =
                left_dense_operators[left_ind][n];
            SparseBlockOperator2x2& O_R =
                right_sparse_operators[q - left_ind][n];

            if (left_ind % 2 == 0) {
                output_tr += O_L.top_block * state_tr * O_R.bottom_block;
                output_bl += O_L.bottom_block * state_bl * O_R.top_block;
            }
            else {
                output_tr += O_L.top_block * state_bl * O_R.top_block;
                output_bl += O_L.bottom_block * state_tr *O_R.bottom_block;
            }
        }
    }
}

void HalfSparseFactorizedHamiltonian::act_odd(
    Mat& output, const Mat& state) {
    
    assert(output.rows() == state.rows());
    assert(output.cols() == state.cols());

    // tl = top-left, br = bottom-right
    Mat state_tr = get_block(state, TOP_RIGHT);
    Mat state_bl = get_block(state, BOTTOM_LEFT);

    Mat output_tr = Mat::Zero(state_tr.rows(), state_tr.cols());
    Mat output_bl = Mat::Zero(state_bl.rows(), state_bl.cols());

    act_odd(output_tr, output_bl, state_tr, state_bl);

    // Write the blocks to the output
    output.block(0,
                 output_bl.cols(),
                 output_tr.rows(),
                 output_tr.cols()) += output_tr;

    output.block(output_tr.rows(),
                 0,
                 output_bl.rows(),
                 output_bl.cols()) += output_bl;
}
