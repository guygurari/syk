#include <stdio.h>
#include <iostream>

#include "gtest/gtest.h"
#include "BasisState.h"
#include "DisorderParameter.h"
#include "KitaevHamiltonianBlock.h"
#include "KitaevHamiltonian.h"
#include "Spectrum.h"
#include "MajoranaDisorderParameter.h"
#include "NaiveMajoranaKitaevHamiltonian.h"
#include "MajoranaKitaevHamiltonian.h"
#include "Correlators.h"
#include "FockSpaceUtils.h"
#include "FactorizedSpaceUtils.h"
#include "FactorizedHamiltonian.h"
#include "Lanczos.h"
#include "RandomMatrix.h"
#include "kitaev.h"
#include "TestUtils.h"

#include <boost/random/normal_distribution.hpp>

extern boost::random::mt19937* gen;

TEST(BasisStateTest, constructor) {
    int indices[] = {0,1,2};
    BasisState expected(list<int>(indices, indices+3), -1);
    BasisState actual(indices, 3, -1);
    ASSERT_TRUE(expected == actual);
}

TEST(BasisStateTest, create_coefficient) {
    BasisState state1;
    state1.create(1);
    state1.create(0);
    ASSERT_EQ(state1.coefficient, 1);

    BasisState state2;
    state2.create(0);
    state2.create(1);
    ASSERT_EQ(state2.coefficient, -1);

    BasisState state3;
    state3.create(0);
    state3.create(0);
    ASSERT_EQ(state3.coefficient, 0);

    BasisState state4;
    state4.create(1);
    state4.create(1);
    ASSERT_EQ(state4.coefficient, 0);
}

TEST(BasisStateTest, create) {
    int indices[] = {0,1,2};
    BasisState expected(list<int>(indices, indices+3));
    //cout << "expected = " << expected.to_string() << endl;

    BasisState actual;
    actual.create(2);
    actual.create(1);
    actual.create(0);
    ASSERT_TRUE(expected == actual) << "actual = " << actual.to_string();

    BasisState wrong;
    wrong.create(0);
    wrong.create(1);
    wrong.create(2);
    ASSERT_TRUE(expected != wrong) << "wrong = " << wrong.to_string();

    BasisState right;
    right.create(0);
    right.create(2);
    right.create(1);
    ASSERT_TRUE(expected == right) << "right= " << right.to_string();
}

TEST(BasisStateTest, annihilate) {
    int ar012[] = {0,1,2};
    BasisState base_state(ar012, 3);
    BasisState actual;

    actual = base_state;
    actual.annihilate(0);
    int ar12[] = {1,2};
    ASSERT_TRUE(actual == BasisState(ar12, 2, 1));

    actual = base_state;
    actual.annihilate(1);
    int ar02[] = {0,2};
    ASSERT_TRUE(actual == BasisState(ar02, 2, -1));

    actual = base_state;
    actual.annihilate(2);
    int ar01[] = {0,1};
    ASSERT_TRUE(actual == BasisState(ar01, 2, 1));

    actual = base_state;
    actual.annihilate(3);
    ASSERT_TRUE(actual.coefficient == 0);
}

TEST(BasisStateTest, state_numbers) {
    int ar012[] = {0,1,2};
    BasisState state1(ar012, 3);
    ASSERT_EQ(state1.get_state_number(), 0);

    int ar134[] = {1,3,4};
    BasisState state2(ar134, 3);
    ASSERT_EQ(state2.get_state_number(), 8);

    // Construct from a state number
    BasisState actual(8, 3);
    ASSERT_TRUE(state2 == actual);
}

TEST(BasisStateTest, next_state) {
    int ar012[] = {0,1,2};
    int ar013[] = {0,1,3};

    BasisState expected(ar013, 3);
    BasisState state(ar012, 3);
    BasisState actual = state.next_state();
    ASSERT_TRUE(expected == actual);
}

TEST(BasisStateTest, global_state_number) {
    BasisState state2(11); // 1011 binary
    ASSERT_TRUE(state2.get_global_state_number() == 11);
    ASSERT_TRUE(state2.charge() == 3);

    list<int>::iterator iter = state2.indices.begin();
    ASSERT_TRUE(*iter == 0);
    ++iter;
    ASSERT_TRUE(*iter == 1);
    ++iter;
    ASSERT_TRUE(*iter == 3);

    int global_state_num = 123;
    BasisState state(global_state_num);
    ASSERT_TRUE(global_state_num == state.get_global_state_number());
}

TEST(KitaevDisorderParameterTest, antisymmetry) {
    KitaevDisorderParameter J(5, 1., gen);

    ASSERT_TRUE(
        J.elem(0,1,2,3) != cpx(0) ||
        J.elem(0,1,0,1) != cpx(0) ||
        J.elem(0,1,2,4) != cpx(0)
                ) << "Sanity test: must be some elements that don't vanish";

    ASSERT_EQ(J.elem(0,1,2,3), -J.elem(1,0,2,3));
    ASSERT_EQ(J.elem(0,1,2,3), -J.elem(0,1,3,2));
    ASSERT_EQ(J.elem(0,1,2,3), +J.elem(1,0,3,2));

    ASSERT_EQ(J.elem(2,3,0,1), -J.elem(3,2,0,1));
    ASSERT_EQ(J.elem(2,3,0,1), -J.elem(2,3,1,0));
    ASSERT_EQ(J.elem(2,3,0,1), +J.elem(3,2,1,0));

    ASSERT_EQ(J.elem(0,0,2,3), cpx(0));
    ASSERT_EQ(J.elem(0,1,2,2), cpx(0));

    ASSERT_EQ(J.elem(0,1,2,3), conj(J.elem(2,3,0,1)));
    ASSERT_EQ(J.elem(0,1,0,1), conj(J.elem(0,1,0,1)));
}

TEST(KitaevHamiltonianBlockTest, hermitian) {
    int N = 4;
    int Q = 2;
    //cout << "making J" << endl;
    KitaevDisorderParameter J(N, 1., gen);
    //cout << "making H" << endl;
    KitaevHamiltonianBlock H(N, Q, &J);
    //cout << "asserting" << endl;

    ASSERT_EQ(H.dim(), 6);
    ASSERT_EQ(H(0, 2), conj(H(2, 0)));
    ASSERT_EQ(H(3, 4), conj(H(4, 3)));
    ASSERT_EQ(H(1, 1), conj(H(1, 1)));
    ASSERT_EQ(H(3, 3), conj(H(3, 3)));

    assert_matrix_is_zero(
        H.matrix - H.matrix.adjoint(),
        "Hamiltonian isn't hermitian");
}

TEST(NaiveKitaevHamiltonianBlockTest, naive_hermitian) {
    int N = 4;
    int Q = 2;
    //cout << "making J" << endl;
    KitaevDisorderParameter J(N, 1., gen);
    //cout << "making H" << endl;
    NaiveKitaevHamiltonianBlock H(N, Q, &J);
    //cout << "asserting" << endl;

    ASSERT_EQ(H.dim(), binomial(N,Q));
    ASSERT_EQ(H(0, 2), conj(H(2, 0)));
    ASSERT_EQ(H(3, 4), conj(H(4, 3)));
    ASSERT_EQ(H(1, 1), conj(H(1, 1)));
    ASSERT_EQ(H(3, 3), conj(H(3, 3)));

    assert_matrix_is_zero(
        H.matrix - H.matrix.adjoint(),
        "Hamiltonian isn't hermitian");
}

TEST(NaiveKitaevHamiltonianBlockTest, naive_computed_element) {
    int N = 6;
    int Q = 4;

    KitaevDisorderParameter J(N, 1., gen);
    NaiveKitaevHamiltonianBlock H(N, Q, &J);

    int ar0123[] = {0,1,2,3};
    int ar0145[] = {0,1,4,5};

    BasisState bra(ar0145, 4);
    BasisState ket(ar0123, 4);
    ASSERT_EQ(
        H(bra.get_state_number(), ket.get_state_number()),
        - sqrt(2.) / pow(N, 3./2.) * J.elem(4,5,2,3));
}

TEST(KitaevHamiltonianBlockTest, computed_element) {
    int N = 6;
    int Q = 4;

    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonianBlock H(N, Q, &J);

    int ar0123[] = {0,1,2,3};
    int ar0145[] = {0,1,4,5};

    BasisState bra(ar0145, 4);
    BasisState ket(ar0123, 4);
    ASSERT_EQ(
        H(bra.get_state_number(), ket.get_state_number()),
        - sqrt(2.) / pow(N, 3./2.) * J.elem(4,5,2,3));
}

TEST(KitaevHamiltonianBlockTest, diagonalize) {
    int N = 4;
    int Q = 2;

    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonianBlock H(N, Q, &J);

    ASSERT_FALSE(H.diagonalized);

    H.diagonalize();
    ASSERT_TRUE(H.diagonalized);
    ASSERT_EQ(H.eigenvalues().size(), H.dim());

    Mat actual = H.U_matrix() * H.D_matrix() * H.U_matrix().adjoint();
    Mat expected = H.matrix;

    assert_matrix_is_zero(
        expected - actual, 
        "Hamiltonian and diagonalized form are different");
}

TEST(KitaevHamiltonianBlockTest, fast_initialization_sanity) {
    int N = 4;
    int Q = 1;

    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonianBlock H_fast(N, Q, &J);
    double norm = abs((H_fast.matrix * H_fast.matrix).trace());
    ASSERT_NEAR(0., norm, epsilon);
}

TEST(KitaevHamiltonianBlockTest, fast_vs_naive_initialization) {
    int N = 4;
    int Q = 2;

    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonianBlock H_fast(N, Q, &J);
    NaiveKitaevHamiltonianBlock H_naive(N, Q, &J);

    /*cout << H_fast.matrix << endl << endl;
        cout << H_naive.matrix << endl << endl;

        cout << "fast:\n" << H_fast.matrix(1,0) << endl;
        cout << "\nnaive:\n" << H_naive.matrix(1,0) << endl;

        BasisState state1(state_number_to_occupations(1,Q));
        cout << "indices in 1: " << state1.to_string() << endl;

        BasisState state0(state_number_to_occupations(0,Q));
        cout << "indices in 0: " << state0.to_string() << endl;*/

    assert_matrix_is_zero(
        H_fast.matrix - H_naive.matrix,
        "Fast and naive Hamiltonians are different"
                            );
}

TEST(KitaevHamiltonianBlockTest, fast_vs_naive_initialization2) {
    int N = 6;

    for (int Q = 0; Q <= N; Q++) {
        KitaevDisorderParameter J(N, 1., gen);
        KitaevHamiltonianBlock H_fast(N, Q, &J);
        NaiveKitaevHamiltonianBlock H_naive(N, Q, &J);

        assert_matrix_is_zero(
            H_fast.matrix - H_naive.matrix,
            "Fast and naive Hamiltonians are different"
                                );
    }
}

TEST(KitaevHamiltonianTest, basics) {
    int N = 4;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);

    ASSERT_EQ(H.blocks.size(), N+1);

    ASSERT_FALSE(H.is_diagonalized());

    H.diagonalize();
    ASSERT_TRUE(H.is_diagonalized());

    int sum_of_block_dims = 0;

    for (int Q = 0; Q <= N; Q++) {
        ASSERT_TRUE(H.blocks[Q]->diagonalized);
        sum_of_block_dims += H.blocks[Q]->dim();
    }

    ASSERT_EQ(sum_of_block_dims, H.dim());
}

TEST(KitaevHamiltonianTest, eigenvalues) {
    int N = 3;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);

    //cout << H.as_matrix() << endl;

    H.diagonalize();
    RealVec evs = H.eigenvalues();

    //cout << "evs= " << evs.transpose() << endl;

    RealVec blockNevs = H.blocks[N]->eigenvalues();
    ASSERT_EQ(blockNevs.size(), 1);

    /*cout << "blockNevs= " << blockNevs << endl;
        for (int Q = 0; Q <= N; Q++) {
        cout << "Q=" << Q << ": " << H.blocks[Q]->eigenvalues().transpose()
        << endl;
        }*/

    ASSERT_NEAR(evs[H.dim()-1], blockNevs(0), epsilon);
}

TEST(SpectrumTest, constructor) {
    int N = 3;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize();

    Spectrum s(H);
    ASSERT_TRUE((s.all_eigenvalues() - H.eigenvalues()).norm() < epsilon);
}

TEST(SpectrumTest, save_and_load) {
    int N = 3;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize();

    Spectrum s(H);
    string filename = "/tmp/test-spectrum.tsv";
    remove(filename.c_str());
    s.save(filename);

    Spectrum s2(filename);

    for (int Q = 0; Q <= N; Q++) {
        ASSERT_TRUE(
            (s.eigenvalues_by_Q[Q] - s2.eigenvalues_by_Q[Q]).norm() 
            < epsilon) << "Q=" << Q;
    }
}

TEST(MajoranaDisorderParameter, antisymmetry) {
    int N = 5;
    MajoranaKitaevDisorderParameter J(N, 1., gen);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    // These permutations generate all others,
                    // so it is enough to verify them
                    ASSERT_NEAR(
                        J.elem(i,j,k,l), 
                        -J.elem(j,i,k,l), 
                        epsilon) 
                        << i << "," << j << "," << k << "," << l;
                    ASSERT_NEAR(
                        J.elem(i,j,k,l), 
                        -J.elem(i,k,j,l), 
                        epsilon)
                        << i << "," << j << "," << k << "," << l;
                    ASSERT_NEAR(
                        J.elem(i,j,k,l), 
                        -J.elem(i,j,l,k), 
                        epsilon)
                        << i << "," << j << "," << k << "," << l;
                }
            }
        }
    }
}

TEST(NaiveMajoranaKitaevHamiltonianTest, hermitian) {
    int N = 8;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    NaiveMajoranaKitaevHamiltonian H(N, &J);

    //cout << "H = \n" << H.matrix << endl;

    ASSERT_TRUE(H.dim() == pow(2,N/2));

    assert_matrix_is_zero(
        H.matrix - H.matrix.adjoint(),
        "Hamiltonian isn't hermitian"
                            );
}

// Computed this Hamiltonian explicitly
TEST(NaiveMajoranaKitaevHamiltonianTest, explicit_matrix) {
    int N = 4;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    NaiveMajoranaKitaevHamiltonian H(N, &J);
    ASSERT_TRUE(H.dim() == 4);

    // In this case there is one Majorana term with coefficient
    // J(0,1,2,3), and the matrix turns out to be diagonal.
    ASSERT_NEAR(H.matrix(0,0).real(), - (J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(H.matrix(1,1).real(), + (J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(H.matrix(2,2).real(), + (J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(H.matrix(3,3).real(), - (J.elem(0,1,2,3)/4.), epsilon);

    for (int i = 0; i < H.dim(); i++) {
        for (int j = i+1; j < H.dim(); j++) {
            ASSERT_NEAR(abs(H.matrix(i,j)), 0, epsilon);
        }
    }
}

TEST(NaiveMajoranaKitaevHamiltonianTest, diagonalize) {
    // Use the case where the matrix is already diagonal
    int N = 4;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    NaiveMajoranaKitaevHamiltonian H(N, &J);

    ASSERT_FALSE(H.is_diagonalized());
    H.diagonalize();
    ASSERT_TRUE(H.is_diagonalized());

    RealVec evs = H.all_eigenvalues();

    //cout << evs << endl;

    ASSERT_NEAR(evs(0), - abs(J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(evs(1), - abs(J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(evs(2), + abs(J.elem(0,1,2,3)/4.), epsilon);
    ASSERT_NEAR(evs(3), + abs(J.elem(0,1,2,3)/4.), epsilon);
}

TEST(NaiveMajoranaKitaevHamiltonianTest, charge_parity_conservation) {
    int N = 8;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    NaiveMajoranaKitaevHamiltonian H(N, &J);

    for (int a = 0; a < H.dim(); a++) {
        for (int b = 0; b < H.dim(); b++) {
            // If charge parity is different, matrix element should vanish
            if (BasisState(a).charge() % 2 == 0  &&  
                BasisState(b).charge() % 2 == 1) {
                ASSERT_NEAR(abs(H.matrix(a,b)), 0., epsilon);
            }

            if (BasisState(a).charge() % 2 == 1  &&  
                BasisState(b).charge() % 2 == 0) {
                ASSERT_NEAR(abs(H.matrix(a,b)), 0., epsilon);
            }
        }
    }
}

class NullMajoranaDisorderParameterTest : public MajoranaDisorderParameter {
public:
    NullMajoranaDisorderParameterTest(double _J) : J(_J) {}
    virtual ~NullMajoranaDisorderParameterTest() {}

    virtual double elem(int a, int b, int c, int d) {
        assert(a<b);
        assert(b<c);
        assert(c<d);

        if (a == 0 && b == 4 && c == 10 && d == 14) {
            return J;
        }
        else {
            return 0;
        }
    }

    double J;
};

TEST(NaiveMajoranaKitaevHamiltonianTest, specific_element) {
    if (!run_all_slow_tests) {
        cout << "This test was disabled by setting run_all_slow_tests=false"
             << endl;
        return;
    }
    
    int N = 16;
    double Jnum = 2.5;
    NullMajoranaDisorderParameterTest J(Jnum);
    NaiveMajoranaKitaevHamiltonian H(N, &J);

    list<int> ket_indices;
    ket_indices.push_back(0);
    ket_indices.push_back(3);
    ket_indices.push_back(4);
    ket_indices.push_back(5);
    ket_indices.push_back(6);
    ket_indices.push_back(7);

    list<int> bra_indices;
    bra_indices.push_back(2);
    bra_indices.push_back(3);
    bra_indices.push_back(4);
    bra_indices.push_back(6);

    BasisState ket(ket_indices);
    BasisState bra(bra_indices);

    int bra_n = bra.get_global_state_number();
    int ket_n = ket.get_global_state_number();

    /*cout << "bra: " << bra_n << endl;
        cout << "ket: " << ket_n << endl;
        cout << "rows = " << H.matrix.rows() << endl;
        cout << "cols = " << H.matrix.cols() << endl;
        cout << "elem(" << bra_n << "," << ket_n << ") = " 
        << H.matrix(bra_n, ket_n) << endl;*/

    cpx elem = H.matrix(bra_n, ket_n);
    ASSERT_NEAR(abs(elem - cpx(-Jnum/4.,0)), 0, epsilon);
}

TEST(MajoranaKitaevHamiltonianTest, same_as_naive_implementation) {
    int N = 10;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &J);
    NaiveMajoranaKitaevHamiltonian H_naive(N, &J);

    assert_matrix_is_zero(
        H.dense_matrix() - H_naive.matrix,
        "Naive and full Hamiltonians are different");
}

TEST(MajoranaKitaevHamiltonianTest, charge_parity_conservation) {
    int N = 10;
    MajoranaKitaevDisorderParameter J(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &J);

    for (int a = 0; a < H.dim(); a++) {
        for (int b = 0; b < H.dim(); b++) {
            // If charge parity is different, matrix element should vanish
            if (BasisState(a).charge() % 2 == 0  &&  
                BasisState(b).charge() % 2 == 1) {
                ASSERT_NEAR(abs(H.dense_matrix()(a,b)), 0., epsilon);
            }

            if (BasisState(a).charge() % 2 == 1  &&  
                BasisState(b).charge() % 2 == 0) {
                ASSERT_NEAR(abs(H.dense_matrix()(a,b)), 0., epsilon);
            }
        }
    }
}

TEST(MajoranaKitaevHamiltonianTest, hermitian_charge_parity_blocks) {
    for (int N = 4; N <= 12; N += 2) {
        MajoranaKitaevDisorderParameter J(N, 1., gen);
        MajoranaKitaevHamiltonian H(N, &J);

        SpMat even_block(H.dim()/2, H.dim()/2);
        SpMat odd_block(H.dim()/2, H.dim()/2);
        H.prepare_charge_parity_blocks(even_block, odd_block);

        Mat even_block_dense(even_block);
        Mat odd_block_dense(odd_block);

        assert_matrix_is_zero(even_block_dense - even_block_dense.adjoint(), 
                                "even block should be Hermitian");
        assert_matrix_is_zero(odd_block_dense - odd_block_dense.adjoint(), 
                                "even block should be Hermitian");
    }
}

TEST(MajoranaKitaevHamiltonianTest, diagonalize) {
    // Some optimizations depend on the value of (N mod 8),
    // so we go over the whole range.
    for (int N = 4; N <= 12; N += 2) {
        MajoranaKitaevDisorderParameter J(N, 1., gen);
        MajoranaKitaevHamiltonian H(N, &J);
        NaiveMajoranaKitaevHamiltonian H_naive(N, &J);

        ASSERT_FALSE(H.is_diagonalized());
        H.diagonalize();
        ASSERT_TRUE(H.is_diagonalized());

        H_naive.diagonalize();

        RealVec H_evs = H.all_eigenvalues();
        RealVec H_naive_evs = H_naive.all_eigenvalues();

        sort(H_evs.data(), H_evs.data() + H_evs.size());
        sort(H_naive_evs.data(), H_naive_evs.data() + H_naive_evs.size());

        for (int i = 0; i < H_evs.size(); i++) {
            ASSERT_NEAR(H_evs(i), H_naive_evs(i), epsilon);
        }
    }
}

TEST(CorrelatorsTest, c_matrix_anti_commutators) {
    int N = 8;

    for (int Q = 1; Q < N; Q++) {
        for (int i = 0; i < N; i++) {
            Mat c_Q = compute_c_matrix_dense(i, N, Q);
            Mat cdagger_Qm1 = compute_cdagger_matrix_dense(i, N, Q-1);

            Mat cdagger_Q = compute_cdagger_matrix_dense(i, N, Q);
            Mat c_Qp1 = compute_c_matrix_dense(i, N, Q+1);

            Mat anti_commutator = cdagger_Qm1 * c_Q + c_Qp1 * cdagger_Q;
            Mat identity = Mat::Identity(binomial(N,Q), binomial(N,Q));
            assert_equal_matrices(identity, anti_commutator, 
                                    "anti-commutator should give identity");

            for (int j = i+1; j < N; j++) {
                Mat ci_Q = compute_c_matrix_dense(i, N, Q);
                Mat cjdagger_Qm1 = compute_cdagger_matrix_dense(j, N, Q-1);

                Mat cjdagger_Q = compute_cdagger_matrix_dense(j, N, Q);
                Mat ci_Qp1 = compute_c_matrix_dense(i, N, Q+1);

                Mat anti_commutator_ij = 
                    cjdagger_Qm1 * ci_Q + ci_Qp1 * cjdagger_Q;
                assert_matrix_is_zero(anti_commutator_ij, 
                                        "anti-commutator should give zero");
            }
        }
    }
}

// Check that Q = \sum_i c^*_i c_i
TEST(CorrelatorsTest, c_matrix_sum_gives_charge) {
    int N = 8;

    for (int Q = 1; Q < N; Q++) {
        int dim = binomial(N,Q);
        Mat charge_matrix = Mat::Zero(dim, dim);

        for (int i = 0; i < N; i++) {
            Mat c_Q = compute_c_matrix_dense(i, N, Q);
            Mat cdagger_Qm1 = compute_cdagger_matrix_dense(i, N, Q-1);

            charge_matrix += cdagger_Qm1 * c_Q;
        }

        //cout << "Q=" << Q << ":\n" << charge_matrix << endl << endl;

        Mat expected_charge_matrix = Mat::Identity(dim, dim);
        expected_charge_matrix *= Q;
        assert_equal_matrices(expected_charge_matrix, charge_matrix,
                                "charge matrix should be scalar in sector");
    }
}

TEST(CorrelatorsTest, exp_H) {
    int N = 3;
    int Q = 1;
    int dim = binomial(N,Q);

    // Make a random Hermitian block
    Mat block_mat(dim,dim);
    block_mat <<
        1.1,         cpx(2.3, 4), cpx(0.1, -1),
        cpx(2.3,-4), -2         , cpx(0.4, 5.6),
        cpx(0.1, 1), cpx(0.4, -5.6), 0.3;

    KitaevHamiltonianBlock block(N, Q, block_mat);
    block.diagonalize(true);
    cpx a = cpx(0.5, 0.2);
    Mat exp_H = compute_exp_H_naive(block, a, N);

    Mat expected(dim, dim);
    expected << 
        cpx(2.95051, + 6.57713), cpx(-6.47667, + 4.80697), 
        cpx(-2.92444, - 6.16074),
        cpx(8.31789, + 0.638982), cpx(2.9859, + 9.94082), 
        cpx(-8.58576, + 2.11314),
        cpx(0.674726, - 6.44286), cpx(8.55963, - 3.07484), 
        cpx(2.98828, + 7.60301);

    assert_equal_matrices(exp_H, expected, "exp_H != expected", 1.e-4);
}

TEST(CorrelatorsTest, exp_H_fast_vs_naive) {
    int N = 6;
    int Q = 3;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize(true, false);
    cpx a(1.2, 3.4);
    Mat naive_result = compute_exp_H_naive(H, a, N, Q);
    Mat fast_result = compute_exp_H(H, a, N, Q);
    assert_equal_matrices(naive_result, fast_result);
}

// At t=0, beta=0 the 2-point function should be exactly N/2.
TEST(CorrelatorsTest, two_pt_function_t_zero_beta_zero) {
    int N = 6;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize(true, false);

    cpx result = compute_2_point_function_naive(N, 0., 0., H);
    cpx expected(N/2., 0.);

    //cout << "result = " << result << endl;

    ASSERT_NEAR(real(result), real(expected), epsilon) << "real part";
    ASSERT_NEAR(imag(result), imag(expected), epsilon) << "imaginary part";
}

// At t=0 the 2-point is the charge density:
//      Tr(e^{-beta H} Q) / Tr(e^{-beta H}) .
// We compute it independently here.
TEST(CorrelatorsTest, two_pt_function_t_zero_is_expected_charge) {
    int N = 6;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize(true, false);

    double beta = 1.;
    double t = 0.;

    cpx result = compute_2_point_function_naive(N, beta, t, H);

    double Z = 0.;
    double expected_charge = 0.;

    for (int Q = 0; Q <= N; Q++) {
        KitaevHamiltonianBlock* block = H.blocks[Q];
        double expected_charge_in_sector = 0.;

        for (int a = 0; a < block->evs.size(); a++) {
            double E = block->evs(a);
            Z += exp(-beta * E);
            expected_charge_in_sector += exp(-beta * E) * Q;
        }

        /*cout << "Test numerator in Q=" << Q << ": " <<
            expected_charge_in_sector << endl;*/
        expected_charge += expected_charge_in_sector;
    }

    //cout << "test Z = " << Z << endl;

    expected_charge /= Z;
    ASSERT_NEAR(real(result), expected_charge, epsilon);
    ASSERT_NEAR(imag(result), 0., epsilon);
}

TEST(CorrelatorsTest, sparse_versus_dense_c_cdag_matrices) {
    int N = 6;

    for (int Q = 1; Q < N; Q++) {
        for (int i = 0; i < N; i++) {
            Mat c = compute_c_matrix_dense(i, N, Q);
            Mat cdag = compute_cdagger_matrix_dense(i, N, Q-1);

            SpMat c_sparse = compute_c_matrix(i,N,Q);
            SpMat cdag_sparse = compute_cdagger_matrix(i, N, Q-1);

            Mat c_sparse_as_dense(c_sparse);
            Mat cdag_sparse_as_dense(cdag_sparse);

            assert_equal_matrices(c, c_sparse_as_dense);
            assert_equal_matrices(cdag, cdag_sparse_as_dense);
        }
    }
}

TEST(CorrelatorsTest, naive_vs_fast_2pt_func) {
    int N = 6;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize(true, false);

    double beta = 1.3;
    double t = 1.2;

    cpx naive_result = compute_2_point_function_naive(N, beta, t, H);
    cpx fast_result = compute_2_point_function(N, beta, t, H);
    assert_cpx_equal(naive_result, fast_result);

    // Compute correlators for a range of betas and times
    vector<double> betas;
    vector<double> times;

    betas.push_back(1.5);
    betas.push_back(2.5);
    betas.push_back(10.0);

    times.push_back(0.2);
    times.push_back(0.6);
    times.push_back(5.7);
    times.push_back(2.41);

    Mat fast_correlators = compute_2_point_function(N, betas, times, H);

    for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
        for (unsigned time_i = 0; time_i < betas.size(); time_i++) {
            double beta = betas[beta_i];
            double t = times[time_i];
            naive_result = compute_2_point_function_naive(N, beta, t, H);
            assert_cpx_equal(
                naive_result, fast_correlators(beta_i,time_i));
        }
    }
}

TEST(CorrelatorsTest, euclidean_time_periodicity) {
    int N = 6;
    KitaevDisorderParameter J(N, 1., gen);
    KitaevHamiltonian H(N, &J);
    H.diagonalize(true, false);

    double beta = 2.3;
    cpx corr_zero = compute_2_point_function(N, beta, 0., H, EUCLIDEAN_TIME);
    cpx corr_beta = compute_2_point_function(N, beta, beta, H, EUCLIDEAN_TIME);
    assert_cpx_equal(cpx((double) N,0) - corr_zero, corr_beta);
}

TEST(CorrelatorsTest, majorana_2_pt_func_versus_reference_impl) {
    int N = 8;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize_full(false);
    ASSERT_TRUE(H.is_fully_diagonalized());

    vector<double> betas;
    vector<double> times;

    betas.push_back(0.);
    betas.push_back(1.);

    times.push_back(0.);
    times.push_back(1.);
    times.push_back(10.2);

    TIME_TYPE time_type = REAL_TIME;

    Mat correlators = Mat::Zero(times.size(), betas.size());
    Mat ref_correlators = Mat::Zero(times.size(), betas.size());

    vector<double> Z(betas.size());
    vector<double> ref_Z(betas.size());

    compute_majorana_2_pt_function(
        H,
        betas,
        times,
        time_type,
        correlators,
        Z,
        false);

    compute_majorana_2_pt_function_reference_implementation(
        H,
        betas,
        times,
        time_type,
        ref_correlators,
        ref_Z,
        false);

    assert_equal_matrices(ref_correlators, correlators);

    for (size_t i = 0; i < Z.size(); i++) {
        ASSERT_NEAR(ref_Z[i], Z[i], epsilon);
    }
}

// At beta=0 the correlators should be real
TEST(CorrelatorsTest, majorana_2pt_squared_trace_zero_beta) {
    int N = 6;

    double beta = 0;
    double t = 1.2;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize_full(false);
    ASSERT_TRUE(H.is_fully_diagonalized());

    vector<double> betas;
    vector<double> times;

    betas.push_back(beta);
    times.push_back(t);

    Mat corr = Mat::Zero(1,1);
    Mat corr_sqr = Mat::Zero(1,1);
    vector<double> Z(1);

    compute_majorana_2_pt_function_with_fluctuations(
        H, betas, times, REAL_TIME, corr, corr_sqr, Z, false);

    ASSERT_NEAR(0, corr(0,0).imag(), epsilon);
    ASSERT_NEAR(0, corr_sqr(0,0).imag(), epsilon);
}

TEST(CorrelatorsTest, majorana_2pt_func_with_fluctuations) {
    int N = 6;

    double beta = 2.3;
    double t = 1.2;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize_full(false);
    ASSERT_TRUE(H.is_fully_diagonalized());

    Mat exp_beta_H = compute_exp_H(H, -beta);
    Mat exp_it_H = compute_exp_H(H, cpx(0,t));

    cpx expected_corr = 0.;
    for (int i = 0; i < N; i++) {
        expected_corr += (
            exp_beta_H * 
            exp_it_H * H.chi[i] * exp_it_H.adjoint() *
            H.chi[i]
                            ).trace();
    }
    expected_corr /= N;


    cpx expected_fluc = 0.;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            expected_fluc += (
                exp_beta_H * 
                exp_it_H * H.chi[i] * exp_it_H.adjoint() *
                H.chi[i] *
                H.chi[j] *
                exp_it_H * H.chi[j] * exp_it_H.adjoint()
                                ).trace();
        }
    }
    expected_fluc /= (N*N);

    vector<double> betas;
    vector<double> times;

    betas.push_back(beta);
    times.push_back(t);

    Mat corr = Mat::Zero(1,1);
    Mat corr_sqr = Mat::Zero(1,1);
    vector<double> Z(1);

    compute_majorana_2_pt_function_with_fluctuations(
        H, betas, times, REAL_TIME, corr, corr_sqr, Z, false);

    //cout << expected_fluc << endl;
    //cout << corr_sqr(0,0) << endl;

    assert_cpx_equal(compute_partition_function(H.evs, beta), Z[0], "Z");
    assert_cpx_equal(expected_corr, corr(0,0), "correlator");
    assert_cpx_equal(expected_fluc, corr_sqr(0,0), "correlator squared");
}

TEST(CorrelatorsTest, majorana_2pt_func_with_fluctuations_euclidean) {
    int N = 6;

    double beta = 2.3;
    double tau = 1.2;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize_full(false);
    ASSERT_TRUE(H.is_fully_diagonalized());

    Mat exp_beta_H = compute_exp_H(H, -beta);
    Mat exp_tau_H = compute_exp_H(H, tau);
    Mat exp_minus_tau_H = compute_exp_H(H, -tau);

    cpx expected_corr = 0.;
    for (int i = 0; i < N; i++) {
        expected_corr += (
            exp_beta_H * 
            exp_tau_H * H.chi[i] * exp_minus_tau_H *
            H.chi[i]
                            ).trace();
    }
    expected_corr /= N;

    cpx expected_fluc = 0.;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            expected_fluc += (
                exp_beta_H * 
                exp_tau_H * H.chi[i] * exp_minus_tau_H *
                H.chi[i] *
                H.chi[j] *
                exp_tau_H * H.chi[j] * exp_minus_tau_H
                                ).trace();
        }
    }
    expected_fluc /= (N*N);

    vector<double> betas;
    vector<double> times;

    betas.push_back(beta);
    times.push_back(tau);

    Mat corr = Mat::Zero(1,1);
    Mat corr_sqr = Mat::Zero(1,1);
    vector<double> Z(1);

    compute_majorana_2_pt_function_with_fluctuations(
        H, betas, times, EUCLIDEAN_TIME, corr, corr_sqr, Z, false);

    assert_cpx_equal(compute_partition_function(H.evs, beta), Z[0], "Z");
    assert_cpx_equal(expected_corr, corr(0,0), "corr");
    assert_cpx_equal(expected_fluc, corr_sqr(0,0), "corr^2");
}

// Test the algebra of the sparse c,cdagger matrices
TEST(FockSpaceUtilsTest, compute_c_matrix) {
    int N = 8;
    Mat Id = Mat::Identity(dim(N), dim(N));

    for (int i = 0; i < N; i++) {
        SpMat c_i = compute_c_matrix(i, N);

        for (int j = 0; j < N; j++) {
            SpMat c_j = compute_c_matrix(j, N);
            SpMat cd_j = compute_cdagger_matrix(j, N);

            SpMat c_c_anticomm = c_i * c_j + c_j * c_i;
            SpMat c_cd_anticomm = c_i * cd_j + cd_j * c_i;

            assert_matrix_is_zero(c_c_anticomm);

            if (i == j) {
                assert_equal_matrices(c_cd_anticomm, Id);
            }
            else {
                assert_matrix_is_zero(c_cd_anticomm);
            }
        }
    }
}

TEST(FockSpaceUtilsTest, compute_chi_matrix) {
    int mN = 14;
    int dirac_N = mN/2;
    Mat Id = Mat::Identity(dim(dirac_N), dim(dirac_N));

    for (int a = 0; a < mN; a++) {
        SpMat chi_a = compute_chi_matrix(a, mN);

        for (int b = 0; b < mN; b++) {
            SpMat chi_b = compute_chi_matrix(b, mN);

            SpMat chi_chi_anticomm = chi_a * chi_b + chi_b * chi_a;

            if (a == b) {
                assert_equal_matrices(chi_chi_anticomm, Id);
            }
            else {
                assert_matrix_is_zero(chi_chi_anticomm);
            }
        }
    }
}

TEST(FockSpaceUtilsTest, compute_chi_matrices) {
    int mN = 14;
    vector<SpMat> chi = compute_chi_matrices(mN);

    for (int a = 0; a < mN; a++) {
        SpMat chi_a = compute_chi_matrix(a, mN);
        assert_equal_matrices(chi[a], chi_a);
    }
}

// Compute the trace of a sparse matrix
cpx trace(SpMat A) {
    // Eigen can't take the trace of a sparse
    // matrix, so we have to do it by hand.
    cpx trace = 0.;
    for (int i = 0; i < A.outerSize(); i++) {
        trace += A.coeff(i,i);
    }
    return trace;
}

TEST(FockSpaceUtilsTest, trace_of_chi_matrices_sanity) {
    int mN = 8;
    vector<SpMat> chi = compute_chi_matrices(mN);
    assert_cpx_equal(pow(2.,mN/2) / 2., trace(chi[0] * chi[0]));
    assert_cpx_equal(0., trace(chi[0] * chi[1]));
}

// Compare trace of chi matrices to analytic result
TEST(FockSpaceUtilsTest, trace_of_chi_matrices_2_prod) {
    int mN = 8;
    vector<SpMat> chi = compute_chi_matrices(mN);

    cpx sum_of_traces = 0.;

    // Trace of sum of product of 8 Majorana matrics.
    // This shows up in the calculation of g_cf at t=0 at leading
    // order in beta.
    for (int a = 0; a < mN; a++) {
        for (int b = a+1; b < mN; b++) {
            for (int ap = 0; ap < mN; ap++) {
                for (int bp = ap+1; bp < mN; bp++) {
                    SpMat prod = chi[a] * chi[b] * chi[ap] * chi[bp];
                    sum_of_traces += trace(prod);
                }
            }
        }
    }

    double expected = - pow(2.,mN/2) * binomial(mN,2) / 4.;
    assert_cpx_equal(expected, sum_of_traces, "trace of 8 matrices");
}

// Compare trace of chi matrices to analytic result
TEST(FockSpaceUtilsTest, trace_of_chi_matrices_8_prod) {
    int mN = 8;
    vector<SpMat> chi = compute_chi_matrices(mN);

    cpx sum_of_traces = 0.;

    // Trace of sum of product of 8 Majorana matrics.
    // This shows up in the calculation of g_cf at t=0 at leading
    // order in beta.
    for (int a = 0; a < mN; a++) {
        for (int b = a+1; b < mN; b++) {
            for (int c = b+1; c < mN; c++) {
                for (int d = c+1; d < mN; d++) {
                    for (int ap = 0; ap < mN; ap++) {
                        for (int bp = ap+1; bp < mN; bp++) {
                            for (int cp = bp+1; cp < mN; cp++) {
                                for (int dp = cp+1; dp < mN; dp++) {
                                    SpMat prod = 
                                        chi[a] * chi[b] * chi[c] * chi[d] *
                                        chi[ap] * chi[bp] * chi[cp] * chi[dp];

                                    sum_of_traces += trace(prod);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double expected = pow(2.,mN/2) * binomial(mN,4) / 16.;
    assert_cpx_equal(expected, sum_of_traces, "trace of 8 matrices");
}

TEST(SparseRandomMatrixTest, test_num_nonzero_elements) {
    int K = 100;
    int num_nonzeros = 10;
    SparseHermitianRandomMatrix matrix(K, num_nonzeros, gen);
    ASSERT_EQ(matrix.matrix.nonZeros(), num_nonzeros);
}

TEST(SparseRandomMatrixTest, big_matrix_doesnt_crash) {
    int K = 1000000;
    int num_nonzeros = 10;
    SparseHermitianRandomMatrix matrix(K, num_nonzeros, gen);
}

void test_operator_size(
    FactorizedSpace& space,
    FactorizedOperatorPair& op) {

    int D_left = space.left.D;
    int D_right = space.right.D;

    ASSERT_EQ(op.O_left.rows(), D_left);
    ASSERT_EQ(op.O_left.cols(), D_left);
    ASSERT_EQ(op.O_right.rows(), D_right);
    ASSERT_EQ(op.O_right.cols(), D_right);
}

void test_FactorizedHamiltonianGenericParity(int N) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);
    FactorizedHamiltonianGenericParity H(space, Jtensor);

    for (size_t left_ind = 0;
            left_ind < H.operators.size();
            left_ind++) {

        for (size_t n = 0; n < H.operators[left_ind].size(); n++) {
            test_operator_size(H.space, H.operators[left_ind][n]);
        }
    }
}

TEST(FactorizedHamiltonianTest,
        FactorizedHamiltonianGenericParity_even_N) {

    test_FactorizedHamiltonianGenericParity(8);
}

TEST(FactorizedHamiltonianTest,
        FactorizedHamiltonianGenericParity_odd_N) {

    test_FactorizedHamiltonianGenericParity(10);
}

void test_hamiltonian_state_product_case(
    int N,
    MajoranaKitaevDisorderParameter& Jtensor) {

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    FactorizedHamiltonianGenericParity factH(space, Jtensor);

    // Initialize random state
    boost::random::normal_distribution<> dist(0., 1.);
    Vec state = Vec::Zero(H.dim());

    //state(2) = 1;

    for (int i = 0; i < state.size(); i++) {
        state(i) = cpx(dist(*gen), dist(*gen));
    }

    state = state / state.norm();

    // Ordinary product
    Vec product = H.matrix * state;

    Mat expected_fact_product = get_factorized_state(space, product);

    // Initialize factorized state
    Mat fact_state = get_factorized_state(space, state);

    Mat fact_product = Mat::Zero(factH.space.left.D,
                                    factH.space.right.D);

    factH.act(fact_product, fact_state);

    assert_equal_matrices(expected_fact_product, fact_product);
}

TEST(FactorizedHamiltonianTest, hamiltonian_state_product) {
    int N = 8;

    MajoranaKitaevDisorderParameter Jtensor_4_left(N);
    Jtensor_4_left.Jelems[0][1][2][3] = 1.;
    test_hamiltonian_state_product_case(N, Jtensor_4_left);

    MajoranaKitaevDisorderParameter Jtensor_3_left(N);
    Jtensor_3_left.Jelems[0][1][2][4] = 1.;
    test_hamiltonian_state_product_case(N, Jtensor_3_left);

    MajoranaKitaevDisorderParameter Jtensor_2_left(N);
    Jtensor_2_left.Jelems[0][1][4][5] = 1.;
    test_hamiltonian_state_product_case(N, Jtensor_2_left);

    MajoranaKitaevDisorderParameter Jtensor_1_left(N);
    Jtensor_1_left.Jelems[0][4][5][6] = 1.;
    test_hamiltonian_state_product_case(N, Jtensor_1_left);

    MajoranaKitaevDisorderParameter Jtensor_0_left(N);
    Jtensor_0_left.Jelems[4][5][6][7] = 1.;
    test_hamiltonian_state_product_case(N, Jtensor_0_left);

    MajoranaKitaevDisorderParameter Jtensor_random(N, 1., gen);
    test_hamiltonian_state_product_case(N, Jtensor_random);
}

TEST(BasisStateTest, GlobalStateIterator) {
    Space space = Space::from_dirac(4);
    GlobalStateIterator iter(space);

    int num_states = 0;

    while (!iter.done()) {
        num_states++;
        iter.next();
    }

    ASSERT_EQ(num_states, space.D);
}

TEST(FactorizedSpaceUtilsTest, FactorizedSpace_even) {
    // Default N_left
    FactorizedSpace space = FactorizedSpace::from_majorana(12);
    ASSERT_EQ(space.N, 12);
    ASSERT_EQ(space.Nd, 6);
    ASSERT_EQ(space.D, pow(2, 6));

    ASSERT_EQ(space.left.Nd, 3);
    ASSERT_EQ(space.left.D, pow(2, 3));
    ASSERT_EQ(space.left.N, 6);

    ASSERT_EQ(space.right.Nd, 3);
    ASSERT_EQ(space.right.D, pow(2, 3));
    ASSERT_EQ(space.right.N, 6);

    // Nd
    space = FactorizedSpace::from_dirac(6);
    ASSERT_EQ(space.N, 12);
    ASSERT_EQ(space.Nd, 6);
    ASSERT_EQ(space.D, pow(2, 6));

    ASSERT_EQ(space.left.Nd, 3);
    ASSERT_EQ(space.left.D, pow(2, 3));
    ASSERT_EQ(space.left.N, 6);

    ASSERT_EQ(space.right.Nd, 3);
    ASSERT_EQ(space.right.D, pow(2, 3));
    ASSERT_EQ(space.right.N, 6);

    // Specified N_left
    space = FactorizedSpace::from_majorana(12, 4);
    ASSERT_EQ(space.N, 12);
    ASSERT_EQ(space.Nd, 6);
    ASSERT_EQ(space.D, pow(2, 6));

    ASSERT_EQ(space.left.N, 4);
    ASSERT_EQ(space.left.Nd, 2);
    ASSERT_EQ(space.left.D, pow(2, 2));

    ASSERT_EQ(space.right.N, 8);
    ASSERT_EQ(space.right.Nd, 4);
    ASSERT_EQ(space.right.D, pow(2, 4));

    // Dirac
    space = FactorizedSpace::from_dirac(6, 2);
    ASSERT_EQ(space.N, 12);
    ASSERT_EQ(space.Nd, 6);
    ASSERT_EQ(space.D, pow(2, 6));

    ASSERT_EQ(space.left.N, 4);
    ASSERT_EQ(space.left.Nd, 2);
    ASSERT_EQ(space.left.D, pow(2, 2));

    ASSERT_EQ(space.right.N, 8);
    ASSERT_EQ(space.right.Nd, 4);
    ASSERT_EQ(space.right.D, pow(2, 4));
}

TEST(FactorizedSpaceUtilsTest, FactorizedSpace_odd) {
    // Default N_left
    FactorizedSpace space = FactorizedSpace::from_majorana(10);
    ASSERT_EQ(space.N, 10);
    ASSERT_EQ(space.Nd, 5);
    ASSERT_EQ(space.D, pow(2, 5));

    ASSERT_EQ(space.left.Nd, 2);
    ASSERT_EQ(space.left.D, pow(2, 2));
    ASSERT_EQ(space.left.N, 4);

    ASSERT_EQ(space.right.Nd, 3);
    ASSERT_EQ(space.right.D, pow(2, 3));
    ASSERT_EQ(space.right.N, 6);

    // Dirac
    space = FactorizedSpace::from_dirac(5);
    ASSERT_EQ(space.N, 10);
    ASSERT_EQ(space.Nd, 5);
    ASSERT_EQ(space.D, pow(2, 5));

    ASSERT_EQ(space.left.Nd, 2);
    ASSERT_EQ(space.left.D, pow(2, 2));
    ASSERT_EQ(space.left.N, 4);

    ASSERT_EQ(space.right.Nd, 3);
    ASSERT_EQ(space.right.D, pow(2, 3));
    ASSERT_EQ(space.right.N, 6);

    // Specified N_left
    space = FactorizedSpace::from_majorana(10, 2);
    ASSERT_EQ(space.N, 10);
    ASSERT_EQ(space.Nd, 5);
    ASSERT_EQ(space.D, pow(2, 5));

    ASSERT_EQ(space.left.N, 2);
    ASSERT_EQ(space.left.Nd, 1);
    ASSERT_EQ(space.left.D, pow(2, 1));

    ASSERT_EQ(space.right.N, 8);
    ASSERT_EQ(space.right.Nd, 4);
    ASSERT_EQ(space.right.D, pow(2, 4));

    // Dirac
    space = FactorizedSpace::from_dirac(5, 1);
    ASSERT_EQ(space.N, 10);
    ASSERT_EQ(space.Nd, 5);
    ASSERT_EQ(space.D, pow(2, 5));

    ASSERT_EQ(space.left.N, 2);
    ASSERT_EQ(space.left.Nd, 1);
    ASSERT_EQ(space.left.D, pow(2, 1));

    ASSERT_EQ(space.right.N, 8);
    ASSERT_EQ(space.right.Nd, 4);
    ASSERT_EQ(space.right.D, pow(2, 4));
}

TEST(FactorizedSpaceUtilsTest, get_unfactorized_state) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);
    Vec state = get_random_state(space, gen);
    Mat fact_state = get_factorized_state(space, state);
    Vec unfact_state = get_unfactorized_state(space, fact_state);
    assert_equal_vectors(state, unfact_state);
}

// compare ordinary psi/state product with the factorized space
// product.
TEST(FactorizedHamiltonianTest, psi_in_combinatorial_ordering) {
    int N = 8;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    // Use global ordering
    vector<SpMat> chi_fock = compute_chi_matrices(N);

    // Unfactorized
    Vec state = Vec::Zero(space.D);
    state(1) = 1.;

    Vec expected = chi_fock[0] * chi_fock[4] * state;
    Mat fact_expected = get_factorized_state(space, expected);

    // Factorized
    vector<SpMat> chi_fock_left = compute_chi_matrices(space.left.N);
    vector<SpMat> chi_fock_right = compute_chi_matrices(space.right.N);

    Mat chi0_left = psi_in_charge_parity_ordering(space.left,
                                                    chi_fock_left[0]);

    Mat chi4_right = psi_in_charge_parity_ordering(space.right,
                                                    chi_fock_right[0]);

    Mat sign_flipper = Mat::Zero(space.left.D, space.left.D);

    for (int i = 0; i < space.left.D/2; i++) {
        sign_flipper(i,i) = 1.;
        sign_flipper(i + space.left.D/2, i + space.left.D/2) = -1.;
    }

    Mat fact_state = get_factorized_state(space, state);
    Mat actual =
        chi0_left * sign_flipper *
        fact_state * chi4_right.transpose();

    //cout << "chi4_fock = \n" << Mat(chi_fock[4]) << endl << endl;
    /*cout << "chi4_fock_right = \n" << Mat(chi_fock_right[0]) << endl << endl;
    cout << "chi4_right = \n" << chi4_right << endl << endl;
    cout << "expected =\n" << fact_expected << endl << endl;
    cout << "actual = \n" << actual << endl << endl;*/

    assert_equal_matrices(fact_expected, actual);
}

void test_act_with_parity(int N, ChargeParity parity) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedHamiltonianGenericParity gold_H(space, Jtensor);
    FactorizedHamiltonian H(space, Jtensor);

    Mat state = get_factorized_random_state(space, parity, gen);

    Mat gold_output = Mat::Zero(space.left.D, space.right.D);
    gold_H.act(gold_output, state);

    Mat output = Mat::Zero(space.left.D, space.right.D);

    if (parity == EVEN_CHARGE) {
        H.act_even(output, state);
    }
    else {
        H.act_odd(output, state);
    }

    assert_equal_matrices(gold_output, output);
}

TEST(FactorizedHamiltonianTest, act_even) {
    test_act_with_parity(8, EVEN_CHARGE);
    test_act_with_parity(10, EVEN_CHARGE);
}

TEST(FactorizedHamiltonianTest, act_odd) {
    test_act_with_parity(8, ODD_CHARGE);
    test_act_with_parity(10, ODD_CHARGE);
}

void test_sparse_act_with_parity(int N, ChargeParity parity) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedHamiltonianGenericParity gold_H(space, Jtensor);
    SparseFactorizedHamiltonian H(space, Jtensor);

    Mat state = get_factorized_random_state(space, parity, gen);

    Mat gold_output = Mat::Zero(space.left.D, space.right.D);
    gold_H.act(gold_output, state);

    Mat output = Mat::Zero(space.left.D, space.right.D);

    if (parity == EVEN_CHARGE) {
        H.act_even(output, state);
    }
    else {
        H.act_odd(output, state);
    }

    assert_equal_matrices(gold_output, output);
}

TEST(FactorizedHamiltonianTest, sparse_act_even) {
    test_sparse_act_with_parity(8, EVEN_CHARGE);
    test_sparse_act_with_parity(10, EVEN_CHARGE);
}

TEST(FactorizedHamiltonianTest, sparse_act_odd) {
    test_sparse_act_with_parity(8, ODD_CHARGE);
    test_sparse_act_with_parity(10, ODD_CHARGE);
}

void test_half_sparse_act_with_parity(int N, ChargeParity parity) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedHamiltonianGenericParity gold_H(space, Jtensor);
    HalfSparseFactorizedHamiltonian H(space, Jtensor);

    Mat state = get_factorized_random_state(space, parity, gen);

    Mat gold_output = Mat::Zero(space.left.D, space.right.D);
    gold_H.act(gold_output, state);

    Mat output = Mat::Zero(space.left.D, space.right.D);

    if (parity == EVEN_CHARGE) {
        H.act_even(output, state);
    }
    else {
        H.act_odd(output, state);
    }

    assert_equal_matrices(gold_output, output);
}

TEST(FactorizedHamiltonianTest, half_sparse_act_even) {
    test_half_sparse_act_with_parity(8, EVEN_CHARGE);
    test_half_sparse_act_with_parity(10, EVEN_CHARGE);
}

TEST(FactorizedHamiltonianTest, half_sparse_act_odd) {
    test_half_sparse_act_with_parity(8, ODD_CHARGE);
    test_half_sparse_act_with_parity(10, ODD_CHARGE);
}

TEST(eigen_utils_test, read_write_matrix_binary) {
    Mat mat = get_random_matrix(10, 15, gen);

    string filename = "/tmp/test-kitaev-matrix-binary";
    ofstream out(filename.c_str(), ofstream::binary);
    write_matrix_binary(out, mat);
    out.close();

    ifstream in(filename.c_str(), ofstream::binary);
    Mat actual = read_matrix_binary(in);
    in.close();

    assert_equal_matrices(mat, actual);
}

TEST(FactorizedHamiltonianTest, psi_in_combinatorial_ordering_test) {
    // Space space = Space::from_majorana(12);
    Space space = Space::from_majorana(6);
    vector<SpMat> orig_psis = compute_chi_matrices(space.N);

    for (size_t a = 0; a < orig_psis.size(); a++) {
        Mat psi_fock_ordering = Mat(orig_psis[a]);
        Mat psi_comb_ordering =
            psi_in_charge_parity_ordering(space, psi_fock_ordering);

        assert_equal_matrices(psi_comb_ordering,
                              psi_comb_ordering.adjoint());

        // Make sure they have the same eigenvalues
        SelfAdjointEigenSolver<Mat> fock_solver;
        fock_solver.compute(psi_fock_ordering, EigenvaluesOnly);
        RealVec fock_evs = fock_solver.eigenvalues();

        SelfAdjointEigenSolver<Mat> comb_solver;
        comb_solver.compute(psi_comb_ordering, EigenvaluesOnly);
        RealVec comb_evs = comb_solver.eigenvalues();

        // cout << "a = " << a << endl;
        //     cout << fock_evs << endl << endl;
        //     cout << comb_evs << endl << endl;

        assert_equal_vectors(fock_evs, comb_evs);
    }
}
