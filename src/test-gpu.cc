#include "gtest/gtest.h"
#include "CudaUtils.h"
#include "TestUtils.h"
#include "FactorizedSpaceUtils.h"
#include "Lanczos.h"

#include "CudaState.h"
#include "CudaHamiltonian.h"
#include "CudaLanczos.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

boost::random::mt19937* gen = new boost::random::mt19937(time(0));
size_t default_avail_memory = 4000000000;

class CudaTestFixture : public ::testing::Test {
public:
    cublasHandle_t handle;
    cusparseHandle_t handle_sp;

    virtual void SetUp() {
        handle = cublas_init();
        handle_sp = cusparse_init();
    }

    virtual void TearDown() {
        cublas_destroy(handle);
        cusparse_destroy(handle_sp);
    }
};

class CudaUtilsTest : public CudaTestFixture {};
class CudaLanczosTest : public CudaTestFixture {};
class CudaStateTest : public CudaTestFixture {};
class CudaEvenStateTest : public CudaTestFixture {};

void assert_equal_cucpx(cucpx a, cucpx b) {
    ASSERT_NEAR(a.x, b.x, epsilon);
    ASSERT_NEAR(a.y, b.y, epsilon);
}

TEST_F(CudaUtilsTest, copying_matrices) {
    int n = 2;

    Mat A(n,n);
    A(0,0) = 1.; A(0,1) = 2.;
    A(1,0) = 3.; A(1,1) = 4.;

    Mat B(n,n) ;
    B(0,0) = 1.1; B(0,1) = 2.2;
    B(1,0) = 3.3; B(1,1) = 4.4;

    Mat C = A * B;

    cucpx* d_A = d_alloc_copy_matrix(A);
    cucpx* d_B = d_alloc_copy_matrix(B);
    cucpx* d_C = d_alloc(sizeof(cucpx) * n * n);

    CudaScalar d_zero(0., 0.);
    CudaScalar d_one(1., 0.);

    checkCublasErrors(cublasZgemm(
                        handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n,
                        d_one.ptr,
                        d_A, n,
                        d_B, n,
                        d_zero.ptr,
                        d_C, n));

    checkCudaErrors(cudaDeviceSynchronize());

    Mat actual_C = copy_matrix_device_to_host(d_C, n, n);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    assert_equal_matrices(C, actual_C);
}

TEST_F(CudaUtilsTest, inverse) {
    double* d_x = (double*) d_alloc(sizeof(double));
    double* d_y = (double*) d_alloc(sizeof(double));

    double x = 1.2;
    double y = 0.;

    checkCudaErrors(cudaMemcpy(d_x, &x, sizeof(double),
                                cudaMemcpyHostToDevice));

    d_inverse(d_y, d_x);
    checkCudaErrors(cudaMemcpy(&y, d_y, sizeof(double),
                                cudaMemcpyDeviceToHost));

    d_free(d_x);
    d_free(d_y);

    ASSERT_EQ(1./x, y);
}

TEST_F(CudaUtilsTest, CudaSparseMatrix) {
    CudaScalar d_one(1., 0.);
    CudaScalar d_zero(0., 0.);

    int A_rows = 2;
    int A_cols = 2;
    int B_rows = 2; 
    int B_cols = 2;
    int C_rows = 2; 
    int C_cols = 2; 

    Mat A = Mat::Zero(A_rows, A_cols);
    A(0,0) = 1.;
    A(0,1) = 2.;
    A(1,0) = 3.;

    CudaSparseMatrix d_sp_A(A);

    Mat B = Mat::Zero(B_rows, B_cols);
    B(0,0) = 5.;
    B(1,0) = -6.;
    B(1,1) = 0.5;

    cucpx* d_B = d_alloc(sizeof(cucpx) * B_rows * B_cols);
    checkCudaErrors(
        cudaMemcpy(d_B, B.data(), sizeof(cucpx) * B_rows * B_cols,
                    cudaMemcpyHostToDevice));

    cucpx* d_C = d_alloc(sizeof(cucpx) * C_rows * C_cols);
    checkCudaErrors(cudaMemset(d_C, 0, sizeof(cucpx) * C_rows * C_cols));

    cusparseMatDescr_t sp_description = 0;
    checkCusparseErrors(cusparseCreateMatDescr(&sp_description));
    checkCusparseErrors(
        cusparseSetMatType(sp_description,
                            CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCusparseErrors(
        cusparseSetMatIndexBase(sp_description,
                                CUSPARSE_INDEX_BASE_ZERO));

    checkCusparseErrors(
        cusparseZcsrmm(handle_sp,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        A_rows,
                        B_cols,
                        A_cols,
                        d_sp_A.nz_elems, // nnz
                        d_one.ptr, // alpha
                        sp_description,
                        d_sp_A.d_values,   // A values
                        d_sp_A.d_row_ptr,  // A row ptr
                        d_sp_A.d_col_ind,  // A col index
                        d_B, // B
                        B_rows,
                        d_zero.ptr, // beta
                        d_C,
                        C_rows));

    Mat C = Mat::Zero(C_rows, C_cols);
    checkCudaErrors(
        cudaMemcpy(C.data(), d_C, sizeof(cucpx) * C_rows * C_cols,
                    cudaMemcpyDeviceToHost));

    d_free(d_B);
    d_free(d_C);

    checkCusparseErrors(cusparseDestroyMatDescr(sp_description));

    assert_equal_matrices(C, A*B);
}

TEST_F(CudaStateTest, copying) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    Mat zero = Mat::Zero(space.left.D, space.right.D);
    Mat matrix = zero;
    matrix(0,0) = 1.;
    matrix(0,1) = 2.;

    CudaEvenState state(space, matrix);

    Mat actual = state.get_matrix();
    assert_equal_matrices(matrix, actual);

    CudaEvenState state2(space);
    state2.set(matrix);
    actual = state2.get_matrix();
    assert_equal_matrices(matrix, actual);
}

TEST_F(CudaStateTest, dot) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    Mat mat1 = get_factorized_random_state(space, EVEN_CHARGE, gen);
    Mat mat2 = get_factorized_random_state(space, EVEN_CHARGE, gen);

    CudaEvenState state1(space, mat1);
    CudaEvenState state2(space, mat2);

    cucpx* d_result = d_alloc(sizeof(cucpx));

    state1.dot(handle, d_result, state2);

    cucpx result;
    checkCudaErrors(cudaMemcpy(&result,
                                d_result, 
                                sizeof(cucpx),
                                cudaMemcpyDeviceToHost));

    cpx c_expected = mat1.conjugate().cwiseProduct(mat2).sum();
    cucpx expected = make_cuDoubleComplex(c_expected.real(),
                                            c_expected.imag());

    assert_equal_cucpx(expected, result);

    d_free(d_result);
}

TEST_F(CudaStateTest, add) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    Mat mat1 = get_factorized_random_state(space, EVEN_CHARGE, gen);
    Mat mat2 = get_factorized_random_state(space, EVEN_CHARGE, gen);

    CudaEvenState state1(space, mat1);
    CudaEvenState state2(space, mat2);

    cpx alpha(2., -3.);
    CudaScalar d_alpha(alpha);

    state1.add(handle, d_alpha.ptr, state2);
    Mat result = state1.get_matrix();

    Mat expected = mat1 + alpha * mat2;

    assert_equal_matrices(expected, result);
}

TEST_F(CudaStateTest, norm) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    Mat mat1 = get_factorized_random_state(space, EVEN_CHARGE, gen);
    CudaEvenState state1(space, mat1);

    double* d_result;
    checkCudaErrors(cudaMalloc((void**) &d_result, sizeof(double)));

    state1.norm(handle, d_result);

    double result;
    checkCudaErrors(cudaMemcpy(&result, d_result, sizeof(double),
                                cudaMemcpyDeviceToHost));
    d_free(d_result);

    double expected = mat1.norm();

    ASSERT_NEAR(expected, result, epsilon);
}

TEST_F(CudaStateTest, set_from_other) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    Mat mat1 = get_factorized_random_state(space, EVEN_CHARGE, gen);
    Mat mat2 = get_factorized_random_state(space, EVEN_CHARGE, gen);

    CudaEvenState state1(space, mat1);
    CudaEvenState state2(space, mat2);

    state2.set(state1);

    Mat result = state2.get_matrix();
    assert_equal_matrices(mat1, result);
}

TEST_F(CudaEvenStateTest, scale) {
    FactorizedSpace space = FactorizedSpace::from_majorana(10);

    double alpha = 2.3;
    double* d_alpha = (double*) d_alloc(sizeof(double));
    checkCudaErrors(cudaMemcpy(d_alpha, &alpha, sizeof(double),
                                cudaMemcpyHostToDevice));

    Mat mat1 = get_factorized_random_state(space, EVEN_CHARGE, gen);
    CudaEvenState state1(space, mat1);
    state1.scale(handle, d_alpha);
    Mat result = state1.get_matrix();

    d_free(d_alpha);

    assert_equal_matrices(alpha * mat1, result);
}

/////////////////////////////////////////////////////////////

void test_act_even(
    int N,
    MajoranaKitaevDisorderParameter& Jtensor,
    size_t available_memory = default_avail_memory) {

    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    CudaEvenState state(space, gen);
    CudaEvenState out_state(space);

    // MajoranaKitaevDisorderParameter Jtensor(space.N);
    // Jtensor.Jelems[0][1][2][4] = 1.;
    // Mat state = Mat::Zero(space.left.D, space.right.D);
    // state(0,0) = 1.;

    CudaHamiltonian H(space, Jtensor, available_memory, false);
    H.act_even(out_state, state);

    checkCudaErrors(cudaDeviceSynchronize());

    Mat out_state_matrix = out_state.get_matrix();

    // Print and compare
    // cout << "\nState:\n\n" << state.get_matrix() << "\n\n";
    // cout << "\nResult:\n\n" << out_state.get_matrix() << "\n\n";

    Mat expected = Mat::Zero(space.left.D, space.right.D);
    FactorizedHamiltonian host_H(space, Jtensor);
    host_H.act_even(expected, state.get_matrix());

    // cout << "Expected result:\n\n" << expected << "\n\n";

    // Teardown
    H.destroy();
    state.destroy();
    out_state.destroy();

    assert_equal_matrices(expected, out_state_matrix);
}

void test_act_even(int N, size_t available_memory = default_avail_memory) {
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    test_act_even(N, Jtensor, available_memory);
}

void test_act_even_left4_right0(int N) {
    MajoranaKitaevDisorderParameter Jtensor(N);
    Jtensor.Jelems[0][1][2][3] = 1.;
    test_act_even(N, Jtensor);
}

void test_act_even_left3_right1(int N) {
    MajoranaKitaevDisorderParameter Jtensor(N);
    Jtensor.Jelems[0][1][2][N-1] = 1.;
    test_act_even(N, Jtensor);
}

void test_act_even_left2_right2(int N) {
    MajoranaKitaevDisorderParameter Jtensor(N);
    Jtensor.Jelems[0][1][N-2][N-1] = 1.;
    test_act_even(N, Jtensor);
}

void test_act_even_left1_right3(int N) {
    MajoranaKitaevDisorderParameter Jtensor(N);
    Jtensor.Jelems[0][N-3][N-2][N-1] = 1.;
    test_act_even(N, Jtensor);
}

void test_act_even_left0_right4(int N) {
    MajoranaKitaevDisorderParameter Jtensor(N);
    Jtensor.Jelems[N-4][N-3][N-2][N-1] = 1.;
    test_act_even(N, Jtensor);
}

TEST(CudaHamiltonian, act_even_left4_right0) {
    test_act_even_left4_right0(8);
    test_act_even_left4_right0(10);
    test_act_even_left4_right0(12);
    test_act_even_left4_right0(14);
}

TEST(CudaHamiltonian, act_even_left3_right1) {
    test_act_even_left3_right1(8);
    test_act_even_left3_right1(10);
    test_act_even_left3_right1(12);
    test_act_even_left3_right1(14);
}

TEST(CudaHamiltonian, act_even_left2_right2) {
    test_act_even_left2_right2(8);
    test_act_even_left2_right2(10);
    test_act_even_left2_right2(12);
    test_act_even_left2_right2(14);
}

TEST(CudaHamiltonian, act_even_left1_right3) {
    test_act_even_left1_right3(8);
    test_act_even_left1_right3(10);
    test_act_even_left1_right3(12);
    test_act_even_left1_right3(14);
}

TEST(CudaHamiltonian, act_even_left0_right4) {
    test_act_even_left0_right4(8);
    test_act_even_left0_right4(10);
    test_act_even_left0_right4(12);
    test_act_even_left0_right4(14);
}

TEST(CudaHamiltonian, act_even) {
    test_act_even(8);
    test_act_even(10);
    test_act_even(12);
    test_act_even(14);
}

/////////////////////////////////////////////////////////////

TEST(CudaHamiltonian, act_even_single_chunk_async) {
    int N = 10;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);
    CudaEvenState state(space, gen);
    CudaEvenState out_state(space);

    CudaHamiltonian H(space, Jtensor, default_avail_memory, false);

    cudaStream_t stream = d_create_stream();
    CudaEvent completion;
    H.act_even_single_chunk_async(out_state, state, completion.event);
    completion.wait(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    d_destroy_stream(stream);

    Mat out_state_matrix = out_state.get_matrix();

    // Print and compare
    // cout << "\nState:\n\n" << state.get_matrix() << "\n\n";
    // cout << "\nResult:\n\n" << out_state.get_matrix() << "\n\n";

    Mat expected = Mat::Zero(space.left.D, space.right.D);
    FactorizedHamiltonian host_H(space, Jtensor);
    host_H.act_even(expected, state.get_matrix());

    // cout << "Expected result:\n\n" << expected << "\n\n";

    assert_equal_matrices(expected, out_state_matrix);
}

TEST(CudaHamiltonian, act_even_left4_right0_limited_mem) {
    int N = 14;

    for (size_t available_mem = 40000;
         available_mem < 100000;
         available_mem += 20000) {

        MajoranaKitaevDisorderParameter Jtensor(N);
        Jtensor.Jelems[0][1][2][3] = 1.;
        test_act_even(N, Jtensor, available_mem);
    }
}

TEST(CudaHamiltonian, act_even_left3_right1_limited_mem) {
    int N = 14;

    for (size_t available_mem = 20000;
         available_mem < 100000;
         available_mem += 20000) {

        MajoranaKitaevDisorderParameter Jtensor(N);
        Jtensor.Jelems[0][1][2][N-1] = 1.;
        test_act_even(N, Jtensor, available_mem);
    }
}

TEST(CudaHamiltonian, act_even_left2_right2_limited_mem) {
    int N = 14;

    for (size_t available_mem = 20000;
         available_mem < 100000;
         available_mem += 20000) {

        MajoranaKitaevDisorderParameter Jtensor(N);
        Jtensor.Jelems[0][1][N-2][N-1] = 1.;
        test_act_even(N, Jtensor, available_mem);
    }
}

TEST(CudaHamiltonian, act_even_left1_right3_limited_mem) {
    int N = 14;

    for (size_t available_mem = 20000;
         available_mem < 100000;
         available_mem += 20000) {

        MajoranaKitaevDisorderParameter Jtensor(N);
        Jtensor.Jelems[0][N-3][N-2][N-1] = 1.;
        test_act_even(N, Jtensor, available_mem);
    }
}

TEST(CudaHamiltonian, act_even_left0_right4_limited_mem) {
    int N = 14;

    for (size_t available_mem = 20000;
         available_mem < 100000;
         available_mem += 20000) {
        MajoranaKitaevDisorderParameter Jtensor(N);
        Jtensor.Jelems[N-4][N-3][N-2][N-1] = 1.;
        test_act_even(N, Jtensor, available_mem);
    }
}

TEST(CudaHamiltonian, act_even_limited_mem) {
    int N = 14;

    for (size_t available_mem = 20000;
         available_mem < 100000;
         available_mem += 20000) {
        test_act_even(N, available_mem);
    }
}

///////////////////////////////////////////////////////////////////

TEST(CudaHamiltonian, limited_memory_chunks) {
    int N = 14;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    for (size_t available_memory = 20000;
         available_memory <= 100000;
         available_memory += 20000) {

        CudaHamiltonian H(space, Jtensor, available_memory, false);
        ASSERT_TRUE(H.done_processing_ops());

        // Make sure all operators are accounted for
        map<int, int> total_num_ops;

        for (int i = 0; i <= q; i++) {
            total_num_ops[i] = 0;
        }

        for (int i = 0; i < H.op_chunks.size(); i++) {
            for (int j = 0; j <= q; j++) {
                ASSERT_TRUE(H.op_chunks.at(i)->predicted_d_alloc_size() <
                            available_memory);
                total_num_ops.at(j) +=
                    H.op_chunks.at(i)->num_operators.at(j);
            }
        }

        for (int i = 0; i <= q; i++) {
            ASSERT_EQ(total_num_ops.at(i),
                      num_factorized_hamiltonian_operator_pairs(space, i));
        }

        CudaEvenState state(space, gen);
        CudaEvenState out_state(space);

        // Just make sure it doesn't crash
        H.act_even(out_state, state);
    }
}

TEST(CudaHamiltonian, combining_two_chunks) {
    int N = 14;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    // This is enough memory for 1 big chunk, but this/2 is not enough for 1 chunk.
    // So 2 chunks will be generated and should be combined.
    size_t available_memory = 200000;

    CudaHamiltonian H(space, Jtensor, available_memory, false);
    ASSERT_TRUE(H.done_processing_ops());
    ASSERT_EQ(H.op_chunks.size(), 1);

    // Make sure all operators are accounted for
    for (int i = 0; i <= q; i++) {
        ASSERT_TRUE(H.op_chunks.at(0)->predicted_d_alloc_size() < available_memory);
        ASSERT_EQ(H.op_chunks.at(0)->num_operators.at(i),
                  num_factorized_hamiltonian_operator_pairs(space, i));
    }

    CudaEvenState state(space, gen);
    CudaEvenState out_state(space);

    // Just make sure it doesn't crash
    H.act_even(out_state, state);
}

TEST_F(CudaLanczosTest, compute_two_calls_vs_one) {
    int N = 10;

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaHamiltonian H(space, Jtensor, default_avail_memory);

    RealVec alpha;
    RealVec beta;

    int max_steps = 16;
    int steps = max_steps / 2;
    double mu = 1.;

    // two calls to compute
    CudaLanczos lanczos2(max_steps, initial_state);
    lanczos2.compute(handle, handle_sp, H, mu, steps);
    lanczos2.compute(handle, handle_sp, H, mu, steps);

    RealVec alpha2;
    RealVec beta2;
    lanczos2.read_coeffs(alpha2, beta2);

    // one call to compute
    CudaLanczos lanczos1(max_steps, initial_state);
    lanczos1.compute(handle, handle_sp, H, mu, max_steps);

    RealVec alpha1;
    RealVec beta1;
    lanczos1.read_coeffs(alpha1, beta1);

    assert_equal_vectors(alpha1, alpha2);
    assert_equal_vectors(beta1, beta2);
}

static void test_cuda_lanczos_even_one_call(int N) {
    cublasHandle_t handle = cublas_init();
    cusparseHandle_t handle_sp = cusparse_init();

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);

    // MajoranaKitaevDisorderParameter Jtensor(space.N);
    // Jtensor.Jelems[0][1][2][4] = 1.;
    // Mat state = Mat::Zero(space.left.D, space.right.D);
    // state(0,0) = 1.;

    CudaHamiltonian H(space, Jtensor, default_avail_memory);

    FactorizedHamiltonianGenericParity H_host(space, Jtensor);
    RealVec expected_alpha;
    RealVec expected_beta;

    RealVec alpha;
    RealVec beta;

    // half the space size, beyond that we get a zero beta value,
    // and u vector, and then accuracy is completely lost
    // when computing u/beta
    int steps = initial_state.size() / 4;

    double mu = 1.;

    CudaLanczos lanczos(steps, initial_state);
    lanczos.compute(handle, handle_sp, H, mu, steps);
    lanczos.read_coeffs(alpha, beta);

    ASSERT_EQ(alpha.size(), steps);
    ASSERT_EQ(beta.size(), steps - 1);

    factorized_lanczos(H_host, mu, steps, initial_state.matrix,
                       expected_alpha, expected_beta);

    assert_equal_vectors(expected_alpha, alpha, "ALPHA");
    assert_equal_vectors(expected_beta, beta, "BETA");

    cublas_destroy(handle);
    cusparse_destroy(handle_sp);
}

static void test_cuda_lanczos_even_two_calls(int N) {
    cublasHandle_t handle = cublas_init();
    cusparseHandle_t handle_sp = cusparse_init();

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);

    // MajoranaKitaevDisorderParameter Jtensor(space.N);
    // Jtensor.Jelems[0][1][2][4] = 1.;
    // Mat state = Mat::Zero(space.left.D, space.right.D);
    // state(0,0) = 1.;

    CudaHamiltonian H(space, Jtensor, default_avail_memory);

    FactorizedHamiltonianGenericParity H_host(space, Jtensor);
    RealVec expected_alpha;
    RealVec expected_beta;

    RealVec alpha;
    RealVec beta;

    // half the space size, beyond that we get a zero beta value,
    // and u vector, and then accuracy is completely lost
    // when computing u/beta
    int max_steps = initial_state.size() / 2;
    assert(max_steps % 2 == 0);

    int steps = max_steps / 2;
    double mu = 1.;

    CudaLanczos lanczos(max_steps, initial_state);
    lanczos.compute(handle, handle_sp, H, mu, steps);
    lanczos.read_coeffs(alpha, beta);

    ASSERT_EQ(alpha.size(), steps);
    ASSERT_EQ(beta.size(), steps - 1);

    factorized_lanczos(H_host, mu, steps, initial_state.matrix,
                        expected_alpha, expected_beta);

    assert_equal_vectors(expected_alpha, alpha);
    assert_equal_vectors(expected_beta, beta);

    // compute the rest of the steps
    lanczos.compute(handle, handle_sp, H, mu, steps);

    lanczos.read_coeffs(alpha, beta);

    ASSERT_EQ(alpha.size(), max_steps);
    ASSERT_EQ(beta.size(), max_steps - 1);

    factorized_lanczos(H_host, mu, max_steps, initial_state.matrix,
                        expected_alpha, expected_beta);

    assert_equal_vectors(expected_alpha, alpha);
    assert_equal_vectors(expected_beta, beta);

    cublas_destroy(handle);
    cusparse_destroy(handle_sp);
}

TEST(CudaLanczosTestNoF, compute_even_one_call_8) {
    test_cuda_lanczos_even_one_call(8);
}

TEST(CudaLanczosTestNoF, compute_even_one_call_10) {
    test_cuda_lanczos_even_one_call(10);
}

TEST(CudaLanczosTestNoF, compute_even_one_call_12) {
    test_cuda_lanczos_even_one_call(12);
}

TEST(CudaLanczosTestNoF, compute_even_one_call_14) {
    test_cuda_lanczos_even_one_call(14);
}

TEST(CudaLanczosTestNoF, compute_even_two_calls) {
    test_cuda_lanczos_even_two_calls(10);
    test_cuda_lanczos_even_two_calls(12);
    test_cuda_lanczos_even_two_calls(14);
}

TEST_F(CudaLanczosTest, load_and_save) {
    int N = 6;

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaHamiltonian H(space, Jtensor, default_avail_memory);

    RealVec alpha;
    RealVec beta;

    int max_steps = 4;
    double mu = 1.;

    CudaLanczos lanczos(max_steps, initial_state);

    // cout << "\n=== Reference ===" << endl;
    lanczos.compute(handle, handle_sp, H, mu, max_steps / 2);

    string filename = "/tmp/syk-test-gpu-load-and-save";
    lanczos.save_state(filename);

    RealVec exp_alpha, exp_beta;

    // Load with no computation
    // cout << "\nComparison 0" << endl;
    CudaLanczos lanczos0(space, filename, max_steps);

    RealVec actual_alpha0, actual_beta0;
    lanczos0.read_coeffs(actual_alpha0, actual_beta0);
    lanczos.read_coeffs(exp_alpha, exp_beta);

    ASSERT_EQ(lanczos.current_step(), lanczos0.current_step());
    ASSERT_EQ(lanczos.max_steps(), lanczos0.max_steps());
    assert_equal_vectors(exp_alpha, actual_alpha0, "comparison 0 alpha");
    assert_equal_vectors(exp_beta, actual_beta0, "comparison 0 beta");

    lanczos.compute(handle, handle_sp, H, mu, max_steps / 2);
    lanczos.read_coeffs(exp_alpha, exp_beta);

    // Load from save with same max_steps
    // cout << "\n=== Comparison 1 ===" << endl;
    CudaLanczos lanczos1(space, filename, max_steps);

    RealVec actual_alpha1, actual_beta1;
    lanczos1.read_coeffs(actual_alpha1, actual_beta1);

    lanczos1.compute(handle, handle_sp, H, mu, max_steps / 2);
    lanczos1.read_coeffs(actual_alpha1, actual_beta1);

    ASSERT_EQ(lanczos.current_step(), lanczos1.current_step());
    ASSERT_EQ(lanczos.max_steps(), lanczos1.max_steps());
    assert_equal_vectors(exp_alpha, actual_alpha1, "comparison 1 alpha");
    assert_equal_vectors(exp_beta, actual_beta1, "comparison 1 beta");

    // Load from save with bigger max_steps
    // cout << "\nComparison 2" << endl;
    CudaLanczos lanczos2(space, filename, max_steps * 2);
    lanczos2.compute(handle, handle_sp, H, mu, max_steps / 2);

    RealVec actual_alpha2, actual_beta2;
    lanczos2.read_coeffs(actual_alpha2, actual_beta2);

    ASSERT_EQ(lanczos.current_step(), lanczos2.current_step());
    assert_equal_vectors(exp_alpha, actual_alpha2, "comparison 2 alpha");
    assert_equal_vectors(exp_beta, actual_beta2, "comparison 2 beta");
}

static void full_lanczos_test(
    int N,
    double mu,
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    size_t available_memory = default_avail_memory) {

    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    // This is 2 times bigger than the space, but we need it to get
    // all the eigenvalues
    int m = 2 * space.D;

    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    // Reference diagonzliation
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize();
    RealVec true_evs = H.even_charge_parity_evs;
    true_evs += RealVec::Constant(true_evs.size(), mu);
    true_evs = get_unique_elements(true_evs, epsilon);

    // Lanczos
    CudaHamiltonian cudaH(space, Jtensor, available_memory, false);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaLanczos lanczos(m, initial_state);
    lanczos.compute(handle, handle_sp, cudaH, mu, m);

    RealVec lanczos_evs = lanczos.compute_eigenvalues();

    // print_lanczos_results(true_evs, lanczos_evs);

    verify_same_evs(true_evs, lanczos_evs);
}

TEST_F(CudaLanczosTest, full_lanczos_process_10) {
    full_lanczos_test(10, 0., handle, handle_sp);
    full_lanczos_test(10, 1., handle, handle_sp);
}

TEST_F(CudaLanczosTest, full_lanczos_process_12) {
    full_lanczos_test(12, 0., handle, handle_sp);
    full_lanczos_test(12, 1., handle, handle_sp);
}

TEST_F(CudaLanczosTest, full_lanczos_process_14) {
    full_lanczos_test(14, 0., handle, handle_sp);
    full_lanczos_test(14, 1., handle, handle_sp);
}

TEST_F(CudaLanczosTest, full_lanczos_process_limited_memory_14) {
    full_lanczos_test(14, 0., handle, handle_sp, 40000);
    full_lanczos_test(14, 0., handle, handle_sp, 100000);
}

void full_lanczos_test_with_errs(int N,
                                 double mu,
                                 cublasHandle_t handle,
                                 cusparseHandle_t handle_sp) {
    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    // Reference diagonzliation
    MajoranaKitaevHamiltonian H(N, &Jtensor);
    H.diagonalize();

    RealVec unique_evs = H.even_charge_parity_evs;
    unique_evs += RealVec::Constant(unique_evs.size(), mu);
    unique_evs = get_unique_elements(unique_evs, epsilon);

    // Lanczos
    CudaHamiltonian cudaH(space, Jtensor, default_avail_memory);

    int max_steps = space.D * 2;
    int iteration_steps = 10;

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaLanczos lanczos(max_steps, initial_state);

    for (int m = iteration_steps; m < max_steps; m += iteration_steps) {
        // cout << "\n\n*************  m = " << m << "  *****************\n";
        lanczos.compute(handle, handle_sp, cudaH, mu, iteration_steps);

        RealVec errs;
        RealVec lanczos_evs = lanczos.compute_eigenvalues(errs, gen);

        map<int, int> nearest = find_nearest_true_ev(unique_evs,
                                                     lanczos_evs);

        // Verify the error estimates over-estimate the true errors
        for (map<int,int>::iterator iter = nearest.begin();
             iter != nearest.end();
             ++iter) {

            double true_ev = unique_evs(iter->first);
            double lanczos_ev = lanczos_evs(iter->second);
            double err_estimate = errs(iter->second);
            double true_err = abs(true_ev - lanczos_ev);

            if (err_estimate < epsilon) {
                // If the error estimate is tiny, make sure the true
                // error is tiny (it doesn't make sense to compare them
                // because they are both essentially zero)
                ASSERT_LT(true_err, epsilon);
            }
            else {
                // Errors are larger, so make sure the estimate gives an
                // upper bound
                ASSERT_LT(true_err, err_estimate);
            }
        }

        // print_lanczos_results(unique_evs, lanczos_evs, errs);
    }
}

TEST_F(CudaLanczosTest, full_lanczos_process_with_errs_10) {
    full_lanczos_test_with_errs(10, 0., handle, handle_sp);
    full_lanczos_test_with_errs(10, 1., handle, handle_sp);
}

TEST_F(CudaLanczosTest, full_lanczos_process_with_errs_12) {
    full_lanczos_test_with_errs(12, 0., handle, handle_sp);
    full_lanczos_test_with_errs(12, 1., handle, handle_sp);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
