////////////////////////////////////////////////////////////////////////
//
// These tests should be run on a node with 2 GPUs.
//
////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "CudaUtils.h"
#include "TestUtils.h"
#include "FactorizedSpaceUtils.h"
#include "Lanczos.h"

#include "CudaState.h"
#include "CudaHamiltonian.h"
#include "CudaMultiGpuHamiltonian.h"
#include "CudaMultiGpuHamiltonianNaive.h"
#include "CudaLanczos.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

boost::random::mt19937* gen = new boost::random::mt19937(time(0));

TEST(CudaMultiGpuHamiltonianTest, sanity) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    FactorizedSpace space = FactorizedSpace::from_majorana(30);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    size_t mem_per_device = 50000000;
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonian H(space, Jtensor, memory);
}

TEST(CudaMultiGpuHamiltonianTest, sanity_and_act) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    FactorizedSpace space = FactorizedSpace::from_majorana(30);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);
    CudaEvenState state(space, gen);
    CudaEvenState out_state(space);

    size_t mem_per_device = 50000000;
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonian H(space, Jtensor, memory);
    H.act_even(out_state, state);
    H.act_even(out_state, state);
    H.act_even(out_state, state);
}

TEST(CudaStateTest, allocation) {
    cuda_set_device(0);
    FactorizedSpace space = FactorizedSpace::from_majorana(10);
    CudaEvenState* state = new CudaEvenState(space);
    cuda_set_device(1);
    delete state;
}

TEST(CudaResourceManagerTest, allocation) {
    cuda_set_device(0);
    CudaResourceManager* manager = new CudaResourceManager();
    manager->get_handles();
    manager->get_stream();
    cuda_set_device(1);
    delete manager;
}

/////////////////////////////////////////////////////////////

void test_act_even_2devices(int N, size_t mem_per_device) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    cuda_set_device(0);
    CudaEvenState state(space, gen);
    CudaEvenState state_copy(state);
    CudaEvenState out_state(space);
    CudaEvenState out_state2(space);

    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonian H(space, Jtensor, memory);
    H.act_even(out_state, state);

    // checkCudaErrors(cudaDeviceSynchronize());

    Mat out_state_matrix = out_state.get_matrix();

    // Print and compare
    // cout << "\nState:\n\n" << state.get_matrix() << "\n\n";
    // cout << "\nResult:\n\n" << out_state.get_matrix() << "\n\n";

    Mat expected = Mat::Zero(space.left.D, space.right.D);
    FactorizedHamiltonian host_H(space, Jtensor);
    host_H.act_even(expected, state.get_matrix());

    // cout << "Expected result:\n\n" << expected << "\n\n";

    assert_equal_matrices(expected, out_state_matrix);
    assert_equal_matrices(state.get_matrix(), state_copy.get_matrix());

    // Act again
    H.act_even(out_state2, out_state);
    out_state_matrix = out_state2.get_matrix();

    Mat expected2 = Mat::Zero(space.left.D, space.right.D);
    host_H.act_even(expected2, expected);
    assert_equal_matrices(expected2, out_state_matrix);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_22_2devices) {
    test_act_even_2devices(22, 1000000);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_24_2devices) {
    test_act_even_2devices(24, 1000000);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_combine_two_chunks_2devices) {
    // Here a Hamiltonian ends up with 2 chunks and must combine them into
    // one
    test_act_even_2devices(18, 500000);
}

void test_act_even_allDevices(int N, size_t mem_per_device) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    cuda_set_device(0);
    CudaEvenState state(space, gen);
    CudaEvenState state_copy(state);
    CudaEvenState out_state(space);
    CudaEvenState out_state2(space);

    vector<size_t> memory;

    for (int i = 0; i < cuda_get_num_devices(); i++) {
        memory.push_back(mem_per_device);
    }

    CudaMultiGpuHamiltonian H(space, Jtensor, memory);
    H.act_even(out_state, state);

    // checkCudaErrors(cudaDeviceSynchronize());

    Mat out_state_matrix = out_state.get_matrix();

    // Print and compare
    // cout << "\nState:\n\n" << state.get_matrix() << "\n\n";
    // cout << "\nResult:\n\n" << out_state.get_matrix() << "\n\n";

    Mat expected = Mat::Zero(space.left.D, space.right.D);
    FactorizedHamiltonian host_H(space, Jtensor);
    host_H.act_even(expected, state.get_matrix());

    // cout << "Expected result:\n\n" << expected << "\n\n";

    assert_equal_matrices(expected, out_state_matrix);
    assert_equal_matrices(state.get_matrix(), state_copy.get_matrix());

    // Act again
    H.act_even(out_state2, out_state);
    out_state_matrix = out_state2.get_matrix();

    Mat expected2 = Mat::Zero(space.left.D, space.right.D);
    host_H.act_even(expected2, expected);
    assert_equal_matrices(expected2, out_state_matrix);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_22_allDevices) {
    test_act_even_allDevices(22, 1000000);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_24_allDevices) {
    test_act_even_allDevices(24, 1000000);
}

TEST(CudaMultiGpuHamiltonianTest, act_even_combine_two_chunks_allDevices) {
    // Here a Hamiltonian ends up with 2 chunks and must combine them into
    // one
    test_act_even_allDevices(18, 500000);
}

static void full_lanczos_test_multi_gpu_2devices(
    int N, size_t mem_per_device) {

    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    CudaResourceManager cuda_manager;
    CudaHandles handles = cuda_manager.get_handles();

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    double mu = 0.;

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
    vector<size_t> memory;

    for (int i = 0; i < cuda_get_num_devices(); i++) {
        memory.push_back(mem_per_device);
    }

    CudaMultiGpuHamiltonian cudaH(space, Jtensor, memory);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaLanczos lanczos(m, initial_state);
    lanczos.compute(handles.cublas_handle, handles.cusparse_handle,
                    cudaH, mu, m);

    RealVec lanczos_evs = lanczos.compute_eigenvalues();

    // print_lanczos_results(true_evs, lanczos_evs);

    verify_same_evs(true_evs, lanczos_evs);
}

TEST(CudaMultiGpuHamiltonianTest, full_lanczos_16_2devices) {
    full_lanczos_test_multi_gpu_2devices(16, 300000);
}

TEST(CudaMultiGpuHamiltonianTest, full_lanczos_18_2devices) {
    full_lanczos_test_multi_gpu_2devices(18, 500000);
}

static void full_lanczos_test_multi_gpu_allDevices(
    int N, size_t mem_per_device) {

    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    CudaResourceManager cuda_manager;
    CudaHandles handles = cuda_manager.get_handles();

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    double mu = 0.;

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
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonian cudaH(space, Jtensor, memory);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaLanczos lanczos(m, initial_state);
    lanczos.compute(handles.cublas_handle, handles.cusparse_handle,
                    cudaH, mu, m);

    RealVec lanczos_evs = lanczos.compute_eigenvalues();

    // print_lanczos_results(true_evs, lanczos_evs);

    verify_same_evs(true_evs, lanczos_evs);
}

TEST(CudaMultiGpuHamiltonianTest, full_lanczos_16_allDevices) {
    full_lanczos_test_multi_gpu_allDevices(16, 300000);
}

TEST(CudaMultiGpuHamiltonianTest, full_lanczos_18_allDevices) {
    full_lanczos_test_multi_gpu_allDevices(18, 500000);
}

TEST(CudaMultiGpuHamiltonianTest, alloc_and_memset) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);

    size_t alloc_size = 1024;

    cuda_set_device(0);
    cucpx* p0 = d_alloc(alloc_size);

    cuda_set_device(1);
    cucpx* p1 = d_alloc(alloc_size);

    cuda_set_device(0);
    checkCudaErrors(cudaMemset(p0, 0, alloc_size));

    cuda_set_device(1);
    checkCudaErrors(cudaMemset(p1, 0, alloc_size));
    checkCudaErrors(cudaMemset(p1, 0, alloc_size));

    // Cannot have pointer pointing to different device
    // cuda_set_device(0);
    // checkCudaErrors(cudaMemset(p1, 0, alloc_size));

    // cuda_set_device(1);
    // checkCudaErrors(cudaMemset(p0, 0, alloc_size));

    d_free(p0);
    d_free(p1);

    cuda_set_device(0);
}

//////////////////////////////// Naive implementation //////////////////////////

TEST(CudaMultiGpuHamiltonianNaiveTest, sanity) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    FactorizedSpace space = FactorizedSpace::from_majorana(30);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);

    size_t mem_per_device = 50000000;
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonianNaive H(space, Jtensor, memory);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, sanity_and_act) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    FactorizedSpace space = FactorizedSpace::from_majorana(30);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., gen);
    CudaEvenState state(space, gen);
    CudaEvenState out_state(space);

    size_t mem_per_device = 50000000;
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonianNaive H(space, Jtensor, memory);
    H.act_even(out_state, state);
    H.act_even(out_state, state);
    H.act_even(out_state, state);
}

void test_act_even_naive(int N, size_t mem_per_device) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    MajoranaKitaevDisorderParameter Jtensor(N, 1., gen);

    cuda_set_device(0);
    CudaEvenState state(space, gen);
    CudaEvenState state_copy(state);
    CudaEvenState out_state(space);
    CudaEvenState out_state2(space);

    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonianNaive H(space, Jtensor, memory);
    H.act_even(out_state, state);

    // checkCudaErrors(cudaDeviceSynchronize());

    Mat out_state_matrix = out_state.get_matrix();

    // Print and compare
    // cout << "\nState:\n\n" << state.get_matrix() << "\n\n";
    // cout << "\nResult:\n\n" << out_state.get_matrix() << "\n\n";

    Mat expected = Mat::Zero(space.left.D, space.right.D);
    FactorizedHamiltonian host_H(space, Jtensor);
    host_H.act_even(expected, state.get_matrix());

    // cout << "Expected result:\n\n" << expected << "\n\n";

    assert_equal_matrices(expected, out_state_matrix);
    assert_equal_matrices(state.get_matrix(), state_copy.get_matrix());

    // Act again
    H.act_even(out_state2, out_state);
    out_state_matrix = out_state2.get_matrix();

    Mat expected2 = Mat::Zero(space.left.D, space.right.D);
    host_H.act_even(expected2, expected);
    assert_equal_matrices(expected2, out_state_matrix);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, act_even_22) {
    test_act_even_naive(22, 1000000);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, act_even_24) {
    test_act_even_naive(24, 1000000);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, act_even_two_remaining_chunks) {
    // In this case the last Hamiltonian has two remaining chunks,
    // and they need to be combined into one
    test_act_even_naive(18, 500000);
}

static void full_lanczos_test_multi_gpu_naive(
    int N, size_t mem_per_device) {

    ASSERT_TRUE(cuda_get_num_devices() > 1);
    cuda_set_device(0);

    CudaResourceManager cuda_manager;
    CudaHandles handles = cuda_manager.get_handles();

    FactorizedSpace space = FactorizedSpace::from_majorana(N);
    double mu = 0.;

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
    vector<size_t> memory;
    memory.push_back(mem_per_device);
    memory.push_back(mem_per_device);

    CudaMultiGpuHamiltonianNaive cudaH(space, Jtensor, memory);

    FactorizedParityState initial_state(space, EVEN_CHARGE, gen);
    CudaLanczos lanczos(m, initial_state);
    lanczos.compute(handles.cublas_handle, handles.cusparse_handle,
                    cudaH, mu, m);

    RealVec lanczos_evs = lanczos.compute_eigenvalues();

    // print_lanczos_results(true_evs, lanczos_evs);

    verify_same_evs(true_evs, lanczos_evs);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, full_lanczos_16) {
    full_lanczos_test_multi_gpu_naive(16, 300000);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, full_lanczos_18) {
    full_lanczos_test_multi_gpu_naive(18, 500000);
}

TEST(CudaMultiGpuHamiltonianNaiveTest, alloc_and_memset) {
    ASSERT_TRUE(cuda_get_num_devices() > 1);

    size_t alloc_size = 1024;

    cuda_set_device(0);
    cucpx* p0 = d_alloc(alloc_size);

    cuda_set_device(1);
    cucpx* p1 = d_alloc(alloc_size);

    cuda_set_device(0);
    checkCudaErrors(cudaMemset(p0, 0, alloc_size));

    cuda_set_device(1);
    checkCudaErrors(cudaMemset(p1, 0, alloc_size));
    checkCudaErrors(cudaMemset(p1, 0, alloc_size));

    // Cannot have pointer pointing to different device
    // cuda_set_device(0);
    // checkCudaErrors(cudaMemset(p1, 0, alloc_size));

    // cuda_set_device(1);
    // checkCudaErrors(cudaMemset(p0, 0, alloc_size));

    d_free(p0);
    d_free(p1);

    cuda_set_device(0);
}

/////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
