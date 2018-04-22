#include <iostream>
#include <boost/random/normal_distribution.hpp>
#include "CudaState.h"
#include "CudaUtils.h"

using namespace std;

CudaEvenState::CudaEvenState(FactorizedSpace& _space) {
    init(_space);
}

CudaEvenState::CudaEvenState(
    FactorizedSpace& _space, boost::random::mt19937* gen) {

    init(_space);
    set(get_factorized_random_state(space, EVEN_CHARGE, gen));
}

CudaEvenState::CudaEvenState(
    FactorizedSpace& _space, const Mat& matrix) {

    init(_space);
    set(matrix);
}

CudaEvenState::CudaEvenState(const CudaEvenState& other) {
    init(other.space);
    set(other);
}

CudaEvenState::~CudaEvenState() {
    destroy();
}

void CudaEvenState::init(const FactorizedSpace& _space) {
    space = _space;
    block_rows = space.left.D / 2;
    block_cols = space.right.D / 2;
    block_size = block_rows * block_cols;
    block_alloc_size = sizeof(cucpx) * block_size;

    device_id = cuda_get_device();

    // Store one big chunk of memory for both blocks
    d_top_left_block = d_alloc(block_alloc_size * 2);
    d_bottom_right_block = d_top_left_block + block_size;
}

void CudaEvenState::destroy() {
    if (d_top_left_block == 0) {
        return;
    }

    cuda_set_device(device_id);
    
    // Free only once because allocated only once
    d_free(d_top_left_block);

    d_top_left_block = 0;
    d_bottom_right_block = 0;
}

int CudaEvenState::size() {
    return block_rows * block_cols * 2;
}

void CudaEvenState::set(const Mat& matrix) {
    assert(matrix.rows() == block_rows * 2);
    assert(matrix.cols() == block_cols * 2);

    Mat top = matrix.block(0, 0, block_rows, block_cols);
    Mat bottom = matrix.block(block_rows, block_cols,
                              block_rows, block_cols);

    copy_matrix_host_to_device(d_top_left_block, top);
    copy_matrix_host_to_device(d_bottom_right_block, bottom);
}

void CudaEvenState::set(const CudaEvenState& other) {
    assert(block_size == other.block_size);
    
    checkCudaErrors(cudaMemcpy(d_top_left_block,
                               other.d_top_left_block,
                               sizeof(cucpx) * block_size * 2,
                               cudaMemcpyDeviceToDevice));
}

void CudaEvenState::set_async(const CudaEvenState& other, cudaStream_t stream) {
    assert(block_size == other.block_size);
    
    checkCudaErrors(cudaMemcpyAsync(d_top_left_block,
                                    other.d_top_left_block,
                                    sizeof(cucpx) * block_size * 2,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
}

void CudaEvenState::set_to_zero() {
    checkCudaErrors(cudaMemset(d_top_left_block, 0,
                               sizeof(cucpx) * block_size));

    checkCudaErrors(cudaMemset(d_bottom_right_block, 0,
                               sizeof(cucpx) * block_size));
}

void CudaEvenState::add(cublasHandle_t handle,
                                  cucpx* alpha,
                                  CudaEvenState& other) {
    // Copy both blocks at once (they are contiguous)
    add_vector_times_scalar(handle,
                            d_top_left_block,
                            alpha,
                            other.d_top_left_block,
                            block_size * 2);
}

// Compute the dot product of this state with another (including
// conjugation), and write the result to 'result' (a device vector). 
void CudaEvenState::dot(cublasHandle_t handle,
                                  cucpx* result,
                                  CudaEvenState& other) {
    // The blocks are stored contiguously, so we can do just one product
    assert(other.block_rows == block_rows);
    assert(other.block_cols == block_cols);

    checkCublasErrors(cublasZdotc(handle,
                                  block_size * 2, // num elements
                                  d_top_left_block, 1,
                                  other.d_top_left_block, 1,
                                  result));
}

void CudaEvenState::norm(cublasHandle_t handle,
                                   double* result) {

    checkCublasErrors(cublasDznrm2(handle,
                                   block_size * 2, // num elements
                                   d_top_left_block, 1,
                                   result));
}

void CudaEvenState::scale(cublasHandle_t handle,
                                    double* alpha) {
    checkCublasErrors(cublasZdscal(handle,
                                   block_size * 2,
                                   alpha,
                                   d_top_left_block, 1));
}

Mat CudaEvenState::get_matrix() {
    Mat matrix = Mat::Zero(block_rows * 2, block_cols * 2);
        
    // Top-left block
    matrix.block(0, 0, block_rows, block_cols) =
        copy_matrix_device_to_host(d_top_left_block,
                                   block_rows,
                                   block_cols);

    // Bottom-right block
    matrix.block(block_rows, block_cols,
                 block_rows, block_cols) = 
        copy_matrix_device_to_host(d_bottom_right_block,
                                   block_rows,
                                   block_cols);

    return matrix;
}
