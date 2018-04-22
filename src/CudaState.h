#ifndef __CUDA_STATE_H__
#define __CUDA_STATE_H__

#include <boost/random/mersenne_twister.hpp>

#include "defs.h"
#include "eigen_utils.h"
#include "FactorizedSpaceUtils.h"
#include "CudaUtils.h"

//
// A state with even charge parity, stored on the GPU.
// 
// We only store the non-zero blocks. They are stored contiguously
// in memory, so one can compute a dot product of two states
// by treating the two blocks as one long vector.
//
class CudaEvenState {
public:
    // An uninitialized state (useful as output)
    CudaEvenState(FactorizedSpace& space);

    // Initialize a random state
    CudaEvenState(FactorizedSpace& space, boost::random::mt19937* gen);

    // Initialize a state from a given matrix
    CudaEvenState(FactorizedSpace& space, const Mat& matrix);

    CudaEvenState(const CudaEvenState& other);

    ~CudaEvenState();

    void destroy();

    // The dimension of the state
    int size();

    // Copies the state matrix from the device and returns it
    Mat get_matrix();

    // Set the state to the given matrix, including copying to the
    // device
    void set(const Mat& _matrix);

    // Set this state to other 
    void set(const CudaEvenState& other);

    // Async version of set()
    void set_async(const CudaEvenState& other, cudaStream_t stream);

    // Set this state to zero
    void set_to_zero();

    // this = this + alpha * other
    // alpha points to a scalar in host or device memory
    void add(cublasHandle_t handle,
             cucpx* alpha,
             CudaEvenState& other);

    // Compute the dot product of this state with another (including
    // conjugation), and write the result to 'result' (a device pointer). 
    void dot(cublasHandle_t handle,
             cucpx* result,
             CudaEvenState& other);

    // Compute the 2-norm of the state into result
    void norm(cublasHandle_t handle, double* result);

    // Rescale the state by alpha
    void scale(cublasHandle_t handle, double* alpha);

    FactorizedSpace space;

    int block_rows;
    int block_cols;
    int block_size; // total block elements = rows * columns
    int block_alloc_size; // how much memory each block takes

    int device_id;

    // Blocks on the device
    cucpx* d_top_left_block;
    cucpx* d_bottom_right_block;

private:
    void init(const FactorizedSpace& space);
};


#endif
