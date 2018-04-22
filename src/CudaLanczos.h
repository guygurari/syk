#ifndef __CUDA_LANCZOS_H__
#define __CUDA_LANCZOS_H__

#include <cublas_v2.h>

#include "defs.h"
#include "eigen_utils.h"
#include "CudaHamiltonian.h"

class CudaLanczos {
public:
    // Only even parity states are supported
    CudaLanczos(int max_steps,
                const FactorizedParityState& initial_state);

    // Load the state from the given file
    CudaLanczos(FactorizedSpace& space, string filename, int max_steps);

    ~CudaLanczos();

    // Initialize memory for an even parity state
    /* void init_even(int max_steps, const Mat& initial_even_state); */

    // Run some Lanczos steps. State is saved, and can
    // be resumed by calling this function again.
    // Total number of steps cannot exceed max_steps.
    void compute(cublasHandle_t handle,
                 cusparseHandle_t handle_sp,
                 CudaHamiltonianInterface& H,
                 double mu,
                 int steps);

    void read_coeffs(RealVec& alpha, RealVec& beta);

    // Computes the 'good' eigenvalues from the existing Lanczos
    // coefficients. This is done on the host.
    RealVec compute_eigenvalues();

    // Computes the 'good' eigenvalues and their error estimates.
    // The estimates give an upper bound on difference between Lanczos
    // eigenvalues and actual eigenvalues.
    RealVec compute_eigenvalues(RealVec& error_estimates,
                                boost::random::mt19937* gen);

    void save_state(string filename);

    static bool get_state_info(string filename, int& num_steps, int& max_steps);

    int current_step();
    int max_steps();

    FactorizedSpace space;
    Mat zero_state;

    // amount of device allocated memory
    size_t d_alloc_size;

    // the next iteration to be executed
    int iter;

    // max number of steps allocated for.
    // also size of allocated d_alpha.
    int m;

    // These are complex for simplicity in cuBLAS operations
    cucpx* d_alpha;
    cucpx* d_beta;
    double* d_s;

    CudaEvenState* vi_minus_1;
    CudaEvenState* vi;
    CudaEvenState* u;
    CudaEvenState* work;

    CudaScalar minus_one;

private:
    // Number of elements allocated in the alpha, beta vectors
    // (For beta, this is bigger than the number of useful elements.
    // The first element vanishes and the last one is unused.)
    int alpha_size();
    int beta_size();

    int alpha_alloc_size();
    int beta_alloc_size();

    void read_coeffs(RealVec& alpha, RealVec& beta,
                     bool read_extended_beta);

    size_t alloc_states();
    size_t alloc_coeffs();
    void load_state(string filename, int max_steps);
};

#endif
