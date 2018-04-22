#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include "defs.h"
#include "eigen_utils.h"
#include "CudaHamiltonian.h"
#include "CudaState.h"
#include "CudaLanczos.h"
#include "Lanczos.h"

using namespace std;

CudaLanczos::CudaLanczos(int max_steps,
                         const FactorizedParityState& initial_state)
    : minus_one(-1., 0.) {

    assert(initial_state.charge_parity == EVEN_CHARGE);

    space = initial_state.space;
    iter = 1;
    m = max_steps;

    d_alloc_size = 0;
    d_alloc_size += alloc_states();
    d_alloc_size += alloc_coeffs();
    vi->set(initial_state.matrix);
}

CudaLanczos::CudaLanczos(FactorizedSpace& _space,
                         string filename,
                         int max_steps)
    : minus_one(-1., 0.) {

    d_alloc_size = 0;
    space = _space;
    alloc_states();
    load_state(filename, max_steps);
}

size_t CudaLanczos::alloc_states() {
    zero_state = Mat::Zero(space.left.D, space.right.D);

    vi_minus_1 = new CudaEvenState(space, zero_state);
    vi = new CudaEvenState(space, zero_state);
    u = new CudaEvenState(space, zero_state);
    work = new CudaEvenState(space, zero_state);

    d_s = (double*) d_alloc(sizeof(double));

    return 4 * space.state_alloc_size() + sizeof(double);
}

CudaLanczos::~CudaLanczos() {
    delete vi_minus_1;
    delete vi;
    delete u;
    delete work;

    if (d_alpha != 0) {
        d_free(d_alpha);
    }

    if (d_beta != 0) {
        d_free(d_beta);
    }

    d_free(d_s);
}

int CudaLanczos::alpha_size() {
    return m;
}

int CudaLanczos::beta_size() {
    return m + 1;
}

int CudaLanczos::alpha_alloc_size() {
    return sizeof(cucpx) * alpha_size();
}

int CudaLanczos::beta_alloc_size() {
    return sizeof(cucpx) * beta_size();
}

size_t CudaLanczos::alloc_coeffs() {
    d_alpha = d_alloc(alpha_alloc_size());
    d_beta = d_alloc(beta_alloc_size());

    checkCudaErrors(cudaMemset(d_alpha, 0, alpha_alloc_size()));
    checkCudaErrors(cudaMemset(d_beta,  0, beta_alloc_size()));

    return alpha_alloc_size() + beta_alloc_size();
}

void CudaLanczos::read_coeffs(RealVec& alpha, RealVec& beta) {
    // Skip the last element of beta
    read_coeffs(alpha, beta, false);
}

void CudaLanczos::read_coeffs(RealVec& alpha, RealVec& beta,
                              bool read_extended_beta) {
    assert(m > 0); // space allocated
    assert(iter > 1); // at least one step was run

    int alpha_size = iter - 1;
    int beta_size = read_extended_beta ? alpha_size : alpha_size-1;

    // Copy the real parts of alpha, beta
    alpha = RealVec(alpha_size);
    beta = RealVec(beta_size);

    checkCublasErrors(cublasGetVector(
                          alpha_size, sizeof(double),
                          d_alpha, 2,
                          alpha.data(), 1));
    
    // For beta, we skip the first element. The last element
    // is read if read_extended_beta=true.
    checkCublasErrors(cublasGetVector(
                          beta_size, sizeof(double),
                          d_beta + 1, 2,
                          beta.data(), 1));
}

RealVec CudaLanczos::compute_eigenvalues() {
    RealVec alpha;
    RealVec beta;
    read_coeffs(alpha, beta);
    return find_good_lanczos_evs(alpha, beta);
}

RealVec CudaLanczos::compute_eigenvalues(RealVec& error_estimates,
                                         boost::random::mt19937* gen) {
    RealVec alpha;
    RealVec extended_beta;
    read_coeffs(alpha, extended_beta, true);
    return find_good_lanczos_evs_and_errs(alpha, extended_beta,
                                          error_estimates, gen);
}

void CudaLanczos::save_state(string filename) {
    Mat vi_matrix = vi->get_matrix();
    Mat vi_minus_1_matrix = vi_minus_1->get_matrix();

    Vec alpha(alpha_size());
    Vec beta(beta_size());

    checkCudaErrors(cudaMemcpy(alpha.data(), d_alpha,
                               sizeof(cucpx) * alpha_size(),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(beta.data(), d_beta,
                               sizeof(cucpx) * beta_size(),
                               cudaMemcpyDeviceToHost));

    ofstream out(filename.c_str(), ofstream::binary);
    // out << setprecision(precision);

    out.write((char*) (&iter), sizeof(iter));
    out.write((char*) (&m), sizeof(m));
    out.write((char*) (&d_alloc_size), sizeof(d_alloc_size));

    write_matrix_binary(out, vi_matrix);
    write_matrix_binary(out, vi_minus_1_matrix);
    write_vector_binary(out, alpha);
    write_vector_binary(out, beta);

    out.close();
}

bool CudaLanczos::get_state_info(string filename,
                                 int& num_steps, 
                                 int& max_steps) {
    ifstream in(filename.c_str(), ifstream::binary);

    if (!in.is_open()) {
        return false;
    }

    int iter;
    in.read((char*) (&iter), sizeof(iter));
    num_steps = iter - 1;

    in.read((char*) (&max_steps), sizeof(max_steps));

    return !in.eof();
}

void CudaLanczos::load_state(string filename, int max_steps) {
    ifstream in(filename.c_str(), ifstream::binary);
    assert(in.is_open());

    // read from the file to host memory
    in.read((char*) (&iter), sizeof(iter));
    in.read((char*) (&m), sizeof(m));
    in.read((char*) (&d_alloc_size), sizeof(d_alloc_size));

    Mat vi_matrix = read_matrix_binary(in);
    Mat vi_minus_1_matrix = read_matrix_binary(in);
    Vec alpha = read_vector_binary(in);
    Vec beta = read_vector_binary(in);

    assert(in.good());
    assert(alpha.size() == alpha_size());
    assert(beta.size() == beta_size());

    // we may be asked to increase the number of max_steps
    assert(alpha.size() <= max_steps);
    m = max_steps;

    // copy states to device
    vi->set(vi_matrix);
    vi_minus_1->set(vi_minus_1_matrix);

    // allocate alpha, beta and copy to device
    alloc_coeffs();

    checkCudaErrors(cudaMemcpy(d_alpha, alpha.data(),
                               sizeof(cucpx) * alpha.size(),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_beta, beta.data(),
                               sizeof(cucpx) * beta.size(),
                               cudaMemcpyHostToDevice));
}

int CudaLanczos::current_step() {
    return iter - 1;
}

int CudaLanczos::max_steps() {
    return m;
}

void CudaLanczos::compute(cublasHandle_t handle,
                          cusparseHandle_t handle_sp,
                          CudaHamiltonianInterface& H,
                          double mu,
                          int steps) {
    // iter is the next iteration to be executed, so it can go one
    // beyond the max number of steps
    assert(iter + steps <= m + 1);
           
    CudaScalar d_mu(mu, 0.);
    int target_iter = iter + steps;

    cudaProfilerStart();

    checkCudaErrors(cudaDeviceSynchronize());

    for (; iter < target_iter; iter++) {
        //// u = A * vi - beta(i-1) * vi_minus_1 + mu * vi ////

        // u = A * vi
        H.act_even(*u, *vi);

        // work = beta(iter-1) * vi_minus_1
        work->set(zero_state);
        work->add(handle, d_beta + iter - 1, *vi_minus_1);

        // u = u - work1 = u - beta(iter-1) * vi_minus_1
        u->add(handle, minus_one.ptr, *work);

        // u += mu * vi
        u->add(handle, d_mu.ptr, *vi);

        //// alpha(iter-1) = u.dot(vi).real() ////
        u->dot(handle, d_alpha + iter - 1, *vi);

        // take real part (set imaginary part to zero_state)
        double* d_dbl_alpha = (double*) (d_alpha + iter - 1);
        checkCudaErrors(cudaMemset(d_dbl_alpha + 1, 0,
                                   sizeof(double)));

        // u -= alpha(iter-1) * vi
        work->set(zero_state);
        work->add(handle, d_alpha + iter - 1, *vi);

        u->add(handle, minus_one.ptr, *work);
        
        // beta(iter) = u.norm()
        u->norm(handle, &(d_beta[iter].x));

        // vi_minus_1 = vi (swap)
        CudaEvenState* tmp = vi_minus_1;
        vi_minus_1 = vi;
        vi = tmp;

        //// vi = u / beta(iter) ////
        // s = 1 / beta[iter]
        d_inverse(d_s, (double*) &(d_beta[iter].x));

        // vi = u * s = u / beta[iter]
        vi->set(*u);
        vi->scale(handle, d_s);
    }

    cudaProfilerStop();
}

