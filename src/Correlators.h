#ifndef CORRELATORS_H__ 
#define CORRELATORS_H__

#include "defs.h"
#include "eigen_utils.h"
#include "KitaevHamiltonian.h"
#include "MajoranaKitaevHamiltonian.h"
#include "FockSpaceUtils.h"

// Return tau = it or t depending on whether it's real or complex time
cpx get_tau(TIME_TYPE time_type, double t);

// Compute exp(a H) where H is a diagonalized Hermitian matrix given by
// H = V D V^{-1}, D = diag(evs)
Mat compute_exp_H(Mat& V, RealVec& evs, cpx a);

// Compute exp(a H) where H is a diagonalized Majorana Hamiltonian
Mat compute_exp_H(MajoranaKitaevHamiltonian& H, cpx a);

// Compute exp(a*H) in the given Q sector.
// Hamiltonian must be diagonalized.
Mat compute_exp_H(KitaevHamiltonian& H, cpx a, int N, int Q);

// Compute exp(a*H) in the given Q sector.
// Hamiltonian block must be diagonalized.
Mat compute_exp_H(KitaevHamiltonianBlock& block, cpx a, int N);

// Naive, slow versions of the functions above
Mat compute_exp_H_naive(KitaevHamiltonian& H, cpx a, int N, int Q);
Mat compute_exp_H_naive(KitaevHamiltonianBlock& block, cpx a, int N);

// Compute diag( exp(a * evs(1)), ... )
SpMat compute_exp_diagonal(RealVec& evs, cpx a);

double compute_partition_function(const RealVec& evs, double beta);
double compute_partition_function(KitaevHamiltonian& H, double beta);
double compute_partition_function(MajoranaKitaevHamiltonian& H, double beta);

// Compute a single data-point for the 2-point function
// (convenience function)
cpx compute_2_point_function(
        int N,
        double beta,
        double t,
        KitaevHamiltonian& H,
        TIME_TYPE time_type = REAL_TIME);

// Compute the 2-point function for the given range of betas and times.
// Returns a matrix of correlators(beta,time).
//
// Using this is faster than calling the function for each beta,t
// separately.
Mat compute_2_point_function(
        int N,
        vector<double> betas,
        vector<double> times,
        KitaevHamiltonian& H,
        TIME_TYPE time_type = REAL_TIME);

// Compute a single data-point for the 2-point function slowly
// using a naive implementation
cpx compute_2_point_function_naive(
        int N,
        double beta,
        double t,
        KitaevHamiltonian& H);

// Compute the long-time plateau value of a Majorana 2-point function
cpx compute_majorana_2pt_long_time(
        double beta,
        int N,
        double Z,
        MajoranaKitaevHamiltonian& H,
        vector<Mat>& Vdag_Chi_V
        );

// Compute the Majorana 2-point function for the given times and beta
// values. The results are written in the correlators matrix(time,beta).
// The thermal partition functions for the beta values are written in Z.
//
// The returned correlators are just the traces: they are not divided by Z.
//
// H = fully diagonalized Hamiltonian
void compute_majorana_2_pt_function(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    vector<double>& Z,
    bool print_progress);

// Compute the 2-point function and the 4-chi quantity
// that gives the fluctuations in the 2-point function.
//
// correlators = 
//       1/N \sum_i 
//       Tr(e^{-beta H) \chi_i(t) \chi_i(0))
//
// correlators_squared = 
//       1/N^2 \sum_{i,j} 
//       Tr(e^{-beta H) \chi_i(t) \chi_i(0) \chi_j(t) \chi_j(0))
void compute_majorana_2_pt_function_with_fluctuations(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    Mat& correlators_squared,
    vector<double>& Z,
    bool print_progress);

// Reference implementation of compute_majorana_2_pt_function used
// for testing
void compute_majorana_2_pt_function_reference_implementation(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    vector<double>& Z,
    bool print_progress);

#endif // CORRELATORS_H__

