#ifndef __LANCZOS_H__
#define __LANCZOS_H__

#include "defs.h"
#include "eigen_utils.h"
#include "MajoranaKitaevHamiltonian.h"
#include "FactorizedSpaceUtils.h"
#include "FactorizedHamiltonian.h"
#include <boost/random/mersenne_twister.hpp>

// Method from Cullum and Willoughby, Lanczos algorithms for large
// symmetric eigenvalue computations (section 2.1).
void reference_lanczos(MajoranaKitaevHamiltonian& H,
                       double mu,
                       int lanczos_steps,
                       RealVec& alpha,
                       RealVec& beta,
                       boost::random::mt19937* gen);

// Same as above but starts with a given initial state
// instead of choosing a random one.
void reference_lanczos(MajoranaKitaevHamiltonian& H,
                       double mu,
                       int lanczos_steps,
                       RealVec& alpha,
                       RealVec& beta,
                       Vec& initial_state);

// Method from Cullum and Willoughby, Lanczos algorithms for large
// symmetric eigenvalue computations (section 2.1).
//
// mu*Id is added to the Hamiltonian.
//
// Outputs the resulting Lanczos coefficients to alpha, beta.
// 
// If extended_beta = true, stores also the extra element beta(m+1)
// in beta.
void factorized_lanczos(FactorizedHamiltonianGenericParity& H,
                        double mu,
                        int lanczos_steps,
                        boost::random::mt19937* gen,
                        RealVec& alpha,
                        RealVec& beta,
                        bool extended_beta = false
                        );

// Same as above but starts with a given initial state
// instead of choosing a random one.
// 
// If extended_beta = true, stores also the extra element beta(m+1)
// in beta.
void factorized_lanczos(FactorizedHamiltonianGenericParity& H,
                        double mu,
                        int lanczos_steps,
                        Mat& initial_state,
                        RealVec& alpha,
                        RealVec& beta,
                        bool extended_beta = false
                        );

// Given the results of the Lanczos algorithm without re-orthogonalization,
// this returns the resulting eigenvalues that are eigenvalues of the
// original matrix.
// 
// alpha should have size m, and beta should have size m-1 where m is the
// number of steps.
RealVec find_good_lanczos_evs(const RealVec& alpha, const RealVec& beta);

// Returns the Lanczos approximation to the eigenvalues of the original
// matrix. error_estimates contains upper bounds on the absolute errors
// of these eigenvalues.
//
// extended_beta should include the m computed elements of beta, of which
// the first m-1 elements are the sub-diagonal of the Lanczos matrix T.
RealVec find_good_lanczos_evs_and_errs(const RealVec& alpha,
                                       const RealVec& extended_beta,
                                       RealVec& error_estimates,
                                       boost::random::mt19937* gen);

void print_lanczos_results(RealVec& H_evs, RealVec& lanczos_evs);

void print_lanczos_results(RealVec& H_evs,
                           RealVec& lanczos_evs,
                           RealVec& error_estimates);

/////////////////////// Private /////////////////////////////////////

// Return all the eigenvalues of the tridiagonal T matrix
RealVec all_lanczos_evs(const RealVec& alpha, const RealVec& beta);

RealVec all_lanczos_evs(const RealVec& alpha,
                        const RealVec& beta,
                        Mat& eigenvectors);

// Return the tridiagonal matrix with the given diagonal and subdiagonal
RealMat get_tridiagonal_matrix(const RealVec& diagonal,
                               const RealVec& subdiagonal);

RealSpMat get_tridiagonal_sparse_matrix(const RealVec& diagonal,
                                        const RealVec& subdiagonal);

// Whether evec is an eigenvector with ev of the tridiagonal matrix
// where alpha is the diagonal and beta the subdiagonal.
bool is_eigenvector(const RealVec& alpha,
                    const RealVec& beta,
                    double ev,
                    const RealVec& evec);

// Compute the eigenvector corresponding to the given eigenvalue (or
// approximate eigenvalue) of a symmetric tridiagonal matrix. alpha is the
// diagonal and beta is the subdiagonal.
RealVec find_eigenvector_for_ev(const RealVec& alpha,
                                const RealVec& beta,
                                double ev,
                                boost::random::mt19937* gen,
                                int max_iterations = 20);

// A slower and very memory-hungry implementation of
// find_good_lanczos_evs_and_errs.
RealVec find_good_lanczos_evs_and_errs_full_diagonalization(
    const RealVec& alpha,
    const RealVec& extended_beta,
    RealVec& error_estimates);

// For each lanczos_evs(j) find the nearest true_evs(i) and
// add (i -> j) to the map.
map<int, int> find_nearest_true_ev(RealVec& true_evs, RealVec& lanczos_evs);

#endif // __LANCZOS_H__
