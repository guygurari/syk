#include <iostream>
#include "Correlators.h"
#include "BasisState.h"
#include "Timer.h"

using namespace std;

cpx get_tau(TIME_TYPE time_type, double t) {
    if (time_type == REAL_TIME) {
        return cpx(0., t);
    }
    else {
        return t;
    }
}

Mat compute_exp_H_naive(KitaevHamiltonian& H, cpx a, int N, int Q) {
    return compute_exp_H(*(H.blocks[Q]), a, N);
}

Mat compute_exp_H_naive(KitaevHamiltonianBlock& block, cpx a, int N) {
    assert(block.is_diagonalized());
    assert(block.dim() == block.evs.size());
    int dim = block.dim();

    Mat expD = Mat::Zero(dim, dim);

    for (int i = 0; i < dim; i++) {
        expD(i,i) = exp(a * block.evs(i));
    }

    Mat expH(dim, dim);
    expH = block.U * expD * block.U.adjoint();
    return expH;
}

Mat compute_exp_H(KitaevHamiltonian& H, cpx a, int N, int Q) {
    return compute_exp_H(*(H.blocks[Q]), a, N);
}

// Compute the diagonal matrix:
//
// ( exp(a*evs[0]) 0                       ... )
// ( 0                        exp(a*evs[1] ... )
// (                   ...                                )
//
SpMat compute_exp_diagonal(RealVec& evs, cpx a) {
    int dim = evs.size();
    vector<CpxTriplet> triplets;

    for (int i = 0; i < dim; i++) {
        triplets.push_back(
                CpxTriplet(i, i, exp(a * evs(i))));
    }

    SpMat expD(dim,dim);
    expD.setFromTriplets(triplets.begin(), triplets.end());

    return expD;
}

Mat compute_exp_H(Mat& U, RealVec& evs, cpx a) {
    SpMat expD = compute_exp_diagonal(evs, a);
    Mat U_times_expD = U * expD;
    Mat Udag = U.adjoint();
    Mat expH = U_times_expD * Udag;
    return expH;
}

Mat compute_exp_H(MajoranaKitaevHamiltonian& H, cpx a) {
    return compute_exp_H(H.V, H.evs, a);
}

Mat compute_exp_H(KitaevHamiltonianBlock& block, cpx a, int N) {
    assert(block.is_diagonalized());
    assert(block.dim() == block.evs.size());
    return compute_exp_H(block.U, block.evs, a);
}

// Compute the diagonal matrix:
//
// ( exp(a*diagonal_elems[0]) 0                       ... )
// ( 0                        exp(a*diagonal_elems[1] ... )
// (                   ...                                )
//
/*SpMat compute_exp_diagonal(cpx a, RealVec diagonal_elems) {
    int dim = diagonal_elems.size();
    vector<CpxTriplet> triplets;

    for (int i = 0; i < dim; i++) {
        triplets.push_back(
                CpxTriplet(i, i, exp(a * diagonal_elems(i))));
    }

    SpMat expMat(dim,dim);
    expMat.setFromTriplets(triplets.begin(), triplets.end());
    return expMat;
}*/

double compute_partition_function(const RealVec& evs, double beta) {
    double Z = 0.;

    for (int i = 0; i < evs.size(); i++) {
        Z += exp(-beta * evs(i));
    }

    return Z;
}

double compute_partition_function(KitaevHamiltonian& H, double beta) {
    return compute_partition_function(H.eigenvalues(), beta);
}

double compute_partition_function(MajoranaKitaevHamiltonian& H, double beta) {
    return compute_partition_function(H.eigenvalues(), beta);
}

// Compute a single data-point for the 2-point function
cpx compute_2_point_function(
        int N,
        double beta,
        double t,
        KitaevHamiltonian& H,
        TIME_TYPE time_type
        ) {
    vector<double> betas;
    vector<double> times;
    betas.push_back(beta);
    times.push_back(t);
    Mat correlators = compute_2_point_function(
            N, betas, times, H, time_type);
    assert(correlators.rows() == 1);
    assert(correlators.cols() == 1);
    return correlators(0,0);

    // This is the old implementation where we compute separately for
    // each beta,t
    /*
    cpx correlator = 0.;
    double Z = compute_partition_function(H, beta);

    // Q = 0 vanishes trivially
    for (int Q = 1; Q <= N; Q++) {
        Mat U = compute_exp_H(H, cpx(0., -t), N, Q-1);
        Mat rho_times_Udag = compute_exp_H(H, cpx(-beta, t), N, Q);

        for (int i = 0; i < N; i++) {
            SpMat c_i = compute_c_matrix(i, N, Q);
            SpMat cdag_i = compute_cdagger_matrix(i, N, Q-1);

            // Doing multiplication + trace in one step is much faster,
            // because eigen knows it doesn't need to compute the
            // full matrix to get the trace!
            correlator += (c_i * rho_times_Udag * cdag_i * U).trace();
        }
    }

    // Normalize by the partition function
    correlator /= Z;
    return correlator;
    */
}

Mat compute_2_point_function(
        int N,
        vector<double> betas,
        vector<double> times,
        KitaevHamiltonian& H,
        TIME_TYPE time_type
        ) {
    Mat correlators = Mat::Zero(betas.size(), times.size());

    for (int i = 0; i < N; i++) {
        // Q = 0 vanishes trivially
        for (int Q = 1; Q <= N; Q++) {
            SpMat c_iQ = compute_c_matrix(i, N, Q);
            KitaevHamiltonianBlock* block_Q   = H.blocks[Q];
            KitaevHamiltonianBlock* block_Qm1 = H.blocks[Q-1];

            Mat C_iQ = block_Qm1->U.adjoint() * c_iQ * block_Q->U;

            for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
                double beta = betas[beta_i];

                for (unsigned time_i = 0; time_i < times.size(); time_i++) {
                    double t = times[time_i];
                    cpx it;

                    if (time_type == REAL_TIME) {
                        it = cpx(0.,t);
                    }
                    else if (time_type == EUCLIDEAN_TIME) {
                        // In this case 't' is actually \tau,
                        // and the Wick rotation gives \tau = it
                        it = cpx(t,0.);
                    }
                    else {
                        assert(false);
                    }

                    SpMat expDq1 = compute_exp_diagonal(
                            block_Q->evs, -beta + it);
                    SpMat expDq2 = compute_exp_diagonal(
                            block_Qm1->evs, -it);

                    cpx term = 
                        (expDq1 * C_iQ.adjoint() * expDq2 * C_iQ).trace();

                    correlators(beta_i, time_i) += term;
                }
            }
        }
    }

    // Divide by Z
    for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
        double beta = betas[beta_i];
        double Z = compute_partition_function(H, beta);

        for (unsigned time_i = 0; time_i < times.size(); time_i++) {
            correlators(beta_i, time_i) /= Z;
        }
    }

    return correlators;
}

cpx compute_2_point_function_naive(
        int N,
        double beta,
        double t,
        KitaevHamiltonian& H) {
    cpx correlator = 0.;
    double Z = compute_partition_function(H, beta);

    // Q = 0 vanishes trivially
    for (int Q = 1; Q <= N; Q++) {
        Mat U = compute_exp_H_naive(H, cpx(0., -t), N, Q-1);
        Mat rho_times_Udagger = compute_exp_H_naive(H, cpx(-beta, t), N, Q);

        for (int i = 0; i < N; i++) {
            Mat c_i = compute_c_matrix_dense(i, N, Q);
            Mat cdagger_i = compute_cdagger_matrix_dense(i, N, Q-1);
            Mat A = rho_times_Udagger * cdagger_i * U * c_i;
            correlator += A.trace();
        }
    }

    // Normalize by the partition function
    correlator /= Z;
    return correlator;
}

cpx compute_majorana_2pt_long_time(
        double beta,
        int N,
        double Z,
        MajoranaKitaevHamiltonian& H,
        vector<Mat>& Vdag_Chi_V
        ) {
    double long_time_value = 0.;

        for (int n = 0; n < H.evs.size(); n++) {
            for (int m = 0; m < H.evs.size(); m++) {
                if (abs(H.evs(n) - H.evs(m)) < epsilon) {
                    // Same energy, so add the contributions
                    for (int i = 0; i < N; i++) {
                        long_time_value +=
                            exp(-beta * H.evs(i)) *
                            norm(Vdag_Chi_V[i](n,m));
                    }
            }
        }
    }

    long_time_value /= N;
    long_time_value /= Z;
    return long_time_value;
}

// \sum_i Tr(e^{-bH} \chi_i(t) \chi_i(0)) / N
cpx compute_majorana_2pt_trace(
        MajoranaKitaevHamiltonian& H,
        double beta,
        cpx tau,
        vector<Mat>& chi_energy_eigenbasis
        ) {
    // \sum_i Tr(e^{-bH} \chi_i(t) \chi_i(0))
    cpx trace = 0.;

    for (int n = 0; n < H.dim(); n++) {
        for (int m = 0; m < H.dim(); m++) {
            double En = H.evs(n);
            double Em = H.evs(m);
            cpx exp_factor = exp(-beta * En + tau * (En - Em));

            double matrix_elems = 0.;

            for (int i = 0; i < H.mN; i++) {
                matrix_elems +=
                    norm(chi_energy_eigenbasis[i](n,m));
            }

            trace += exp_factor * matrix_elems;
        }
    }

    return trace / (double) H.mN;
}

// <GG>
cpx compute_majorana_2pt_GG(
        MajoranaKitaevHamiltonian& H,
        double beta,
        cpx tau,
        vector<Mat>& chi_energy_eigenbasis
        ) {
    // \sum_i Tr(e^{-bH} \chi_i(t) \chi_i(0))
    cpx trace = 0.;

    // chi_chi_exp(k,m) = \sum_{i,l} 
    //     ( exp(-i t E_l) \chi^i_kl \chi^i_lm )
    Mat chi_chi_exp = Mat::Zero(H.dim(), H.dim());

    // Compute chi_chi_exp
    for (int k = 0; k < H.dim(); k++) {
        for (int m = 0; m < H.dim(); m++) {
            for (int l = 0; l < H.dim(); l++) {
                double El = H.evs(l);
                cpx exponent = exp(-tau * El);

                for (int i = 0; i < H.mN; i++) {
                    chi_chi_exp(k,m) += 
                        exponent *
                        chi_energy_eigenbasis[i](k,l) *
                        chi_energy_eigenbasis[i](l,m);
                }
            }
        }
    }

    // Use chi_chi_exp to compute the correlator
    for (int k = 0; k < H.dim(); k++) {
        for (int m = 0; m < H.dim(); m++) {
            double Ek = H.evs(k);
            double Em = H.evs(m);

            cpx exp_factor = exp( -beta * Ek + tau * (Ek + Em) );
            trace += exp_factor * chi_chi_exp(k,m) * chi_chi_exp(m,k);
        }
    }

    return trace / (double) (H.mN * H.mN);
}

// <GG*>
cpx compute_majorana_2pt_GGstar(
        MajoranaKitaevHamiltonian& H,
        double beta,
        Mat& A_t,
        Mat& A_minus_t
        ) {
    cpx result = 0.;

    for (int m = 0; m < H.dim(); m++) {
        cpx exp_beta_E = exp(-beta * H.eigenvalues()(m));

        for (int n = 0; n < H.dim(); n++) {
            result += exp_beta_E * A_t(m,n) * A_minus_t(n,m);
        }
    }

    return result;
}

void compute_majorana_2_pt_function(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    vector<double>& Z,
    bool print_progress
        ) {
    assert(H.is_fully_diagonalized());
    Timer timer;
    vector<Mat> chi_energy_eigenbasis(H.mN);

    if (print_progress) cout << "Compute chi in energy eigenbasis..." << endl;
    for (int i = 0; i < H.mN; i++) {
        chi_energy_eigenbasis[i] = H.V.adjoint() * H.chi[i] * H.V;
    }
    if (print_progress) timer.print();

    for (unsigned bi = 0; bi < betas.size(); bi++) {
        Z[bi] = compute_partition_function(H.evs, betas[bi]);
    }

    // Print long time values
    if (print_progress) {
        for (unsigned bi = 0; bi < betas.size(); bi++) {
            double beta = betas[bi];
            cpx long_time_value = compute_majorana_2pt_long_time(
                    beta, H.mN, Z[bi], H, chi_energy_eigenbasis);
            cout << "Long time value at beta=" << beta << " : "
                << long_time_value << endl;
        }
    }

    // Compute 2-point function
    for (unsigned ti = 0; ti < times.size(); ti++) {
        double t = times[ti];
        cpx tau = get_tau(time_type, t);

        for (unsigned bi = 0; bi < betas.size(); bi++) {
            double beta = betas[bi];

            correlators(ti, bi) = compute_majorana_2pt_trace(
                    H, beta, tau, chi_energy_eigenbasis);
        }
    }
}

// Matrix used in Majorana 2-point function fluctuations
Mat compute_A_matrix(
    MajoranaKitaevHamiltonian& H,
    cpx tau,
    vector<Mat>& chi_energy_eigenbasis) {

    Mat A = Mat::Zero(H.dim(), H.dim());
    RealVec evs = H.eigenvalues();
    SpMat exp_itE = compute_exp_diagonal(evs, -tau);

    for (int i = 0; i < H.mN; i++) {
        A += chi_energy_eigenbasis[i] * exp_itE * chi_energy_eigenbasis[i];
    }

    A /= H.mN;
    return A;
}

void compute_majorana_2_pt_function_with_fluctuations(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    Mat& correlators_squared,
    vector<double>& Z,
    bool print_progress
        ) {
    assert(H.is_fully_diagonalized());
    Timer timer;
    vector<Mat> chi_energy_eigenbasis(H.mN);

    if (print_progress) cout << "Compute chi in energy eigenbasis..." << endl;
    for (int i = 0; i < H.mN; i++) {
        chi_energy_eigenbasis[i] = H.V.adjoint() * H.chi[i] * H.V;
    }
    if (print_progress) timer.print();

    for (unsigned bi = 0; bi < betas.size(); bi++) {
        Z[bi] = compute_partition_function(H.evs, betas[bi]);
    }

    // Compute 2-point function
    for (unsigned ti = 0; ti < times.size(); ti++) {
        double t = times[ti];
        cpx tau = get_tau(time_type, t);

        Mat A_t = compute_A_matrix(H, tau, chi_energy_eigenbasis);
        Mat A_minus_t;

        if (time_type == REAL_TIME) {
            A_minus_t = A_t.adjoint();
        }
        else {
            A_minus_t = compute_A_matrix(H, -tau, chi_energy_eigenbasis);
        }

        for (unsigned bi = 0; bi < betas.size(); bi++) {
            double beta = betas[bi];

            correlators(ti, bi) = compute_majorana_2pt_trace(
                    H, beta, tau, chi_energy_eigenbasis);

            correlators_squared(ti, bi) = compute_majorana_2pt_GGstar(
                    H, beta, A_t, A_minus_t);
        }
    }
}

void compute_majorana_2_pt_function_reference_implementation(
    MajoranaKitaevHamiltonian& H,
    vector<double>& betas,
    vector<double>& times,
    TIME_TYPE time_type,
    Mat& correlators,
    vector<double>& Z,
    bool print_progress
        ) {
    assert(H.is_fully_diagonalized());
    Timer timer;
    vector<Mat> Vdag_Chi_V(H.mN);

    if (print_progress) cout << "Preparing U*chi*Udag products..." << endl;
    for (int i = 0; i < H.mN; i++) {
        Vdag_Chi_V[i] = H.V.adjoint() * H.chi[i] * H.V;
    }
    if (print_progress) timer.print();

    for (unsigned bi = 0; bi < betas.size(); bi++) {
        Z[bi] = compute_partition_function(H.evs, betas[bi]);
    }

    // Print long time values
    if (print_progress) {
        for (unsigned bi = 0; bi < betas.size(); bi++) {
            double beta = betas[bi];
            cpx long_time_value = compute_majorana_2pt_long_time(
                    beta,
                    H.mN,
                    Z[bi],
                    H,
                    Vdag_Chi_V
                    );
            cout << "Long time value at beta=" << beta << " : "
                << long_time_value << endl;
        }
    }

    // Compute 2-point function
    for (unsigned ti = 0; ti < times.size(); ti++) {
        double t = times[ti];
        cpx tau = get_tau(time_type, t);

        SpMat exp_itE = compute_exp_diagonal(H.evs, tau);
        SpMat exp_minus_itE = compute_exp_diagonal(H.evs, -tau);

        for (unsigned bi = 0; bi < betas.size(); bi++) {
            // \sum_i Tr(e^{-bH} \chi_i(t) \chi_i(0))
            cpx sum_of_traces = 0.;

            double beta = betas[bi];

            SpMat exp_minus_beta_E = compute_exp_diagonal(
                    H.evs, -beta);

            for (int i = 0; i < H.mN; i++) {
                cpx trace_i = (
                        exp_minus_beta_E *
                        exp_itE *
                        Vdag_Chi_V[i] * 
                        exp_minus_itE * 
                        Vdag_Chi_V[i]).trace();

                sum_of_traces += trace_i;

                //cout << "t = " << t << "\t" << " Tr(i) = " << trace_i << endl;
            }

            cpx avg_of_traces = sum_of_traces / (double) H.mN;
            correlators(ti, bi) = avg_of_traces;
        }
    }
}

