//
// This program is used for benchmarking CPU Lanczos code.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <bitset>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "defs.h"
#include "DisorderParameter.h"
#include "KitaevHamiltonian.h"
#include "MajoranaDisorderParameter.h"
#include "NaiveMajoranaKitaevHamiltonian.h"
#include "MajoranaKitaevHamiltonian.h"
#include "Spectrum.h"
#include "Correlators.h"
#include "Timer.h"
#include "Lanczos.h"

#include "kitaev.h"

namespace po = boost::program_options;

typedef unsigned int uint;

typedef struct command_line_options {
    string run_name;
    string data_dir;
    int N;
    double J;
    int seed;
    boost::random::mt19937* gen;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\nN = " << N;
        ss << "\nJ = " << J;
        ss << "\nseed = " << seed;

        ss << "\n";
        return ss.str();
    }
} command_line_options;

int parse_command_line_options(int argc, char** argv, command_line_options& opts) {
    opts.J = 1.;
    opts.data_dir = "data";

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("run-name", 
         po::value<string>(&opts.run_name),
         "set run's name (required for fixing a unique seed for each job)")
	    ("data-dir", 
	     po::value<string>(&opts.data_dir),
         "where to save data")
	    ("N", 
	     po::value<int>(&opts.N)->required(),
         "number of fermions")
        ("J",
         po::value<double>(&opts.J),
         "coupling")
	    ("seed",
         po::value<int>(&opts.seed),
	     "random seed")
	    ("help", "produce help message");

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (!vm.count("run-name")) {
            opts.run_name = "test";
        }

        if (!vm.count("seed")) {
            opts.seed = get_random_seed(opts.run_name);
        }

        if (opts.N % 2 != 0) {
            cerr << "N must be even (majorana)" << endl;
            exit(1);
        }

        opts.gen = new boost::random::mt19937(opts.seed);

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

cpx compute_lambda(Vec v, Vec v2, double mu) {
    cpx lambda = 0.;

    for (int i = 0; i < v.size(); i++) {
        lambda += v2(i) / v(i);
    }

    lambda /= v.size();
    lambda -= mu;
    return lambda;
}

void compute_majorana_leading_ev(command_line_options& opts) {
    cout << "Creating J tensor" << endl;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, opts.gen);

    cout << "Computing Hamiltonian" << endl;
    Timer timer;
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    double mu = 1.;
    SpMat Id(H.dim(), H.dim());
    Id.setIdentity();
    
    SpMat H_shifted = H.matrix + mu * Id;

    Vec v(H.dim());
    boost::random::normal_distribution<> dist(0., 1.);
    for (int i = 0; i < v.size(); i++) {
        v(i) = cpx(dist(*opts.gen), dist(*opts.gen));
    }
    v = v / v.norm();

    for (int i = 0; ; i++) {
        Vec v2 = H_shifted * v;
        Vec v2_normalized = v2 / v2.norm();

        if (i % 100 == 0) {
            double dist = (v2_normalized - v).norm();
            cpx lambda = compute_lambda(v, v2, mu);
            printf("%d\td=%e\tlambda=%e\n", i, dist, lambda.real());

            // dist of 2 implies negative eigenvalue
            if (dist < epsilon || (fabs(dist-2.) < epsilon)) {
                v = v2;
                break;
            }
        }
        
        v = v2_normalized;
    }

    // Compute the eigenvalue
    Vec v2 = H_shifted * v;
    cpx lambda = compute_lambda(v, v2, mu);
    printf("\nlambda = (%.2e, %.2e)\n", lambda.real(), lambda.imag());
}

void print_ev_comparison(RealVec& evs1, RealVec& evs2) {
    for (int i = 0; i < evs1.size() && i < evs2.size(); i++) {
        double lambda1 = evs1(i);
        double lambda2 = evs2(i);

        cout << i
             << "\t" << lambda1
             << "\t" << lambda2
             << "\t ( " << relative_err(lambda1, lambda2) << " )"
             << "\n";
    }
    cout << endl;
}

// alpha, beta should only include the values that go into T
Mat get_lanczos_T_matrix(RealVec& alpha, RealVec& beta, int m) {
    assert(alpha.size() == m);
    assert(beta.size() == m-1);

    Mat T = Mat::Zero(m,m);

    for (int i = 0; i < m; i++) {
        T(i,i) = alpha(i);

        if (i < m-1) {
            T(i,i+1) = beta(i);
            T(i+1,i) = beta(i);
        }
    }

    return T;
}

// beta should include a leading 0 (which is not used), and both alpha
// and beta can include trailing terms which are not used.
void print_lanczos_results(RealVec& alpha,
                           RealVec& beta,
                           int m,
                           RealVec& Aevs) {

    assert(alpha.size() == m);
    assert(beta.size() == m - 1);

    Mat T = get_lanczos_T_matrix(alpha, beta, m);

    SelfAdjointEigenSolver<Mat> solver;
    solver.compute(T, EigenvaluesOnly);
    RealVec Tevs = solver.eigenvalues();

    RealVec unique_A_evs = get_unique_elements(Aevs, epsilon);
    RealVec unique_T_evs = get_unique_elements(Tevs, 1e-3);

    RealVec good_T_evs = find_good_lanczos_evs(alpha, beta);

    cout << "i\tA ev\t\tgood T ev\t\trelative error\n";

    int j = 0;

    for (int i = 0; i < unique_A_evs.size(); i++) {
        cout << i << "\t" << unique_A_evs(i);

        if (j < good_T_evs.size() &&
            fabs(good_T_evs(j) - unique_A_evs(i)) < epsilon) {

            cout << "\t" << good_T_evs(j)
                 << "\t (" << relative_err(good_T_evs(j), unique_A_evs(i))
                 << " )";
            j++;
        }

        cout << "\n";
    }
    cout << endl;

    if (j == good_T_evs.size()) {
        cout << "All good T values match A values.\n\n";
    }
    else {
        cout << "Some T values don't match A values!!\n\n";
    }

    double match_ratio = (double) j / (double) unique_A_evs.size();
    cout << "Matched " << match_ratio << " of total eiganvalues\n\n";

    /*
    cout << "All T / A=H+mu*I evs at m=" << m << " :\n";
    print_ev_comparison(Tevs, Aevs);

    cout << "Unique T / H+mu*I evs at m=" << m << " :\n";
    print_ev_comparison(unique_T_evs, unique_A_evs);

    cout << "Good T / unique H+mu*I evs at m=" << m << " :\n";
    print_ev_comparison(good_T_evs, unique_A_evs);*/
}

// Method from Cullum and Willoughby, Lanczos algorithms for large
// symmetric eigenvalue computations (section 2.1).
void run_lanczos(command_line_options& opts) {
    cout << "Computing Hamiltonian" << endl;

    Timer timer;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, opts.gen);
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    double mu = 1.;
    SpMat Id(H.dim(), H.dim());
    Id.setIdentity();
    SpMat A = H.matrix + mu * Id;

    cout << "Diagonalizing" << endl;

    SelfAdjointEigenSolver<Mat> Asolver;
    Asolver.compute(A, EigenvaluesOnly);
    RealVec Aevs = Asolver.eigenvalues();
    timer.print();

    cout << "Running reference Lanczos on A=H+mu*I (mu="
         << mu << ")" << endl;

    RealVec alpha;
    RealVec beta;
    int m = 200;

    timer.reset();
    reference_lanczos(H, 1., m, alpha, beta, opts.gen);
    timer.print();

    print_lanczos_results(alpha, beta, m, Aevs);
}

void run_factorized_lanczos(command_line_options& opts) {
    cout << "Computing Hamiltonian" << endl;

    Timer timer;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, opts.gen);
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    double mu = 1.;
    SpMat Id(H.dim(), H.dim());
    Id.setIdentity();
    SpMat A = H.matrix + mu * Id;

    cout << "Diagonalizing" << endl;

    SelfAdjointEigenSolver<Mat> Asolver;
    Asolver.compute(A, EigenvaluesOnly);
    RealVec Aevs = Asolver.eigenvalues();
    timer.print();

    cout << "Computing factorized Hamiltonian" << endl;

    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    timer.reset();
    FactorizedHamiltonianGenericParity factH(space, Jtensor);
    timer.print();

    cout << "Running Lanczos on A=H+mu*I (mu=" << mu << ")" << endl;
    int lanczos_steps = 200;
    RealVec alpha;
    RealVec beta;
    
    timer.reset();
    factorized_lanczos(factH, mu, lanczos_steps, opts.gen,
                       alpha, beta);
    timer.print();

    print_lanczos_results(alpha, beta, lanczos_steps, Aevs);
}

void benchmark_hamiltonian_memory(command_line_options& opts) {
    Timer timer;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, opts.gen);
    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);

    cout << "Computing factorized Hamiltonian, N=" << opts.N << endl;

    timer.reset();
    FactorizedHamiltonian H(space, Jtensor);
    timer.print();

    int a;
    cout << "Done, waiting for input" << endl;
    cin >> a;
}
                             

//
// Compute the action of a single Majorana Hamiltonian term on a single input element.
//
// Input:
// - dressed_coupling: J_{abcd} / 4. * factor-of-i
// - a,b,c,d:  Indices of the coupling
// - input:    The value of the input vector element
// - addr:     Address of the input vector element. Each bit corresponds to the
//             occupataion of a single qubit. LSB is a=0, next bit is a=1, etc.
//
// Output:
// - output_addr: Output address, same format as addr.
//
// Returns: Value of the output element.
//
//
inline cpx compute_single_action(const cpx dressed_coupling,
                                 const uint a, const uint b, const uint c, const uint d,
                                 const uint a_parity, const uint b_parity,
                                 const uint c_parity, const uint d_parity,
                                 const uint i, const uint j, const uint k, const uint l, 
                                 const cpx input,
                                 const uint addr,
                                 const int N,
                                 int* exclusive_right_popcnt_sign,
                                 uint& output_addr) {
    // Note: does not deal with overlaps like a=0,b=1

    // Dirac indices
    /*
    const uint i = a / 2;
    const uint j = b / 2;
    const uint k = c / 2;
    const uint l = d / 2;

    const uint a_parity = a % 2;
    const uint b_parity = b % 2;
    const uint c_parity = c % 2;
    const uint d_parity = d % 2;
    */

    // The term acts by flipping the spins at the Dirac indices
    // (When opening the brackets of \psi \psi \psi \psi ~ (c+cb)(c+cb)...(c+cb),
    //  only one term survives on the give input: the one that flips the four
    //  occupancies.)
    const uint flipper = (1 << i) | (1 << j) | (1 << k) | (1 << l);
    output_addr = addr ^ flipper;

    // Compute the sign from the odd-numbered Majoranas:
    // \psi_{2i+1} = i (c_i - <--- cbar_i) / sqrt(2)
    // (The minus is there when the fermion is unoccupied in the state,
    //  because that's when we act with cbar_i.)
    const uint i_occ = (addr >> i) & 1;
    const uint j_occ = (addr >> j) & 1;
    const uint k_occ = (addr >> k) & 1;
    const uint l_occ = (addr >> l) & 1;

    const uint i_gets_minus = a_parity & (i_occ ^ 1);
    const uint j_gets_minus = b_parity & (j_occ ^ 1);
    const uint k_gets_minus = c_parity & (k_occ ^ 1);
    const uint l_gets_minus = d_parity & (l_occ ^ 1);

    const uint output_gets_minus =
        i_gets_minus ^ j_gets_minus ^ k_gets_minus ^ l_gets_minus;

    const int minus_factor = 1 - ((int) (output_gets_minus << 1));

    // Compute the sign from anti-commuting fermions to their location
    // in the state
    const int anticomm_sign =
        exclusive_right_popcnt_sign[i] *
        exclusive_right_popcnt_sign[j] *
        exclusive_right_popcnt_sign[k] *
        exclusive_right_popcnt_sign[l];

    // Compute the final matrix element
    cpx output = dressed_coupling *
        ((double) (minus_factor * anticomm_sign)) *
        input;

    return output;
}

// transformed_state should be initialized to zero
void compute_transformed_state(MajoranaKitaevDisorderParameterWithoutNeighbors& Jtensor,
                               int N,
                               int D,
                               Vec& state,
                               Vec& transformed_state) {
    const cpx i_powers[4] = {
        cpx(1.,0.),
        cpx(0.,1.),
        cpx(-1.,0.),
        cpx(0.,-1.)
    };

    const int N_dirac = N/2;
    
    for (uint addr = 0; addr < D; addr++) {
        // Make lookup table for the anticommutator signs
        // exclusive_right_popcnt_sign[n] is the sign we would get
        // by anti-commuting with all Dirac creation operators that have
        // index k<n. Their address bits are to the right of n (i.e.
        // k is less significant than n). In the Fock space state,
        // their creation operators show up to the left of where the n
        // operator ends up.
        uint mask = 1;
        int exclusive_right_popcnt_sign[N_dirac];
        exclusive_right_popcnt_sign[0] = 1;

        for (int n = 1; n < N_dirac; n++) {
            uint n_is_occupied = ((addr & mask) != 0);
            // this is: n_sign = (n_is_occupied ? -1 : 1)
            int n_sign = 1 - ((int) (n_is_occupied << 1));
            exclusive_right_popcnt_sign[n] =
                exclusive_right_popcnt_sign[n-1] * n_sign;
            mask <<= 1;
        }

        for (uint a = 0; a < N; a++) {
            const uint a_parity = a % 2;
            const uint i = a / 2;

            for (uint b = a+1; b < N; b++) {
                const uint b_parity = a % 2;
                const uint j = b / 2;

                for (uint c = b+1; c < N; c++) {
                    const uint c_parity = a % 2;
                    const uint k = c / 2;

                    for (uint d = c+1; d < N; d++) {
                        const uint d_parity = a % 2;
                        const uint l = d / 2;

                        // Compute the total complex 'i' factor
                        // \psi_{2i+1} = i <--- (c_i - cbar_i) / sqrt(2)
                        const uint i_power =
                            a_parity + b_parity + c_parity + d_parity;
                        const cpx i_factor = i_powers[i_power];

                        // Factor 1/4 from definition of Majorana
                        // \psi_{2i+1} = i (c_i - cbar_i) / sqrt(2) <---
                        // (Note: this can all be included in coupling) 
                        const cpx dressed_coupling =
                            Jtensor.Jelems[a][b][c][d] * i_factor / 4.;

                        uint output_addr;
                        cpx output = compute_single_action(dressed_coupling,
                                                           a, b, c, d,
                                                           a_parity, b_parity, c_parity, d_parity,
                                                           i, j, k, l,
                                                           state(addr),
                                                           addr,
                                                           N,
                                                           exclusive_right_popcnt_sign,
                                                           output_addr);

                        transformed_state(output_addr) += output;
                    }
                }
            }
        }
    }
}

void profile_compute_single_action(command_line_options& opts) {
    MajoranaKitaevDisorderParameterWithoutNeighbors Jtensor(opts.N,
                                                            opts.J,
                                                            opts.gen);
    const int iterations = 1;

    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    Space space = Space::from_majorana(opts.N);

    Vec state = get_random_state(space, opts.gen);
    Vec gold_transformed_state = Vec::Zero(space.D);
    Vec transformed_state = Vec::Zero(space.D);

    cout << "Computing gold transformed state\n";
    Timer timer;
    for (int i = 0; i < iterations; i++) {
        gold_transformed_state = H.matrix * state;
    }
    timer.print_msec();

    cout << "Computing transformed state\n";
    timer.reset();
    for (int i = 0; i < iterations; i++) {
        compute_transformed_state(Jtensor,
                                  space.N, space.D,
                                  state, transformed_state);
    }
    timer.print_msec();
}

void test_compute_single_action(command_line_options& opts) {
    MajoranaKitaevDisorderParameterWithoutNeighbors Jtensor(opts.N, opts.J, opts.gen);
    Space space = Space::from_majorana(opts.N);
    MajoranaKitaevHamiltonian H(space.N, &Jtensor);

    Vec state = get_random_state(space, opts.gen);
    Vec gold_transformed_state = Vec::Zero(space.D);
    Vec transformed_state = Vec::Zero(space.D);

    compute_transformed_state(Jtensor,
                              space.N, space.D,
                              state,
                              transformed_state);

    cout << setprecision(4);
    cout << "Input vs. Gold vs. Actual:\n";
    bool correct = true;
        
    for (uint i = 0; i < space.D; i++) {
        bitset<16> bits(i);

        cpx input = state(i);
        cpx gold = gold_transformed_state(i);
        cpx actual = transformed_state(i);

        bool match = abs(gold - actual) < epsilon;
            
        if (!match) {
            cout << bits
                 << "\t" << (match ? " " : "X")
                 << "\t" << input
                 << "\t" << gold
                 << "\t" << actual
                 << "\n";

            correct = false;
        }
    }

    if (correct) {
        cout << "All correct!\n";
    }

    cout << setprecision(precision);
}

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    // FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    // MajoranaKitaevDisorderParameter Jtensor(space.N);
    // Jtensor.Jelems[0][1][2][3] = 1.;
    // Mat state = Mat::Zero(space.left.D, space.right.D);
    // state(0,0) = 1.;
    // FactorizedHamiltonian H(space, Jtensor);

    // cout << "left even:\n";
    // for (int i = 0; i < H.num_even_operators(); i++) {
    //     cout << H.left_even_operators[i].top_block << endl << endl;
    // }

    // cout << "right even:\n";
    // for (int i = 0; i < H.num_even_operators(); i++) {
    //     cout << H.right_even_operators[i].top_block << endl << endl;
    // }

    //compute_majorana_leading_ev(opts);
    // run_lanczos(opts);
    //run_factorized_lanczos(opts);
    //test_compute_single_action(opts);
    //profile_compute_single_action(opts);
    benchmark_hamiltonian_memory(opts);
    return 0;
}
