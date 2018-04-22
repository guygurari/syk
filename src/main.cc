#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "defs.h"
#include "DisorderParameter.h"
#include "KitaevHamiltonian.h"
#include "MajoranaDisorderParameter.h"
#include "NaiveMajoranaKitaevHamiltonian.h"
#include "MajoranaKitaevHamiltonian.h"
#include "Spectrum.h"
#include "Correlators.h"
#include "Timer.h"

#include "kitaev.h"

namespace po = boost::program_options;

boost::random::mt19937* gen = 0;

typedef struct command_line_options {
    string run_name;
    string data_dir;
    int N;
    double J;
    bool real_J;
    int seed;
    bool majorana;

    bool compute_2_pt_func;
    bool compute_2_pt_func_with_fluctuations;
    bool compute_fermion_matrix;
    bool compute_anti_commutator;

    // Correlator temperature range
    vector<double> betas;

    // Beta range

    // Correlator time range
    double t0;
    double t1;
    double dt;
    TIME_TYPE time_type;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\nN = " << N;
        ss << "\nJ = " << J;
        ss << "\nreal_J = " << real_J;
        ss << "\nseed = " << seed;
        ss << "\nmajorana = " << majorana;
        ss << "\ncompute_2_pt_func = " << compute_2_pt_func;
        ss << "\ncompute_anti_commutator = " << compute_anti_commutator;

        if (compute_2_pt_func || compute_anti_commutator) {
	    ss << "\nbetas = ";
	    for (int i = 0; i < betas.size(); i++) {
		ss << betas[i];
		if (i < betas.size() - 1) {
		    ss << ", ";
		}
	    }
	    
	    ss << "\nt0 = " << t0;
	    ss << "\nt1 = " << t1;
            ss << "\ndt = " << dt;
            ss << "\ntime_type = " 
                << (time_type == REAL_TIME ? "real-time" : "euclidean-time");
        }

        ss << "\n";
        return ss.str();
    }
} command_line_options;

vector<double> get_betas_vector_from_beta_range(double beta0,
						double beta1,
						double dbeta) {
    vector<double> betas;
    for (double beta = beta0; beta <= beta1; beta += dbeta) {
	betas.push_back(beta);
    }
    return betas;
}

vector<double> get_betas_vector_from_T_range(double T0,
						double T1,
						double dT) {
    vector<double> betas;
    for (double T = T0; T <= T1; T += dT) {
	double beta = 1. / T;
	betas.push_back(beta);
    }
    return betas;
}

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    opts.J = 1.;
    opts.data_dir = "data";
    opts.majorana = false;
    opts.real_J = false;

    opts.compute_2_pt_func = false;
    opts.compute_2_pt_func_with_fluctuations = false;
    opts.compute_fermion_matrix = false;
    opts.compute_anti_commutator = false;

    double T0 = 1.;
    double T1 = 1.;
    double dT = 0.;

    double beta0 = 1.;
    double beta1 = 1.;
    double dbeta = 0.;

    opts.t0 = 0.;
    opts.t1 = 0.;
    opts.dt = 0.;
    opts.time_type = REAL_TIME;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            // Basic parameters
            ("run-name", 
             po::value<string>(&opts.run_name)->required(),
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
            ("real-J",
             "use real disorder parameters")
            ("majorana",
	     "use the Majorana Hamiltonian (then N is the number of Majorana fermions and must be even)")
	    // Correlator parameters
	    ("2pt",
	     "compute the 2-point function. requires T0,(T1,dT),,t0,(t1,dt)")
            ("2pt-with-fluctuations",
             "compute the 2-point function and its fluctuations. requires T0,(T1,dT),,t0,(t1,dt). only available for majorana.")
            ("compute-fermion-matrix",
             "save the fermion operator in the energy eigenbasis. only available for majorana.")
            ("anti-commutator",
             "compute anti-commutator. requires t0,(t1,dt). if combined with --2pt, computes anti-commutator and also 2-point functions.")
            ("T0",
             po::value<double>(&T0),
	     "initial correlator temperature")
	    ("T1",
             po::value<double>(&T1),
             "final correlator temperature")
            ("dT",
             po::value<double>(&dT),
             "correlator temperature step")
	    ("treat-T-as-beta",
	     "treat the given Temperature range as a beta=1/T range (legacy option. can just specify --betaX instead.)")
	    ("beta",
	     po::value<double>(&beta0),
             "initial or single correlator beta")
	    ("beta0",
	     po::value<double>(&beta0),
             "equivalent to --beta")
            ("beta1",
             po::value<double>(&beta1),
             "final correlator beta")
	    ("dbeta",
	     po::value<double>(&dbeta),
	     "correlator beta step")
	    ("t0",
             po::value<double>(&opts.t0),
	     "initial correlator time")
	    ("t1",
             po::value<double>(&opts.t1),
             "final correlator time")
            ("dt",
             po::value<double>(&opts.dt),
             "correlator time step")
            ("euclidean-t",
	     "use Euclidean time instead of real time in the correlator")
	    // Other stuff
            ("naive-hamiltonian",
	     "use the naive and slow Hamiltonian")
	    ("seed",
             po::value<int>(&opts.seed),
	     "random seed")
	    ("help", "produce help message")
	;

	po::variables_map vm;        
	po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("real-J")) {
            opts.real_J = true;
        }

        if (vm.count("euclidean-t")) {
            opts.time_type = EUCLIDEAN_TIME;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

	if (vm.count("beta0") || vm.count("beta")) {
	    if ((vm.count("beta0") || vm.count("beta"))
		&& (!vm.count("beta1") || !vm.count("betaT"))) {
		beta1 = beta0;
		dbeta = 1.;
	    }

	    opts.betas = get_betas_vector_from_beta_range(beta0, beta1, dbeta);
	}
	else if (vm.count("T0")){
	    if (vm.count("T0")
		&& (!vm.count("T1") || !vm.count("dT"))) {
		T1 = T0;
		dT = 1.;
	    }

	    if (vm.count("treat-T-as-beta")) {
		opts.betas = get_betas_vector_from_beta_range(beta0,
							      beta1,
							      dbeta);
	    }
	    else {
		opts.betas = get_betas_vector_from_T_range(T0, T1, dT);
	    }
	}

        if (vm.count("2pt")) {
            opts.compute_2_pt_func = true;

            if (opts.betas.size() == 0 || !vm.count("t0")) {
                cout << "Missing parameters for --2pt" << endl;
                return 1;
            }
        }

        if (vm.count("2pt-with-fluctuations")) {
            opts.compute_2_pt_func_with_fluctuations = true;

            if (!vm.count("majorana")) {
                cout << "--2pt-with-fluctuations only available with --majorana"
		     << endl;
                return 1;
            }

            if (opts.betas.size() == 0 || !vm.count("t0")) {
                cout << "Missing parameters for --2pt-with-fluctuations" << endl;
                return 1;
            }
        }

        if (vm.count("compute-fermion-matrix")) {
            opts.compute_fermion_matrix = true;

            if (!vm.count("majorana")) {
                cout << "--compute-fermion-matrix only available with --majorana"
		     << endl;
                return 1;
            }
        }

        if (vm.count("anti-commutator")) {
            opts.compute_anti_commutator = true;

            if (opts.betas.size() == 0 || !vm.count("t0")) {
                cout << "Missing parameters for --anti-commutator" << endl;
                return 1;
            }
        }

        if (vm.count("t0") && (!vm.count("t1") || !vm.count("dt"))) {
            opts.t1 = opts.t0;
            opts.dt = 1.;
        }

        if (!vm.count("seed")) {
            opts.seed = get_random_seed(opts.run_name);
        }

        if (vm.count("majorana")) {
            opts.majorana = true;

            if (opts.N % 2 != 0) {
                cerr << "N must be even when --majorana is used" << endl;
                exit(1);
            }
        }

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

vector<double> get_times_vector(command_line_options& opts) {
    vector<double> times;

    for (double t = opts.t0; t <= opts.t1; t += opts.dt) {
        times.push_back(t);
    }
    
    return times;
}

void save_spectrum(KitaevHamiltonian& H, string filename) {
    Spectrum s(H);
    s.save(filename);
}

// Compute the spectrum with a mock (non-random) J tensor
void compute_mock_spectrum(command_line_options& opts) {
    MockDisorderParameter Jtensor;
    KitaevHamiltonian H(opts.N, &Jtensor);
    H.diagonalize();

    string filename = get_base_filename(opts.data_dir, opts.run_name) 
        + "-spectrum.tsv";
    save_spectrum(H, filename);
}

void print_sparseness(Mat A) {
    int num_nonzero_elements = 0;

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            if (abs(A(i,j)) > epsilon) {
                num_nonzero_elements++;
            }
        }
    }

    double sparseness = 
        ((double) num_nonzero_elements) / (A.rows() * A.cols());

    cout << "Number of non-zero elements: " << num_nonzero_elements << endl;
    cout << "Sparseness: " << sparseness << endl;
}

KitaevHamiltonian* compute_spectrum(
        command_line_options& opts, bool full_diagonalization) {
    cout << "Creating J tensor" << endl;
    KitaevDisorderParameter Jtensor(
            opts.N, opts.J, gen, !opts.real_J);
    
    cout << "Computing Hamiltonian" << endl;
    Timer timer;
    KitaevHamiltonian* H = new KitaevHamiltonian(opts.N, &Jtensor);
    timer.print();

    //print_sparseness(H->as_matrix());

    H->diagonalize(full_diagonalization, true);

    // Save the spectrum to a file
    string filename = get_base_filename(opts.data_dir, opts.run_name) 
        + "-spectrum.tsv";
    save_spectrum(*H, filename);

    return H;
}

// Compute:
//
// 1/N \sum_i Tr(e^{-bH} \chi_i(t) \chi_i(0)) =
// 1/N \sum_i Tr(e^{(-b + it) H} \chi_i e^{-itH} \chi_i)
//
void compute_majorana_2_pt_function(command_line_options& opts) {
    Timer timer;
    cout << "Computing Hamiltonian and diagonalizing" << endl;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, gen);
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    H.diagonalize_full(true);
    timer.print();

    cout << "Computing 2-point functions..." << endl;

    string filename = get_base_filename(opts.data_dir, opts.run_name) 
        + "-2pt.tsv";

    ofstream file;
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "#\tbeta\tt";
    file << "\tZ(beta)";
    file << "\tRe<G>\tIm<G>";

    if (opts.compute_2_pt_func_with_fluctuations) {
        file << "\tRe<GG*>";
    }

    file << "\n";

    vector<double> times = get_times_vector(opts);

    vector<double> Z(opts.betas.size());
    Mat correlators = Mat::Zero(times.size(), opts.betas.size());
    Mat correlators_squared = Mat::Zero(times.size(), opts.betas.size());

    if (opts.compute_2_pt_func) {
        compute_majorana_2_pt_function(
                H, opts.betas, times, opts.time_type, correlators, Z, true);
    }
    else {
        assert(opts.compute_2_pt_func_with_fluctuations);
        compute_majorana_2_pt_function_with_fluctuations(
                H, opts.betas, times, opts.time_type, 
                correlators, correlators_squared,
                Z, true);
    }

    for (unsigned ti = 0; ti < times.size(); ti++) {
        for (unsigned bi = 0; bi < opts.betas.size(); bi++) {
            double t = times[ti];
            double beta = opts.betas[bi];

            file << beta 
                << "\t" << t
                << "\t" << Z[bi]
                << "\t" << correlators(ti, bi).real()
                << "\t" << correlators(ti, bi).imag();

            if (opts.compute_2_pt_func_with_fluctuations) {
                file << "\t" << correlators_squared(ti, bi).real();
                    //<< "\t" << correlators_squared(ti, bi).imag();
            }

           file << "\n";
        }
    }

    file.close();
    timer.print();
}

// Take the absolute value squared of each matrix elements
RealMat matrix_abs_sqr(Mat M) {
    RealMat result = RealMat::Zero(M.rows(), M.cols());

    for (int a = 0; a < M.rows(); a++) {
        for (int b = 0; b < M.cols(); b++) {
            result(a,b) = norm(M(a,b));
        }
    }

    return result;
}

RealMat compute_mean_chi_energy(
        int N,
        MajoranaKitaevHamiltonian& H) {
    RealMat mean_chi_energy = RealMat::Zero(H.dim(), H.dim());

    for (int i = 0; i < N; i++) {
        Mat chi_energy_i = H.to_energy_basis(H.chi[i]);
        //Mat chi_energy_i = H.chi[i];

        /////////// Test
        if (i == 0) {
            Mat foo = H.chi[i] * H.dense_matrix();

            Mat D = Mat::Zero(H.dim(), H.dim());
            for (int j = 0; j < H.dim(); j++) {
                D(j,j) = H.evs(j);
            }

            Mat bar = H.V * chi_energy_i * D * H.V.adjoint();

            cpx x = ((foo - bar) * (foo - bar).adjoint()).trace();
            cout << "TEST: " << abs(x) 
                << " should be much smaller than "
                << abs((foo * foo.adjoint()).trace())
                << " and "
                << abs((bar * bar.adjoint()).trace())
                << endl;
        }
        /////////// End Test

        for (int a = 0; a < mean_chi_energy.rows(); a++) {
            for (int b = 0; b < mean_chi_energy.cols(); b++) {
                mean_chi_energy(a,b) += abs(chi_energy_i(a,b));
            }
        }
    }

    for (int a = 0; a < mean_chi_energy.rows(); a++) {
        for (int b = 0; b < mean_chi_energy.cols(); b++) {
            mean_chi_energy(a,b) /= N;
        }
    }

    return mean_chi_energy;
}

// This should just be the identity
RealMat compute_mean_chi_square_energy(
        int N,
        MajoranaKitaevHamiltonian& H) {
    RealMat mean_chi_energy = RealMat::Zero(H.dim(), H.dim());

    SpMat chi_sqr(H.dim(), H.dim());

    for (int i = 0; i < N; i++) {
        chi_sqr += H.chi[i] * H.chi[i];
    }

    //chi_sqr /= N;

    Mat mean_chi_sqr_energy = H.to_energy_basis(chi_sqr) / N;
    return matrix_abs_sqr(mean_chi_sqr_energy);
}

void save_matrix_in_tsv(
        command_line_options& opts, 
        string filename_suffix,
        RealMat& M) {
    ofstream file;
    string filename = 
        get_base_filename(opts.data_dir, opts.run_name) 
        + "-" + filename_suffix + ".tsv";
    file.open(filename.c_str());
    file << setprecision(precision);

    for (int a = 0; a < M.rows(); a++) {
        for (int b = 0; b < M.cols(); b++) {
            if (b > 0) { file << "\t"; }
            file << M(a,b);
        }
        file << "\n";
    }

    file.close();
}

void compute_fermion_matrix(command_line_options& opts) {
    cout << "Computing Hamiltonian" << endl;
    Timer timer;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, gen);
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    cout << "Diagonalizing" << endl;
    H.diagonalize_full(true);

    // Save the spectrum
    if (true) {
        ofstream file;
        string filename = get_base_filename(opts.data_dir, opts.run_name) + "-spectrum.tsv";
        file.open(filename.c_str());
        file << setprecision(precision);
        file << "# i eigenvalue\n";

        for (int i = 0; i < H.evs.size(); i++) {
            file << H.evs(i) << "\n";
        }

        file.close();
    }

    cout << "Computing fermion matrices..." << endl;
    timer.reset();

    //RealMat result = compute_mean_chi_energy(opts.N, H);
    
    //RealMat result = compute_mean_chi_square_energy(opts.N, H);
    //save_matrix_in_tsv(opts, "chi", result);
    
    RealMat result = matrix_abs_sqr(H.to_energy_basis(H.chi[0]));
    save_matrix_in_tsv(opts, "chi0", result);

    timer.print();
}

void compute_majorana_anti_commutator(command_line_options& opts) {
    cout << "Computing Hamiltonian" << endl;
    Timer timer;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, gen);
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    cout << "Diagonalizing" << endl;
    H.diagonalize_full(true);

    cout << "Computing anti-commutators..." << endl;
    timer.reset();
    vector<double> times = get_times_vector(opts);

    SpMat chi1 = H.chi[0];
    SpMat chi2 = H.chi[1];

    if (opts.betas.size() != 1) {
	cerr << "Must specify exactly one temperature" << endl;
	exit(1);
    }

    double beta = opts.betas[0];
    Mat exp_betaH = compute_exp_H(H.V, H.evs, -beta);
    double Z = abs(exp_betaH.trace());

    cout << "beta = " << beta << "\n"
	<< "exp(-bH).norm = " << exp_betaH.norm()
	 << "\nZ = " << Z << endl;

    ofstream file;
    string filename = get_base_filename(opts.data_dir, opts.run_name) + "-anti-commutator.tsv";
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "# t |<{chi1(t),chi2(0)}>| |<{chi1(t),chi2(0)}^2>| biggest-ev avg-abs-ev\n";

    ofstream evs_file;
    string evs_filename = get_base_filename(opts.data_dir, opts.run_name) + "-anti-commutator-evs.tsv";
    evs_file.open(evs_filename.c_str());
    evs_file << setprecision(precision);
    evs_file << "# t i ev\n";

    for (unsigned ti = 0; ti < times.size(); ti++) {
        double t = times[ti];
        cpx tau = get_tau(opts.time_type, t);

        Mat exp_tauH = compute_exp_H(H.V, H.evs, tau);
        Mat chi1_t = exp_tauH * chi1 * exp_tauH.adjoint();

        Mat anti_commutator = cpx(0,1) * (chi1_t * chi2 + chi2 * chi1_t);
        SelfAdjointEigenSolver<Mat> solver;
        solver.compute(anti_commutator, EigenvaluesOnly);
        RealVec evs = solver.eigenvalues();

	double avg_abs_ev = 0.;

	for (int i = 0; i < evs.size(); i++) {
	    avg_abs_ev += abs(evs[i]);
	}
	avg_abs_ev /= (double) evs.size();

	cpx correlator_2pt =
	    (exp_betaH * anti_commutator).trace() / Z;
	cpx correlator_4pt =
	    (exp_betaH * anti_commutator * anti_commutator).trace() / Z;

	cout << t
	     << "\t" << abs(correlator_2pt)
	     << "\t" << abs(correlator_4pt)
	     << "\t" << evs(evs.size() - 1)
	     << "\t" << evs(0)
	     << endl;

	file << t
	     << "\t" << abs(correlator_2pt)
	     << "\t" << abs(correlator_4pt)
	     << "\t" << evs(evs.size() - 1)
	     << "\t" << avg_abs_ev
	     << endl;

        for (int i = 0; i < evs.size(); i++) {
            evs_file << t << "\t" << i << "\t" << evs(i) << endl;
	}
    }

    file.close();
    evs_file.close();
}

// Compute the trace of a sparse matrix
cpx sparse_trace(SpMat A) {
    // Eigen can't take the trace of a sparse
    // matrix, so we have to do it by hand.
    cpx trace = 0.;
    for (int i = 0; i < A.outerSize(); i++) {
        trace += A.coeff(i,i);
    }
    return trace;
}

void compute_majorana_spectrum(command_line_options& opts) {
    cout << "Creating J tensor" << endl;
    MajoranaKitaevDisorderParameter Jtensor(opts.N, opts.J, gen);
    //cout << "J:\n" << Jtensor.to_string() << endl;

    cout << "Computing Hamiltonian" << endl;
    Timer timer;
    MajoranaKitaevHamiltonian H(opts.N, &Jtensor);
    timer.print();

    // cout << "Tr(H) = " << abs(sparse_trace(H.matrix)) << endl;
    // cout << "Tr(H^2) = "
    // 	 << abs(sparse_trace(H.matrix*H.matrix)) << endl;
    // cout << "Tr(H^3) = "
    // 	 << abs(sparse_trace(H.matrix*H.matrix*H.matrix)) << endl;
    // cout << "Tr(H^4) = "
    // 	 << abs(sparse_trace(H.matrix*H.matrix*H.matrix*H.matrix)) << endl;
    // exit(0);

    //print_sparseness(H.dense_matrix());

    /*cout << "Computing Naive Hamiltonian" << endl;
    timer.reset();
    NaiveMajoranaKitaevHamiltonian H_naive(opts.N, &Jtensor);
    timer.print();*/

    cout << "Diagonalizing" << endl;

    timer.reset();
    H.diagonalize(true); // Only compute eigenvalues
    timer.print();

    // Save the spectrum to a file
    string filename = get_base_filename(opts.data_dir, opts.run_name) 
        + "-spectrum.tsv";

    ofstream file;
    file.open(filename.c_str());
    file << setprecision(precision);
    file << "#\tcharge-parity\teigenvalue\n";

    for (int i = 0; i < H.even_charge_parity_evs.size(); i++) {
        file << "1\t" << H.even_charge_parity_evs(i) << "\n";
    }

    for (int i = 0; i < H.odd_charge_parity_evs.size(); i++) {
        file << "-1\t" << H.odd_charge_parity_evs(i) << "\n";
    }

    file.close();
}

// Compute \sum_{i=1}^N < c^*_i(t) c_i >
void compute_2_point_function(
        int N,
        vector<double> betas,
        vector<double> times,
        KitaevHamiltonian& H,
        TIME_TYPE time_type,
        TwoPointOutput* out) {
    Mat correlators = compute_2_point_function(N, betas, times, H, time_type);

    for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
        double beta = betas[beta_i];

        for (unsigned time_i = 0; time_i < times.size(); time_i++) {
            double t = times[time_i];
            cpx correlator = correlators(beta_i, time_i);

            if (out != 0) {
                out->write(1./beta, t, correlator);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cout << setprecision(precision);
    cout << "Command line options:\n" << opts.to_s() << endl;
    gen = new boost::random::mt19937(opts.seed);

    ///////////////////////////
    // MajoranaDisorderParameter* J = new MockMajoranaDisorderParameter();
    // MajoranaKitaevHamiltonian H(opts.N, J);
    // H.diagonalize();
    // RealVec evs = H.all_eigenvalues();
    // std::sort(evs.data(), evs.data() + evs.size());
    // cout << "Eigenvalues:" << endl;
    // for (int i = 0; i < evs.size(); i++) {
    // 	cout << i << "\t" << evs[i] << endl;
    // }
    // return 0;
    ///////////////////////////

    bool full_diagonalization = opts.compute_2_pt_func || opts.compute_fermion_matrix;

    if (opts.majorana) {
        if (opts.compute_2_pt_func ||
            opts.compute_2_pt_func_with_fluctuations) {
            compute_majorana_2_pt_function(opts);
        }
        else if (opts.compute_fermion_matrix) {
            compute_fermion_matrix(opts);
        }
        else if (opts.compute_anti_commutator) {
            compute_majorana_anti_commutator(opts);
        }
        else {
            compute_majorana_spectrum(opts);
        }
    }
    else {
        KitaevHamiltonian* H = compute_spectrum(opts, full_diagonalization);

        if (opts.compute_2_pt_func) {
            cout << "Computing 2-point functions..." << endl;

            vector<double> times = get_times_vector(opts);

            string filename = get_base_filename(opts.data_dir, opts.run_name) 
                + "-2pt.tsv";
            TwoPointFileOutput out(filename);

            Timer timer;
            compute_2_point_function(
                    opts.N, opts.betas, times, *H, opts.time_type, &out);
            timer.print();

            out.close();
        }
    }
    //compute_mock_spectrum(opts);
    return 0;
}


