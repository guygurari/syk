/*
 * Compute time-dependent partition function for a single sample.
 *
 * Input file structure is a TSV where:
 * - First column = sample (run) number
 * - Second column = charge (Q in Dirac or charge parity in Majorana)
 * - Last column = eigenvalue
 * - If there is no charge column then --no-Q-column must be specified.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <boost/program_options.hpp>
#include <boost/multi_array.hpp>
#include "defs.h"
#include "Timer.h"
#include "TSVFile.h"

using namespace std;
namespace po = boost::program_options;

typedef boost::multi_array<cpx, 2> Z_array;
typedef Z_array::index Z_index;

struct command_line_options {
    string spectrum_file;
    string output_file;
    vector<double> betas;
    bool t_provided;
    double t_start;
    double t_end;
    double t_step;
    double Z_cutoff;
    bool scan_for_cutoff;
    bool log_t_step;
    bool no_Q_column;
    int Q;
    //vector<string> input_files;
};

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    opts.Q = -1;
    opts.no_Q_column = false;
    string betas_str;

    opts.t_provided = false;
    opts.t_start = 0.1;
    opts.t_end = 1;
    opts.t_step = 1;
    opts.log_t_step = false;
    opts.scan_for_cutoff = false;

    // Default beta values
    opts.betas.push_back(0);
    opts.betas.push_back(1);
    opts.betas.push_back(5);
    opts.betas.push_back(10);
    opts.betas.push_back(20);
    opts.betas.push_back(30);

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("spectrum-file", 
             po::value<string>(&opts.spectrum_file)->required(),
             "combined spectrum filename (contains multiple samples)")
            ("output-file", 
             po::value<string>(&opts.output_file)->required(),
             "output filename")
            ("betas",
             po::value<string>(&betas_str),
             "comma-separated list of beta values")
            ("t-start",
             po::value<double>(&opts.t_start),
             "first value of time")
            ("t-end",
             po::value<double>(&opts.t_end),
             "last value of time")
            ("t-step",
             po::value<double>(&opts.t_step),
             "step of time")
            ("Z-cutoff",
             po::value<double>(&opts.Z_cutoff),
             "just scan for Z/Z(0) values greater than the cutoff")
            ("log-t-step",
             "t-step is actually the step of log(t)")
            ("no-Q-column",
             "No Q column in the data")
            ("Q",
             po::value<int>(&opts.Q),
             "only trace over the given Q sector (or charge parity in Majorana)")
            /*("input-file", po::value< vector<string> >(), "input file")*/
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        // All the arguments without --blah go into input-file
        // When using multiple input files
        /*po::positional_options_description p;
        p.add("input-file", -1);
        po::store(po::command_line_parser(argc, argv).
                options(desc).positional(p).run(), vm);*/

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (vm.count("no-Q-column")) {
            opts.no_Q_column = true;
        }

        if (vm.count("log-t-step")) {
            opts.log_t_step = true;

            if (opts.t_start <= 0) {
                cerr << "When using log-t-step, initial time must be positive." << endl;
                return 1;
            }
        }

        if (vm.count("t-start")) {
            opts.t_provided = true;
        }

        // Process comma-separated list
        if (vm.count("betas")) {
            opts.betas.clear();
            istringstream ss(betas_str);
            string beta_s;

            while (getline(ss, beta_s, ',')) {
                opts.betas.push_back(atof(beta_s.c_str()));
            }
        }

	if (vm.count("Z-cutoff")) {
	    opts.scan_for_cutoff = true;
	}

        /*if (vm.count("input-file")) {
            opts.input_files = vm["input-file"].as< vector<string> >();
        }*/

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

// Compute Z = Tr(e^{-iHT-\beta H}) for the given list of H eigenvalues
cpx compute_sample_Z(double beta, double t, vector<double>& evs) {
    cpx Z = 0;

    for (unsigned i = 0; i < evs.size(); i++) {
        double Ei = evs[i];
        Z += exp( -cpx(beta,t) * Ei );
    }

    return Z;
}

void process_sample_evs(
        vector<double>& betas,
        vector<double>& times,
        vector<double>& evs,
        Z_array& z_t,
        vector<cpx>& Z_t0
        ) {
    if (evs.size() == 0) {
        return;
    }

    for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
        // Compute at t=0 so we can normalize
        double beta = betas[beta_i];
        cpx sample_Z_t0 = compute_sample_Z(beta, 0, evs);
        Z_t0[beta_i] = sample_Z_t0;


        for (unsigned t_i = 0; t_i < times.size(); t_i++) {
            double t = times[t_i];

            // Tr exp(-beta H -iHt)
            cpx sample_Z = compute_sample_Z(beta, t, evs);
            z_t[beta_i][t_i] += sample_Z / sample_Z_t0;
        }
    }
}

int main(int argc, char *argv[]) {
    command_line_options opts;
    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cout << setprecision(precision);

    // Prepare array of times
    //
    // With log-t-step on, --t-step=c specifies that d log(t) = c,
    // so dt = c*t.
    vector<double> times;

    if (opts.t_provided) {
        for (double t = opts.t_start; 
                t <= opts.t_end; 
                t += opts.log_t_step ? (opts.t_step * t) : opts.t_step) {
            times.push_back(t);
        }
    }
    else {
        double dt_early = 0.02;
        double early_t = 200;
        double late_t = 10000000;

        for (double t = dt_early; t < early_t; t += dt_early) {
            times.push_back(t);
        }

        double log_t_step = 0.01;

        for (double t = early_t; t < late_t; t += t * log_t_step) {
            times.push_back(t);
        }
    }

    // Prepare array of betas
    /*vector<double> betas;
    for (double beta = opts.beta_start; 
         beta <= opts.beta_end; 
         beta += opts.beta_step) {
        betas.push_back(beta);
    }*/

    // Read sample spectrum file
    TSVFile file(opts.spectrum_file);

    vector<double> sample_evs;

    Timer timer;
 
    while (true) {
        double ev;

        if (opts.no_Q_column) {
            file >> ev;
            if (file.eof()) break;
        }
        else {
            int Q;
            file >> Q >> ev;
            if (file.eof()) break;

            if (opts.Q >= 0 && opts.Q != Q) {
                continue;
            }
        }

        sample_evs.push_back(ev);
    }

    //////////////////////////

    if (opts.scan_for_cutoff) {
        double beta = opts.betas[0];
        cpx sample_Z_t0 = compute_sample_Z(beta, 0, sample_evs);

        for (unsigned t_i = 0; t_i < times.size(); t_i++) {
            double t = times[t_i];

            // Tr exp(-beta H -iHt)
            cpx sample_Z = compute_sample_Z(beta, t, sample_evs);
            double sample_Z_norm = abs(sample_Z) / abs(sample_Z_t0);

            if (sample_Z_norm > opts.Z_cutoff) {
                cout << t << "\t" << sample_Z_norm << endl;
            }
        }
    }
    else {
	Z_array z_t(boost::extents[opts.betas.size()][times.size()]);
	vector<cpx> Z_t0(opts.betas.size());

	// Process the sample
	process_sample_evs(
			   opts.betas, times, 
			   sample_evs, 
			   z_t, Z_t0);
	timer.print();

	// Write z(t)
	ofstream output;
	output.open(opts.output_file.c_str());
	output << setprecision(precision);
	output << "#\tbeta\tt"
	       << "\tRe(z)"
	       << "\tIm(z)"
	       << "\tZ(t=0)"
	       << "\n";

	for (unsigned i = 0; i < opts.betas.size(); i++) {
	    for (unsigned j = 0; j < times.size(); j++) {
		output 
		    << opts.betas[i]
		    << "\t" << times[j] 
		    << "\t" << real(z_t[i][j])
		    << "\t" << imag(z_t[i][j])
		    << "\t" << real(Z_t0[i])
		    << "\n";
	    }
	}
 
	file.close();
	output.close();
    }
    
    return 0;
}
