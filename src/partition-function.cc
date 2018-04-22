/*
 * Compute time-dependent partition function.
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
    bool log_t_step;
    bool no_Q_column;
    int Q;
    int max_samples;
    bool single_sample;
    //vector<string> input_files;
};

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    opts.Q = -1;
    opts.max_samples = -1;
    opts.single_sample = false;
    opts.no_Q_column = false;
    string betas_str;

    opts.t_provided = false;
    opts.t_start = 0.1;
    opts.t_end = 1;
    opts.t_step = 1;
    opts.log_t_step = false;

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
             po::value<string>(&betas_str)->required(),
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
            ("log-t-step",
             "t-step is actually the step of log(t)")
            ("max-samples",
             po::value<int>(&opts.max_samples),
             "use at most this many samples")
            ("single-sample",
             "the spectrum file is for a single sample with no sample column")
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

        if (vm.count("single-sample")) {
            opts.single_sample = true;
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
        istringstream ss(betas_str);
        string beta_s;

        while (getline(ss, beta_s, ',')) {
            opts.betas.push_back(atof(beta_s.c_str()));
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
        Z_array& zzstar_t,
        int& num_samples
        ) {
    if (evs.size() == 0) {
        return;
    }

    num_samples++;

    for (unsigned beta_i = 0; beta_i < betas.size(); beta_i++) {
        // Compute at t=0 so we can normalize
        double beta = betas[beta_i];
        cpx sample_z_t0 = compute_sample_Z(beta, 0, evs);

        for (unsigned t_i = 0; t_i < times.size(); t_i++) {
            double t = times[t_i];

            // Tr exp(-beta H -iHt)
            cpx sample_Z = compute_sample_Z(beta, t, evs);

            z_t[beta_i][t_i] += sample_Z / sample_z_t0;
            zzstar_t[beta_i][t_i] += norm(sample_Z) / norm(sample_z_t0);
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
        double late_t = 100000;

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

    // Read spectrum file sample-by-sample.
    //
    // For each sample, read its spectrum and compute the various
    // partition function quantities, add them to the disorder average.
    //
    // Count the samples so we can divide at the end.
    TSVFile file(opts.spectrum_file);
    int num_samples = 0;
    int last_sample = -1;

    Z_array z_t(boost::extents[opts.betas.size()][times.size()]);
    Z_array zzstar_t(boost::extents[opts.betas.size()][times.size()]);
    vector<double> sample_evs;

    Timer timer;
 
    while (true) {
        int sample;
        double ev;

        if (!opts.single_sample) {
            file >> sample;
        }

        if (opts.no_Q_column) {
            file >> ev;

            if (file.eof()) {
                break;
            }
        }
        else {
            int Q;
            file >> Q >> ev;

            if (file.eof()) {
                break;
            }

            if (opts.Q >= 0 && opts.Q != Q) {
                continue;
            }
        }

        if (!opts.single_sample && (last_sample != sample)) {
            //cout << "Sample " << sample << endl;
            last_sample = sample;

            if (sample > 1) {
                timer.print();
                cout << "Processing sample " << sample-1 << endl;
                timer.reset();
            }

            // Process the old sample
            process_sample_evs(
                    opts.betas, times, 
                    sample_evs, 
                    z_t, zzstar_t,
                    num_samples);

            // This ev is the first one in the new sample
            sample_evs.clear();

            // We're done here
            if (opts.max_samples >= 0 && num_samples >= opts.max_samples) {
                break;
            }
        }

        sample_evs.push_back(ev);
    }

    // Process the last sample
    process_sample_evs(
            opts.betas, times, 
            sample_evs, 
            z_t, zzstar_t,
            num_samples);
    timer.print();

    // Normalize by number of samples
    for (unsigned i = 0; i < opts.betas.size(); i++) {
        for (unsigned j = 0; j < times.size(); j++) {
            z_t[i][j] /= num_samples;
            zzstar_t[i][j] /= num_samples;
        }
    }

    cout << "Processed " << num_samples << " samples" << endl;

    // Write the disorder averaged quantities
    ofstream output;
    output.open(opts.output_file.c_str());
    output << setprecision(precision);
    output << "#\tbeta\tt"
        << "\t<Re z(t)>"
        << "\t<Im z(t)>"
        << "\tg=<z(t)z*(t)>"
        << "\tg_c=<z(t)z*(t)>-<z(t)><z*(t)>"
        << "\tg_d=|<z(t)>|^2"
        << "\n";

    for (unsigned i = 0; i < opts.betas.size(); i++) {
        for (unsigned j = 0; j < times.size(); j++) {
            output 
                << opts.betas[i]
                << "\t" << times[j] 
                << "\t" << real(z_t[i][j])
                << "\t" << imag(z_t[i][j])
                << "\t" << real(zzstar_t[i][j])
                << "\t" << real(zzstar_t[i][j] - SQR(abs(z_t[i][j])))
                << "\t" << SQR(abs(z_t[i][j]))
                << "\n";
        }
    }
 
    file.close();
    output.close();
    return 0;
}
