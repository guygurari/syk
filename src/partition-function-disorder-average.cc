/*
 * Compute the disorder average of partition function quantities.
 * Input files are results of partition-function-single-sample to
 * disorder average over.
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
    vector<string> input_files;
    string output_file;
};

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("output-file", 
             po::value<string>(&opts.output_file)->required(),
             "output filename")
            ("input-file", po::value< vector<string> >(), "input files e.g. run1-Z.tsv.bz2, several files can be specified without the flag")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        // All the arguments without --blah go into input-file
        // When using multiple input files
        po::positional_options_description p;
        p.add("input-file", -1);
        po::store(po::command_line_parser(argc, argv).
                options(desc).positional(p).run(), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (vm.count("input-file")) {
            opts.input_files = vm["input-file"].as< vector<string> >();
        }

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char *argv[]) {
    cout << setprecision(precision);
    command_line_options opts;
    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    int num_samples = opts.input_files.size();

    if (num_samples == 0) {
        cout << "No samples provided." << endl;
        return 0;
    }

    vector<double> betas;
    vector<double> times;

    vector<cpx> sum_Z;
    vector<double> sum_G;
    vector<cpx> sum_ZZ;
    vector<cpx> sum_ZG;
    vector<double> sum_Z4;

    vector<double> sum_Z_t0;
    vector<double> sum_G_t0;

    vector<TSVFile*> in;
    for (int sample = 0; sample < num_samples; sample++) {
        //cout << "sample = " << sample << endl;
        TSVFile file(opts.input_files[sample]);
        unsigned line = 0;

        while (true) {
            double beta, t, samp_re_z, samp_im_z, samp_Z_t0;
            file >> beta >> t >> samp_re_z >> samp_im_z >> samp_Z_t0 ;

            cpx samp_z(samp_re_z, samp_im_z);
            cpx samp_Z = samp_z*samp_Z_t0;

            // No more (beta,t) values, we're done
            if (file.eof()) {
                break;
            }

            if (sample == 0) {
                betas.push_back(beta);
                times.push_back(t);

                sum_Z.push_back(cpx(0,0));
                sum_G.push_back(0);

                sum_ZZ.push_back(cpx(0,0));
                sum_ZG.push_back(cpx(0,0));
                sum_Z4.push_back(0);

                sum_Z_t0.push_back(0);
                sum_G_t0.push_back(0);
            }

            if (sum_Z.size() <= line) {
                cout << "Error:" << endl;
                cout << "file = " << opts.input_files[sample] << endl;
                cout << "beta = " << beta << "\tt = " << t << endl;
                cout << "size = " << sum_Z.size() << "\tline = " << line << endl;
            }

            assert(sum_Z.size() > line);
            assert(sum_G.size() > line);

            sum_Z[line] += samp_Z;
            sum_G[line] += norm(samp_Z);

            sum_ZZ[line] += pow(samp_Z,2);
            sum_ZG[line] += samp_Z*norm(samp_Z);
            sum_Z4[line] += pow(norm(samp_Z),2);

            sum_Z_t0[line] += samp_Z_t0;
            sum_G_t0[line] += pow(samp_Z_t0,2);

            line++;
        }

        file.close();
    }

    assert(betas.size() == times.size());
    assert(betas.size() == sum_Z.size());
    assert(betas.size() == sum_G.size());

    ofstream out;
    out.open(opts.output_file.c_str());
    out << setprecision(precision);
    out << "#\tbeta\tt"
        << "\t<Re Z(t)>/<Z(t=0)>"
        << "\t<Im Z(t)>/<Z(t=0)>"
        << "\tG=<Z(t)Z*(t)>/<Z(t=0)>^2"
        << "\tG_c=<Z(t)Z*(t)>/<Z(t=0)>^2-<Z(t)><Z*(t)>/<Z(t=0)>^2"
        << "\tG_d=|<Z(t)>|^2/<Z(t=0)>^2"
        << "\tG4=<|DeltaZ|^4>/<Z(t=0>^4"
        << "\tG4c=<|DeltaZ|^4>/<Z(t=0>^4-<|DeltaZ|^2>^2/<Z(t=0)>^4"
        << "\n";

    for (unsigned line = 0; line < betas.size(); line++) {
        double beta = betas[line];
        double t = times[line];

        cpx avg_Z = sum_Z[line] / sum_Z_t0[line];
        double avg_G = (double) num_samples * sum_G[line] / pow(sum_Z_t0[line],2);
        double avg_Gd = norm(avg_Z);
        double avg_Gc = avg_G - avg_Gd;
	double avg_G4 = (sum_Z4[line] / (double) num_samples - 4*real(sum_ZG[line]*conj(sum_Z[line])) / (double) pow(num_samples,2) + 4*norm(sum_Z[line])*sum_G[line] / (double) pow(num_samples,3) + 2*real(sum_ZZ[line]*pow(conj(sum_Z[line]),2)) / (double) pow(num_samples,3) - 3*pow(norm(sum_Z[line]),2) / (double) pow(num_samples,4)) / (pow(sum_Z_t0[line],4) / (double) pow(num_samples,4));
	double avg_G4c = avg_G4 - pow(avg_Gc,2);

        out << beta
            << "\t" << t
            << "\t" << real(avg_Z)
            << "\t" << imag(avg_Z)
            << "\t" << avg_G
            << "\t" << avg_Gc
            << "\t" << avg_Gd
            << "\t" << avg_G4
            << "\t" << avg_G4c
            << "\n";
    }
    
    out.close();
    return 0;
}
