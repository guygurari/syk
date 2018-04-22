/*
 * Compute the disorder average of Majorana 2-point functions.
 * Input files are 'maj-Nxx-runY-2pt.tsv'.
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

    vector<double> sum_Z;
    vector<cpx> sum_G;
    vector<cpx> sum_G_over_Z;
    vector<double> sum_GGstar;

    vector<TSVFile*> in;
    for (int sample = 0; sample < num_samples; sample++) {
        //cout << "sample = " << sample << endl;
        TSVFile file(opts.input_files[sample]);

        // A 'line' is a (beta,t) pair
        unsigned line = 0;

        while (true) {
            double beta, t, samp_Z, samp_re_G, samp_im_G, samp_GGstar;
            file >> beta >> t >> samp_Z 
                >> samp_re_G >> samp_im_G >> samp_GGstar;

            cpx samp_G(samp_re_G, samp_im_G);

            // No more (beta,t) values, we're done
            if (file.eof()) {
                break;
            }

            // In the first sample, we add an entry for each beta,t pair 
            // (each line in the file) we find. In following samples we
            // already have all the elements in place.
            if (sample == 0) {
                betas.push_back(beta);
                times.push_back(t);

                sum_Z.push_back(0);
                sum_G.push_back(cpx(0,0));
                sum_G_over_Z.push_back(cpx(0,0));
                sum_GGstar.push_back(0);
            }

            if (sum_Z.size() <= line) {
                cout << "Error:" << endl;
                cout << "file = " << opts.input_files[sample] << endl;
                cout << "beta = " << beta << "\tt = " << t << endl;
                cout << "size = " << sum_Z.size() << "\tline = " << line << endl;
            }

            assert(sum_Z.size() > line);
            assert(sum_G.size() > line);
            assert(sum_G_over_Z.size() > line);
            assert(sum_GGstar.size() > line);

            sum_Z[line] += samp_Z;
            sum_G[line] += samp_G;
            sum_G_over_Z[line] += samp_G / samp_Z;
            sum_GGstar[line] += samp_GGstar;

            line++;
        }

        file.close();
    }

    assert(betas.size() == times.size());
    assert(betas.size() == sum_Z.size());
    assert(betas.size() == sum_G.size());
    assert(betas.size() == sum_GGstar.size());

    ofstream out;
    out.open(opts.output_file.c_str());
    out << setprecision(precision);
    out << "#\tbeta\tt"
        << "\t<Z>\tRe<G>/<Z>\tIm<G>/<Z>"
        << "\tRe<G/Z>\tIm<G/Z>"
        << "\t<GG*>/<Z>"
        << "\t(<GG*>/<Z>-|<G>/<Z>|^2)/(|<G>/<Z>|^2)"
        << "\n";

    for (unsigned line = 0; line < betas.size(); line++) {
        double beta = betas[line];
        double t = times[line];

        double avg_Z = sum_Z[line] / (double) num_samples;
        cpx avg_G = sum_G[line] / (double) num_samples;
        cpx avg_G_over_Z = sum_G_over_Z[line] / (double) num_samples;
        double avg_GGstar = sum_GGstar[line] / (double) num_samples;

        double G_corr_norm_sqr = norm(avg_G) / (avg_Z*avg_Z);
        double fractional_variance =
            (avg_GGstar / avg_Z - G_corr_norm_sqr) / G_corr_norm_sqr;

        out << beta
            << "\t" << t
            << "\t" << avg_Z
            << "\t" << real(avg_G) / avg_Z
            << "\t" << imag(avg_G) / avg_Z
            << "\t" << real(avg_G_over_Z)
            << "\t" << imag(avg_G_over_Z)
            << "\t" << avg_GGstar / avg_Z
            << "\t" << fractional_variance
            << "\n";
    }
    
    out.close();
    return 0;
}

