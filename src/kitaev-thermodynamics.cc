/*
 * Compute thermodynamic quantities given a spectrum.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/program_options.hpp>
#include "Spectrum.h"

using namespace std;
namespace po = boost::program_options;

struct command_line_options {
    string spectrum_file;
    string output_file;
    double T_start;
    double T_end;
    double T_step;
    bool majorana;
};

int parse_command_line_options(
        int argc, char** argv, command_line_options& opts) {
    opts.majorana = false;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("spectrum-file", 
             po::value<string>(&opts.spectrum_file)->required(),
             "file containing the spectrum")
            ("output-file", 
             po::value<string>(&opts.output_file)->required(),
             "output filename")
            ("T-start",
             po::value<double>(&opts.T_start)->required(),
             "first value of T/J")
            ("T-end",
             po::value<double>(&opts.T_end)->required(),
             "last value of T/J")
            ("T-step",
             po::value<double>(&opts.T_step)->required(),
             "step of T/J")
            ("majorana",
             "this is a spectrum of the Majorana model")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("majorana")) {
            opts.majorana = true;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

double compute_partition_function(RealVec& evs, double beta) {
    double Z = 0.;

    for (int i = 0; i < evs.size(); i++) {
        double Ei = evs(i);
        //cout << "Ei=" << Ei << "\texp=" << exp(-beta * Ei) << endl;
        Z += exp(-beta * Ei);
    }

    return Z;
}

void write_thermodynamic_quantities(
        ofstream& output, RealVec& evs, int N, double T) {
    double beta = 1./T;
    double Z = compute_partition_function(evs, beta);

    // Free energy
    double F = - log(Z) / beta;

    // Expected energy <E>
    double E = 0.;

    // Ground state energy
    double E0 = evs(0);

    // <E^2>
    double Esqr = 0.;

    for (int i = 0; i < evs.size(); i++) {
        double Ei = evs(i);
        E += Ei * exp(-beta * Ei);
        Esqr += Ei * Ei * exp(-beta * Ei);

        if (Ei < E0) {
            E0 = Ei;
        }
    }

    E /= Z;
    Esqr /= Z;

    // Entropy
    double S = beta * (E - F);

    /*cout << "beta=" << beta << "\tZ=" << Z 
        << "\tE=" << E << "\tF=" << F << "\tS=" << S << endl;*/

    cout << "E0 = " << E0 << endl;

    output << T 
        << "\t" << S/N 
        << "\t" << E/N 
        << "\t" << F/N 
        << "\t" << (Esqr - E*E)
        << "\t" << (Esqr - E*E) / ((E-E0)*(E-E0))
        << endl;
}

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cout << setprecision(precision);

    RealVec evs;
    int N;

    if (opts.majorana) {
        MajoranaSpectrum s(opts.spectrum_file);
        evs = s.all_eigenvalues();
        N = s.majorana_N;
    }
    else {
        Spectrum s(opts.spectrum_file);
        evs = s.all_eigenvalues();
        N = s.N;
    }

    ofstream output;
    output.open(opts.output_file.c_str());
    output << setprecision(precision);
    output << "# T/J S/N E/N F/N Var(E) Var(E)/<E-E0>^2\n";

    for (double T = opts.T_start; T <= opts.T_end; T += opts.T_step) {
        write_thermodynamic_quantities(output, evs, N, T);
    }

    output.close();
    return 0;
}
