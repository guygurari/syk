#!/usr/bin/env perl
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
	print "Usage: $Script [--help] [-n] [--experiment lanc-N42]\n";
}

my $help = 0;
my $dry_run = 0;
my $chosen_experiment;

GetOptions(
	'help' => \$help,
    'n' => \$dry_run,
    'experiment=s' => \$chosen_experiment,
);

if ($help) {
	usage();
	exit 0;
}

if (scalar(@ARGV) != 0) {
	usage();
	exit 1;
}

my $exp_base_dir = "data/lanczos";
my $processed_base_dir = "data/lanczos-processed";
my $spectra_dir = "data/spectra";

my @experiments;

if (defined $chosen_experiment) {
    push @experiments, $chosen_experiment
}
else {
    @experiments = map { s/$exp_base_dir//; s/\///g; $_ } glob("$exp_base_dir/*/");
}

my $min_evs = {
    short => 120,
    long => 800,
};

my $max_evs = {
    short => 500,
    long => 3000,
};

my @betas = (1, 25, 50);
my $t0 = 0.02;
my $t1 = 1e7;
my $step = 0.1;

foreach my $exp (@experiments) {
    print "\n\n===== Processing $exp =====\n\n";
    my $input_dir = "$exp_base_dir/$exp";
    my $processed_dir = "$processed_base_dir/$exp";

    die "Cannot find input dir $input_dir" unless -d $input_dir;

    if (! -d $processed_dir) {
        execute("mkdir -p $processed_dir");
    }

    my $min = $min_evs->{get_experiment_type($exp)};
    my $max = $max_evs->{get_experiment_type($exp)};
    execute("./separate-lanczos-evs.py --max-evs $max --min-evs $min $input_dir/$exp-run*.tsv");

    # Get the good low/high energies
    my @ev_files_list = glob("$processed_dir/$exp-run*-spectrum.tsv");
    my $ev_files = join(" ", @ev_files_list);

    # Concat the spectra into a single file
    execute("./concat-lanczos-spectra.pl --output $spectra_dir/$exp-spectrum.tsv $processed_dir/$exp-run*-spectrum.tsv");

    # Compute partition function for each sample
    my $betas_s = join(",", @betas);
    execute("./compute-partition-function.pl --betas $betas_s --t-start $t0 --t-end $t1 --log-t-step --t-step $step $ev_files");

    # Compute disorder-averged Z
    my @Z_files_list = glob("$processed_dir/*-Z.tsv.bz2");
    my $Z_files = join(" ", @Z_files_list);
    execute("./partition-function-disorder-average --output-file $spectra_dir/$exp-Z.tsv $Z_files");
}

sub execute {
    my $cmd = shift @_;
    print "$cmd\n";

    if (!$dry_run) {
        system($cmd);
        die "Command failed" if $? >> 8;
    }
}

# gndlanc (ground-states) are not processed by this script
# because they don't need sophisticated good ev extraction
# or partition function calculation. Instead use 
# extract-ground-states.xx
sub get_experiment_type {
    my $exp = shift;

    if ($exp =~ /short/) {
        return 'short';
    }
    else {
        return 'long';
    }
}
