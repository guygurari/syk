#!/usr/bin/perl -w

use strict;
use warnings;
use FindBin qw($Script $Dir);
use Getopt::Long;
use KitaevUtils;

my $random_matrix_prog_name = "./random-matrix";
my $Z_prog = "./compute-partition-function.pl";

sub factorial {
    my $x = shift;

    if ($x == 0) {
        return 1;
    }
    elsif ($x > 0) {
        return $x * factorial($x-1);
    }
    else {
        die "Bad number: $x";
    }
}

# n choose k
sub binomial {
    my ($n, $k) = @_;
    return factorial($n) / (factorial($k) * factorial($n-$k));
}

# Most common parameters to tweak

my $post_process_partition_function = 1;

# === Dirac ===
#my @ensembles = ('GUE', 'GOE');
#my @K_values = (binomial(10, 5)); 
#my @K_values = (binomial(12, 6)); 
#my @K_values = (binomial(14, 7)); 
#my @K_values = (binomial(16, 8)); 

#my $min_run = 1;
#my $max_run = 300;

# === Majorana ===
# Majorana ensembles by N_m mod 8
my $majorana_ensembles = {
    0 => 'GOE',
    1 => 'GOE',
    2 => 'GUE',
    3 => 'GSE',
    4 => 'GSE',
    5 => 'GSE',
    6 => 'GUE',
    7 => 'GOE'
};

my $N_majorana = 26;
my $K_value = 2**($N_majorana/2) / 2;
#my $ensemble = $majorana_ensembles->{$N_majorana % 8};
my $ensemble = 'GUE';

#my $target_evs = 2500000;
#my $num_runs = int($target_evs / $K_value);

my $min_run = 1681;
my $max_run = 1681;
#my $max_run = 10000;

my $num_runs = $max_run - $min_run + 1;

my @K_values = ($K_value);
my @ensembles = ($ensemble);

print "N_m = $N_majorana\nK = $K_value\nensemble = $ensemble\nnum runs = $num_runs\n";


# === Sparse ===

#my @ensembles = ("SparseKlogK");
#my $K_value = 2**(28/2);
#my @K_values = ($K_value);
#
#my $target_evs = 1000000;
#my $num_runs = int($target_evs / $K_value);
#
#print "Need $num_runs samples\n";
#
#my $min_run = 25;
#my $max_run = 30;


# === Do it ===

my $help = 0;

# Don't submit jobs, only print the command lines
my $dry_run = 0;

# Run locally instead of submitting the jobs
my $run_locally = 0;

sub usage {
        print "Usage: $Script [--help] [-n] [--local]\n";
}

GetOptions(
        'help' => \$help,
        'n' => \$dry_run,
        'local' => \$run_locally,
        );

if ($help) {
    usage();
    exit 0;
}

submit_jobs();

sub submit_jobs {
    for my $ensemble (@ensembles) {
        for my $K (@K_values) {
            for my $i ($min_run .. $max_run) {
                my $run_name = "rnd-K${K}-${ensemble}-run${i}";

                my $params = prepare_params($run_name, $K, $ensemble);
                submit_job($run_name, $random_matrix_prog_name, $params);
            }
        }
    }
}

sub prepare_params {
    my ($run_name, $K, $ensemble) = @_;

    my $params = [
        "--run-name $run_name",
        "--K $K",
        "--data-dir " . KitaevUtils::data_dir(),
        "--$ensemble",
    ];

    return $params;
}

sub submit_job {
    my ($full_run_name, $prog, $params) = @_;

    my $prog2;
    my $prog2_params;

    if ($post_process_partition_function) {
        $prog2 = $Z_prog;
        $prog2_params = [
            KitaevUtils::data_dir() . "/${full_run_name}-spectrum.tsv",
            "--no-Q-column",
        ];
    }

    while (!KitaevUtils::can_submit_jobs()) {
        print "Cannot submit jobs, waiting for queue to clear...\n";
        sleep 60;
    }

    KitaevUtils::submit_job(
        $full_run_name,
        $dry_run, $run_locally,
        $prog, $params,
        $prog2, $prog2_params,
        #$slurm_mem_mb, $slurm_time,
    );
}

