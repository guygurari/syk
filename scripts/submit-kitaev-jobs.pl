#!/usr/bin/perl -w

use strict;
use warnings;
use FindBin qw($Script $Dir);
use Getopt::Long;
use KitaevUtils;

my $kitaev_prog = "./kitaev";
my $Z_prog = "./compute-partition-function.pl";

# Required Majorana mamory in GB
my $majorana_mem_gb = {
    20 => 1,
    22 => 1,
    24 => 1,
    26 => 1,
    28 => 3,
    30 => 12,
    32 => 48,
    34 => 192,
};

my $majorana_time = {
    28 => '01:00:00',
    30 => '02:00:00',
    32 => '48:00:00',
    34 => '180:00:00',
};

# Most common parameters to tweak
my $majorana = 1;
my $real_J = 0;
my $J = 1;
#my @N_values = (6, 8, 10, 12); 
my @N_values = (22); 

my $target_evs = 2500000;

my $min_run = 1;
my $max_run = 60;
#my $max_run = num_needed_samples($target_evs, $N_values[0]);

print "Num runs: " . ($max_run - $min_run + 1) . "\n\n";

# Whether to run Z post-processing
my $post_process_partition_function = 0;

my $slurm_mem_mb;
my $slurm_time;

# correlators
my $two_pt = 0;
my $two_pt_with_fluctuations = 1;

#my $T0 = 0;
#my $T1 = 1;
#my $dT = 1;
#my $treat_T_as_beta = 1;
#my $t0 = 0;
#my $t1 = 2000;
#my $dt = 0.5;
#my $euclidean_t = 0;

my $T0 = 0;
my $T1 = 0;
my $dT = 1;
my $treat_T_as_beta = 1;
my $t0 = 0;
my $t1 = 100;
my $dt = 2;
my $euclidean_t = 0;

die if $majorana && $real_J;

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
    for my $N (@N_values) {
        if ($majorana && exists $majorana_mem_gb->{$N}) {
            $slurm_mem_mb = $majorana_mem_gb->{$N} * 1024;
        }

        if ($majorana && exists $majorana_time->{$N}) {
            $slurm_time = $majorana_time->{$N};
        }

        for my $i ($min_run .. $max_run) {
            my $run_name = "N${N}-run${i}";

            if ($majorana) {
                $run_name = "maj-${run_name}";
            }

            if ($real_J) {
                $run_name = "real-${run_name}";
            }

            my $kitaev_params = prepare_kitaev_params($run_name, $N, $J);
            submit_job($run_name, $kitaev_prog, $kitaev_params);
        }
    }
}

sub prepare_kitaev_params {
    my ($run_name, $N, $J) = @_;

    my $kitaev_params = [
        "--run-name $run_name",
        "--N $N",
        "--J $J",
        "--data-dir " . KitaevUtils::data_dir(),
    ];

    if ($majorana) {
        push @$kitaev_params, "--majorana";
    }

    if ($real_J) {
        push @$kitaev_params, "--real-J";
    }

    if ($two_pt || $two_pt_with_fluctuations) {
        if ($two_pt) {
            push @$kitaev_params, "--2pt";
        }

        if ($two_pt_with_fluctuations) {
            push @$kitaev_params, "--2pt-with-fluctuations";
        }

        push @$kitaev_params, "--T0 $T0";
        push @$kitaev_params, "--T1 $T1";
        push @$kitaev_params, "--dT $dT";

        if ($treat_T_as_beta) {
            push @$kitaev_params, "--treat-T-as-beta";
        }

        push @$kitaev_params, "--t0 $t0";
        push @$kitaev_params, "--t1 $t1";
        push @$kitaev_params, "--dt $dt";

        if ($euclidean_t) {
            push @$kitaev_params, "--euclidean-t";
        }
    }

    return $kitaev_params;
}

sub submit_job {
    my ($full_run_name, $kitaev_prog, $kitaev_params) = @_;

    my $prog2;
    my $prog2_params;

    if ($post_process_partition_function) {
        $prog2 = $Z_prog;
        $prog2_params = [
            KitaevUtils::data_dir() . "/${full_run_name}-spectrum.tsv"
        ];
    }

    KitaevUtils::submit_job(
        $full_run_name,
        $dry_run, $run_locally,
        $kitaev_prog, $kitaev_params,
        $prog2, $prog2_params,
        $slurm_mem_mb, $slurm_time,
    );
}

sub num_needed_samples {
    my ($target_evs, $N) = @_;
    my $evs_per_sample;

    if ($majorana) {
        $evs_per_sample = 2 ** ($N/2);
    }
    else {
        $evs_per_sample = 2 ** $N;
    }

    return int($target_evs / $evs_per_sample);
}
