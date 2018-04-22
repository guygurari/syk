#!/usr/bin/perl -w

use strict;
use warnings;
use Cwd;
use FindBin qw($Script $Dir);
use Getopt::Long;

my $prog = "./compute-partition-function.pl";

my $max_files_per_job = 1;

my $help = 0;

# Don't submit jobs, only print the command lines
my $dry_run = 0;

# Run locally instead of submitting the jobs
my $run_locally = 0;

# The specific charge sector to focus on
my $Q;

sub usage {
        print "Usage: $Script [--help] [-n] [--local] [--max-files-per-job 1] [--Q Q] run1-spectrum.tsv run2-spectrum.tsv ...\n";
}

GetOptions(
        'help' => \$help,
        'n' => \$dry_run,
        'local' => \$run_locally,
        'max-files-per-job=s' => \$max_files_per_job,
        'Q=s' => \$Q,
        );

if ($help) {
    usage();
    exit 0;
}

my @files = @ARGV;

# Resolve data_dir to its full path, which is on /farmshare/user_data/...
# This is important because when the SSH session ends, eventually the
# jobs lose permission to write to the homedir. So we want all writes
# to go directory to the resolved data dir.
my $data_dir = Cwd::realpath('data');
die "Missing data dir '$data_dir'" unless -d $data_dir;

while (scalar(@files) > $max_files_per_job) {
    my @job_files = @files[0..$max_files_per_job-1];
    @files = @files[$max_files_per_job..scalar(@files)-1];
    submit_majorana_jobs(\@job_files);
}

if (scalar(@files) > 0) {
    submit_majorana_jobs(\@files);
}

sub submit_majorana_jobs {
    my ($files) = @_;
    my $run_name = "Z";
    my @params = @$files;

    if (defined $Q) {
        push @params, "--Q $Q";
    }

    submit_job($run_name, \@params);
}

sub submit_job {
    my ($full_run_name, $params) = @_;

    if ($run_locally) {
        my $cmd = "$prog " . join(' ', @$params);
        execute($cmd);
    }
    else {
        $ENV{PARAM_FULL_RUN_NAME} = $full_run_name;
        $ENV{PARAM_PROG} = $prog;
        $ENV{PARAM_PROG_PARAMS} = join(' ', @$params);
        my $cmd = "qsub -N ${full_run_name} -V -o $data_dir -e $data_dir jobs/kitaev-job";

        print "Params = \n$ENV{PARAM_PROG_PARAMS}\n";
        execute($cmd);
    }
}

sub execute {
    my ($cmd) = @_;
    print ("\n$cmd\n\n");

    if (!$dry_run) {
        system($cmd);
        die $@ if $? >> 8;
    }
}

