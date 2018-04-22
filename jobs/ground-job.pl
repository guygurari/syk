#!/usr/bin/perl
#

use strict;
use warnings;

my $dry_run = 0;

die unless exists $ENV{MIN_RUN};
die unless exists $ENV{MAX_RUN};

my $N = $ENV{SYK_N};
my $min_run = $ENV{MIN_RUN};
my $max_run = $ENV{MAX_RUN};
my $data_dir = $ENV{DATA_DIR};
my $checkpt_dir = $ENV{CHECKPT_DIR};
die "Can't find data dir $data_dir" unless -d $data_dir;

my $steps = 300;

foreach my $run ($min_run .. $max_run) {
    my $run_name = sprintf("gndlanc-N${N}-run%05d", $run);
    my $cmd = "./syk-gpu-lanczos --run-name $run_name --N $N --J 1 --data-dir $data_dir --checkpoint-dir $checkpt_dir --num-steps $steps --checkpoint-steps $steps --ev-steps $steps";
    print "$cmd\n\n";

    if (!$dry_run) {
        system($cmd);
        die "Error running command" if $? >> 8;
    }
}

