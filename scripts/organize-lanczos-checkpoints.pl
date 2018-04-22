#!/usr/bin/env perl

use strict;
use warnings;

my $experiments = {};

foreach my $state (glob("*-N*-run*-state")) {
    $state =~ /.*-N\d+/ || die;
    my $experiment_dir = $&;

    if (! exists $experiments->{$experiment_dir} && ! -d $experiment_dir) {
        print "mkdir $experiment_dir\n";
        mkdir $experiment_dir || die;
        $experiments->{$experiment_dir} = 1;
    }

    my $seed = $state;
    $seed =~ s/-state/-seed/;
    die unless -f $seed;

    if (!-d $experiment_dir) {
        die;
    }

    print "rename $state, $experiment_dir\n";
    print "rename $seed, $experiment_dir\n";

    rename $state, $experiment_dir || die;
    rename $seed, $experiment_dir || die;
}

#sub execute {
#    my $cmd = shift;
#    print "Executing: $cmd\n";
#    system($cmd);
#    die "Failed executing $cmd" if $? >> 8;
#}

