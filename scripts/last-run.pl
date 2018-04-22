#!/usr/bin/perl
#
# Show the last run number in the current directory.
#

use strict;
use warnings;

my $last_run;

foreach my $file (glob("*-run*-*")) {
    $file =~ /-run(\d+)-/ || die "Bad filename $file";
    my $run = $1;

    if (defined $last_run && $run > $last_run) {
        $last_run = $run;
    }
    elsif (!defined $last_run) {
        $last_run = $run;
    }
}

print "$last_run\n";

