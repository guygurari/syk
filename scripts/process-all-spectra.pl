#!/usr/bin/perl

use strict;
use warnings;

use FindBin qw($Script $Dir);

my $data_dir = "$Dir/data";
my $process_script = "$Dir/process-spectra.pl";
die "Can't find data directory $data_dir" unless -d $data_dir;
die "Can't find processing script $process_script" unless -x $process_script;

foreach my $dir (glob("$data_dir/*")) {
    if (-d $dir) {
        my $cmd = "$process_script $dir";
#print "$cmd\n";
        system($cmd);
        die "Failed running $cmd" if $? >> 8;
    }
}

