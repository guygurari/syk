#!/usr/bin/perl

use strict;
use warnings;
use IO::File;
use KitaevUtils;

my $dry = 0;

my $col = 1;
my $threshold = 100000;
my $betas = '0,1,5,10,20,30';

my $num = 0;
my @files = @ARGV;

foreach my $fname (@files) {
    my $last_line = `bunzip2 -c $fname | tail -1`;
    chomp $last_line;

    my @elems = split /\t/, $last_line;
    
    if ($elems[$col] < $threshold) {
        print "Redoing Z for $fname\n";
        my $spectrum_file = $fname;
        $spectrum_file =~ s/-Z.*/-spectrum.tsv/;
        die unless -f $spectrum_file;

        my $Q = '';
        if ($fname =~ /Q(\d)/) {
            $Q = "--Q $1";
        }

        my $output_fname = $fname;
        $output_fname =~ /\.bz2//;

        my $cmd = "~/k/partition-function-single-sample --spectrum-file $spectrum_file --output-file $output_fname $Q --betas $betas";
        KitaevUtils::execute($cmd, $dry);
        KitaevUtils::execute("bzip2 $output_fname", $dry);
        print "\n";

        $num++;
    }
}

print "Number below threshold: $num\n";

