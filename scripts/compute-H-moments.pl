#!/usr/bin/env perl
#
# Compute Tr(H^2) from the J couplings
#

use IO::File;

my $data_dir = 'data';
my $checkpoint_dir = 'data/lanczos-checkpoints';

foreach my $N (24, 26, 28, 30, 32, 34, 36) {
    my $experiment = "gndlanc-N$N";
    my $out_filename = "$data_dir/$experiment/${experiment}-H-moments.tsv";
    print "Processing $experiment into $out_filename\n";

    my $out = IO::File->new("> $out_filename") || die;
    $out->print("# sample\tTr(H^2)\n");

    foreach my $seed_file (glob("$checkpoint_dir/${experiment}-run*-seed")) {
        $seed_file =~ /-N(\d+)-run(\d+)-/ || die;
        my $N = $1;
        my $run = $2;
        my $TrH2 = `./compute_H_moment --N $N --seed-file $seed_file` || die;
        chomp $TrH2;
        $out->print("$run\t$TrH2\n");
    }

    $out->close();
}


