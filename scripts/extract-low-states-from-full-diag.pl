#!/usr/bin/env perl
#
# Extract E0 and E1 from existing Majorana runs, puts them into
# a single file. This is meant to be run on output of the usual
# full-diagonalization implementation for small N values.
# Those results are written to data/maj-NXXX-runXXX-spectrum.tsv, 
# we read them directly from there, and then they should be deleted.
#

use strict;
use warnings;
use IO::File;

my $N = 10;
my $out_fname = "data/gndlanc-N$N.tsv";
my $out;

if (-f $out_fname) {
    $out = IO::File->new(">> $out_fname");
}
else {
    $out = IO::File->new("> $out_fname");
    $out->print("# run\tE0\tE1\n");
}

sub read_energy {
    local $_;
    my $in = shift;
    $_ = <$in>;
    chomp;
    my @elems = split /\t/;
    return $elems[1];
}

sub read_first_two_energies {
    local $_;
    my $fname = shift;
    my $in = IO::File->new("< $fname") || die "Can't open $fname";
    <$in>;
    my $E0 = read_energy($in);
    my $E1 = read_energy($in);
    $in->close();
    return ($E0, $E1);
}

#foreach my $f (glob("data/*.deleteme")) {
#    my $f2 = $f;
#    $f2 =~ s/\.deleteme//;
#    rename $f, $f2;
#}

my @files = glob("data/maj-N$N-run*-spectrum.tsv");
my %runs = map { /run(\d+)/; $1 => $_ } @files;

foreach my $run (sort {$a <=> $b} keys(%runs)) {
    my $fname = $runs{$run};
    print "Processing $run: $fname ...\n";
    $fname =~ /run(\d+)/ || die;
    my $run = $1;
    my ($E0, $E1) = read_first_two_energies($fname);
    $out->print("$run\t$E0\t$E1\n");
    rename $fname, "$fname.deleteme";
}

$out->close();
print "\nDone. Now run this: rm *.deleteme\n\n";
