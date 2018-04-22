#!/usr/bin/env perl
#
# Concatenate the Lanczos spectra in the given TSV files.
# Includes both high and low parts of the spectrum.
# Add a first column which encodes the run number + whether it's the
# high or low part.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
    print "Usage: $Script [--help] --output out-file in-files ...\n";
}

my $help = 0;
my $outfilename;

GetOptions(
    'help' => \$help,
    'output=s' => \$outfilename,
);

if ($help) {
    usage();
    exit 0;
}

if (!defined $outfilename) {
    usage();
    exit 1;
}

my $outfile = IO::File->new("> $outfilename") || die "Failed to open $outfilename for writing";
my $wrote_header = 0;

my $infiles = {};

foreach my $infilename (@ARGV) {
    $infilename =~ /-run(\d+)-(high|low)-/ || die die "Can't find run details in $infilename";
    my $run = $1;
    my $type = ($2 eq 'low' ? 0 : 1);
    $run = 2 * ($run-1) + 1 + $type;
    $infiles->{$run} = $infilename;
}

foreach my $run (sort {$a <=> $b} keys %$infiles) {
    my $infilename = $infiles->{$run};
    my $infile = IO::File->new("< $infilename") 
        || die "Can't open $infilename";

    while (<$infile>) {
        if (/^#/) {
            if (!$wrote_header) {
                s/# /# run\tev\n/;
                $wrote_header = 1;
            }
            next;
        }

        chomp;
        my ($i, $ev) = split /\t/, $_;
        $outfile->print("$run\t$ev\n");
    }
}

$outfile->close();
