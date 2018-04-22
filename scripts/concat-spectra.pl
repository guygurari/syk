#!/usr/bin/env perl
#
# Concatenate the spectra in the given TSV files.
# Add a first column which is the run number.
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

my $outfile = IO::File->new("> $outfilename") || die;
my $wrote_header = 0;

my $infiles = {};

foreach my $infilename (@ARGV) {
    $infilename =~ /-run(\d+)-/ || die die "Can't find run number in $infilename";
    my $run = $1;
    $infiles->{$run} = $infilename;
}

foreach my $run (sort {$a <=> $b} keys %$infiles) {
    my $infilename = $infiles->{$run};
    my $infile = IO::File->new("< $infilename") 
        || die "Can't open $infilename";

    while (<$infile>) {
        if (/^#/) {
            if (!$wrote_header) {
                s/# /# run\t/;
                $outfile->print($_);
                $wrote_header = 1;
            }
            next;
        }

        $outfile->print("$run\t$_");
    }
}

$outfile->close();
