#!/usr/bin/env perl
#
# Concatenate the given TSV data files.
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

foreach my $infilename (@ARGV) {
    my $infile = IO::File->new("< $infilename") 
        || die "Can't open $infilename";

    while (<$infile>) {
        if (/^#/) {
            if (!$wrote_header) {
                $outfile->print($_);
                $wrote_header = 1;
            }
            next;
        }

        $outfile->print($_);
    }
}

$outfile->close();
