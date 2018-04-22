#!/usr/bin/env perl
#
# Truncate the number of samples in the given spectrum file.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
    print "Usage: $Script [--help] [--max-samples n] spectrum.tsv output.tsv\n";
}

my $help = 0;
my $max_samples = 0;

GetOptions(
    'help' => \$help,
    'max-samples=s' => \$max_samples,
);

if ($help) {
    usage();
    exit 0;
}

if (scalar(@ARGV) != 2) {
    usage();
    exit 1;
}

my $in_fname = $ARGV[0];
my $out_fname = $ARGV[1];

my $in = IO::File->new("< $in_fname") || die "Can't open $in_fname";
my $out = IO::File->new("> $out_fname") || die;

my $last_sample = -1;
my $samples = 0;

while (<>) {
    if (/^#/) {
        $out->print($_);
        next;
    }

    my @cols = split /\s+/, $_;
    my $sample = $cols[0];

    if ($sample != $last_sample) {
        $last_sample = $sample;
        $samples++;

        if ($samples > $max_samples) {
            last;
        }
    }

    $out->print($_);
}

$in->close();
$out->close();

