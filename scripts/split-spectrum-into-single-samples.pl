#!/usr/bin/env perl
#
# Split the given spectrum file that contains the spectra of multiple samples
# into single-sample files.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
    print "Usage: $Script [--help] multi-sample-spectrum.tsv\n";
}

my $help = 0;

GetOptions(
    'help' => \$help,
);

if ($help) {
    usage();
    exit 0;
}

if (scalar(@ARGV) != 1) {
    usage();
    exit 1;
}

my $input_filename = $ARGV[0];

$input_filename =~ /-spectrum\.tsv/ || die "Bad input filename: $input_filename";

my $f = IO::File->new("< $input_filename") 
    || die "Can't open input file $input_filename";
my $last_sample;
my $out;

my $header = <$f>;

while (<$f>) {
    chomp;
    my @cols = split /\s+/, $_;
    my $sample = shift @cols;

    if (!defined $last_sample  ||  $sample != $last_sample) {
        my $output_filename = $input_filename;
        $output_filename =~ s/-spectrum\.tsv/-run${sample}-spectrum.tsv/ || die;

        if (defined $out) {
            $out->close();
        }

        $out = IO::File->new("> $output_filename") || die;
        $out->print($header);
        $last_sample = $sample;
    }

    $out->print(join("\t", @cols) . "\n");
}

if (defined $out) {
    $out->close();
}

$f->close();



