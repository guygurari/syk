#!/usr/bin/perl

use warnings;
use strict;

my $wanted_Q = 1;
my @files;

foreach (@ARGV) {
    my $f = IO::File->new("< $_") || die "Can't open $_";
    push @files, $f;
}

while (1) {
    my $s = '';
    
    foreach my $f (@files) {
        if ($_ = <$f>) {
            next if /^#/;
            chomp;
            my ($Q, $ev) = split /\t/;

            if ($Q == $wanted_Q) {
                $s .= "$ev\t";
            }
        }
        else {
            exit 0;
        }
    }

    if (length($s) > 0) {
        $s =~ s/\t$//;
        print "$s\n";
    }
}
