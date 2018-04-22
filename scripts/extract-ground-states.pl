#!/usr/bin/perl
#
# Print the ground state and first excited state from each file.

use strict;
use warnings;
use IO::File;
use Getopt::Long;
use FindBin qw($Script);

my $checkpoints_dir = "data/lanczos-checkpoints";

sub usage {
    print "Usage: $Script [--help] gndlanc-N32-run1.tsv gndlanc-N32-run2.tsv ...\n\n";
}

my $help = 0;

GetOptions('help' => \$help);

if ($help) {
    usage();
    exit 0;
}

my @files = @ARGV;

sub read_state {
    local $_;
    my ($f) = @_;
    $_ = <$f> || die "Can't read from file";
    chomp;
    s/^\d+\s+//;
    s/\s+.*//;
    return $_;
}

sub compute_H_moment {
    my $spectrum_file = shift;

    $spectrum_file =~ /([^\/]+)-N(\d+)-run(\d+)/ || die;
    my ($experiment, $N, $run) = ($1, $2, $3);

    my $seed_file = "$checkpoints_dir/${experiment}-N${N}-run${run}-seed";
    die "Can't find seed file $seed_file" unless -f $seed_file;

    my $TrH2 = `./compute_H_moment --N $N --seed-file $seed_file` || die;
    chomp $TrH2;
    return $TrH2;
}

print "# run\tE0\tE1\tSum(Jijkl^2)\n";

# Get the lowest energy state
for my $fname (@files) {
    $fname =~ /-run(\d+)/ || die;
    my $run = $1;

    my $low = IO::File->new("head -n 3 -q $fname | grep -v \\# |");
    die "Can't read head from $fname" unless defined $low;
    my $E0 = read_state($low);
    my $E1 = read_state($low);

    my $TrH2 = compute_H_moment($fname);

    print "$run\t$E0\t$E1\t$TrH2\n";
    $low->close();
}

# Get the highest energy state and add a minus
# for my $fname (@files) {
#     my $high = IO::File->new("tail -n 1 -q $fname |");
#     die "Can't read tail" unless defined $high;
#     my $E0 = read_state($high);
#     my $E1 = read_state($high);
#     print "$E0\n";
#     $high->close();
# }

