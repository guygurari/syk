#!/usr/bin/env perl
#
# Compute thermodynamic quantities for the given files.
#

use FindBin qw($Script $Dir);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

my $majorana = 0;
my $T_start;
my $T_end;
my $T_step;

my $dry_run = 0;
my $help = 0;

my $thermo_prog = "$Dir/kitaev-thermodynamics";

if (! -x $thermo_prog) {
    die "Can't find thermodynamics executable: $thermo_prog";
}

sub usage {
    print "Usage: $Script [--help] [-n] --T-start start ";
    print "--T-end end --T-step step [--majorana]\n";
}

GetOptions(
    'help' => \$help,
    'n' => \$dry_run,
    'majorana' => \$majorana,
    'T-start=s' => \$T_start,
    'T-end=s' => \$T_end,
    'T-step=s' => \$T_step,
);

if ($help) {
    usage();
    exit 0;
}

if (!defined $T_start || !defined $T_end || !defined $T_step) {
    usage();
    exit 1;
}

if (scalar(@ARGV) == 0) {
    usage();
    exit 1;
}

foreach my $spectrum_file (@ARGV) {
    my $thermo_file = $spectrum_file;
    $thermo_file =~ s/-spectrum/-thermo/ || die;

#    if (-f $thermo_file) {
#        print "Skipping $spectrum_file (thermo file already exists)\n";
#        next;
#    }

    my $cmd = "$thermo_prog --T-start $T_start --T-end $T_end ".
        "--T-step $T_step --spectrum-file $spectrum_file ".
        "--output-file $thermo_file";
        #print "$cmd\n";

    if ($majorana) {
        $cmd .= " --majorana";
    }

    if (!$dry_run) {
        system($cmd);
        die "Cannot process $spectrum_file" if $? >> 8;
    }
}



