#!/usr/bin/env perl
#
# Compute time-dependent partition function for the given files.
#

use FindBin qw($Script $Dir);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

my @betas = (0,1,5,10,20,30);

my $no_Q_column = 0;
my $dry_run = 0;
my $help = 0;
my $betas_s;
my $Q;
my $t_start;
my $t_end;
my $t_step;
my $log_t_step;
my $dont_compress = 0;

my $Z_prog = "$Dir/partition-function-single-sample";

if (! -x $Z_prog) {
    die "Can't find partition function executable: $Z_prog";
}

sub usage {
    print "Usage: $Script [--help] [-n] [--no-Q-column] [--betas 0,1,5,10] [--t-start t0] [--t-end t1] [--t-step ts] [--log-t-step] [--Q Q] [--dont-compress] run1-spectrum.tsv run2-spectrum.tsv ...\n";
}

GetOptions(
    'help' => \$help,
    'n' => \$dry_run,
    'no-Q-column' => \$no_Q_column,
    'betas=s' => \$betas_s,
    'Q=s' => \$Q,
    't-start=s' => \$t_start,
    't-end=s' => \$t_end,
    't-step=s' => \$t_step,
    'log-t-step' => \$log_t_step,
    'dont-compress' => \$dont_compress,
);

if ($help) {
    usage();
    exit 0;
}

if (scalar(@ARGV) == 0) {
    usage();
    exit 1;
}

if (defined $betas_s) {
    @betas = split /,/, $betas_s;
}

#print "betas = " . join(" ", @betas) . "\n";

my @files = @ARGV;

foreach my $spectrum_file (@files) {
    my $Z_file = $spectrum_file;
    $Z_file =~ s/-spectrum/-Z/ || die "File must end with -spectrum";

    my $cmd = "$Z_prog ".
        "--spectrum-file $spectrum_file ".
        "--betas " . join(',', @betas)
        ;

    if ($no_Q_column) {
        $cmd .= " --no-Q-column";
    }

    if (defined $t_start) {
        $cmd .= " --t-start $t_start";
    }

    if (defined $t_end) {
        $cmd .= " --t-end $t_end";
    }

    if (defined $t_step) {
        $cmd .= " --t-step $t_step";
    }

    if (defined $log_t_step) {
        $cmd .= " --log-t-step";
    }

    # Always compute Z over all charge sectors
    if (!output_file_up_to_date($spectrum_file, "$Z_file.bz2")) {
        print "\nComputing Z: All sectors...\n\n";
        my $cmd_all_sectors = "$cmd --output-file $Z_file";
        execute($cmd_all_sectors);
        compress($Z_file) unless $dont_compress;
    }

    if (defined $Q && !$no_Q_column) {
        # For Majorana, also compute Z in the even charge parity sector.
        # For Dirac, also compute Z in the Q=N/2 sector.
        my $majorana = ($spectrum_file =~ /\bmaj-/);
        $Z_file =~ /\bN(\d+)\b/ || die "Can't find N in filename $Z_file";
        my $N = $1;

        my $sector_Z_file = $Z_file;
        $sector_Z_file =~ s/\bZ\b/ZQ${Q}/;

        if (!output_file_up_to_date($spectrum_file, "$sector_Z_file.bz2")) {
            print "\nComputing Z: Q=$Q sector...\n\n";
            my $cmd_sector = "$cmd --output-file $sector_Z_file --Q $Q";
            execute($cmd_sector);
            compress($sector_Z_file) unless $dont_compress;
        }
    }
}

# Check that output file exists and is newer than input file
sub output_file_up_to_date {
    my ($in, $out) = @_;
    return (-f $out) && (-M $out < -M $in);
}

sub execute {
    my $cmd = shift;
    print "$cmd\n";

    if (!$dry_run) {
        system($cmd);
        die "Command failed" if $? >> 8;
    }
}

sub compress {
    my $file = shift;
    print "Compressing $file\n";
    execute("bzip2 -f $file");
}

