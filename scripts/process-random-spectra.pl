#!/usr/bin/env perl
#
# Process the spectra of random matrices in the current directory:
# - Collect combined spectrum
#

use FindBin qw($Script $Dir);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub verify_executable {
    my $executable = shift @_;

    if (!-x $executable) {
        die "Can't find executable $executable";
    }

    return $executable;
}

my $concat_script = verify_executable("$Dir/concat-spectra.pl");
my $combine_logs_script = verify_executable("$Dir/combine-logs.pl");
my $avg_y_script = verify_executable("$Dir/average-y-columns.pl");
my $Z_disorder_script = verify_executable("$Dir/partition-function-disorder-average");

sub usage {
    print "Usage: $Script [--help] [-n] [directory]\n";
}

my $help = 0;
my $dry_run = 0;

GetOptions(
    'help' => \$help,
    'n' => \$dry_run,
);

if ($help) {
    usage();
    exit 0;
}

if (scalar(@ARGV) == 1) {
    my $dir = $ARGV[0];
    die "No such directory: $dir" unless -d $dir;
    chdir $dir;
}
elsif (scalar(@ARGV) > 1) {
    usage();
    exit 1;
}

my @spectrum_files = glob("*-run*-spectrum.tsv");
die "Can't find spectrum files" if scalar(@spectrum_files) == 0;

my $spectrum_file = $spectrum_files[0];
$spectrum_file =~ s/-run\d+//;

my $Z_file = $spectrum_file;
$Z_file =~ s/-spectrum/-Z/;

my $cmd;

# Logs
print "\nCombining logs ...\n";
$cmd = "$combine_logs_script";
execute($cmd);

# Total spectrum
print "\nComputing total spectrum ...\n";
$cmd = "$concat_script --output $spectrum_file *-run*-spectrum.tsv";
execute($cmd);

# Partition function
print "\nComputing partition function disorder average ...\n";
$cmd = "$Z_disorder_script --output-file $Z_file *-run*-Z.tsv.bz2";
execute($cmd);

print "\nDone.\n";

exit 0;

sub execute {
    my $cmd = shift @_;
    print "$cmd\n";

    if (!$dry_run) {
        system($cmd);
        die "Command failed" if $? >> 8;
    }
}
