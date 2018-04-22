#!/usr/bin/env perl
#
# Process the spectra in the current directory: 
# - Compute thermodynamics for each run
# - Compute disorder-average of thermodynamics
# - Collect combined spectrum
#

use FindBin qw($Script $Dir);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

my $T_start = 0.01;
my $T_end = 2.;
my $T_step = 0.01;

sub verify_executable {
    my $executable = shift @_;

    if (!-x $executable) {
        die "Can't find executable $executable";
    }

    return $executable;
}

my $thermo_script = verify_executable("$Dir/compute-thermodynamics.pl");
my $avg_y_script = verify_executable("$Dir/average-y-columns.pl");
my $concat_script = verify_executable("$Dir/concat-spectra.pl");
my $Z_disorder_script = verify_executable("$Dir/partition-function-disorder-average");
my $combine_logs_script = verify_executable("$Dir/combine-logs.pl");

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

# Find out some filenames
my $one_spectrum_file = get_one_file("*-run*-spectrum.tsv");
my $spectrum_file = $one_spectrum_file;
$spectrum_file =~ s/-run\d+//;

my $thermo_file = $spectrum_file;
$thermo_file =~ s/-spectrum/-thermo/;

my $Z_file = $spectrum_file;
$Z_file =~ s/-spectrum/-Z/;

my $twopt_file = $spectrum_file;
$twopt_file =~ s/-spectrum/-2pt/;

my $majorana = ($one_spectrum_file =~ /maj-/) ? 1 : 0;

if ($majorana) {
    print "Found Majorana files\n";
}

my $cmd;

# Logs
print "\nCombining logs ...\n";
execute($combine_logs_script);

# Total spectrum
print "\nComputing total spectrum ...\n";
execute("$concat_script --output $spectrum_file *-run*-spectrum.tsv");

# Parition function
print "\nComputing partition function disorder average ...\n";
$cmd = "$Z_disorder_script --output-file $Z_file *-run*-Z.tsv.bz2";
execute($cmd);

my $one_ZQ_file = get_one_file("*-run*-ZQ*.tsv.bz2");
$one_ZQ_file =~ /ZQ(\d+)/ || die "Can't find Q value in $one_ZQ_file";
my $Q = $1;
my $ZQ_output_file = $Z_file;
$ZQ_output_file =~ s/\bZ\b/ZQ${Q}/;
$cmd = "$Z_disorder_script --output-file $ZQ_output_file *-run*-ZQ*.tsv.bz2";
execute($cmd);

# Thermodynamics
print "\nComputing thermodynamics ...\n";
$cmd = "$thermo_script";

if ($majorana) {
    $cmd .= " --majorana";
}

$cmd .= " --T-start $T_start";
$cmd .= " --T-end $T_end";
$cmd .= " --T-step $T_step";
$cmd .= " *-run*-spectrum.tsv";
execute($cmd);

print "\nComputing thermodynamics disorder average ...\n";
$cmd = "$avg_y_script --output $thermo_file *-run*-thermo.tsv";
execute($cmd);

if (! $majorana) {
    my @twopt_files = glob("*-2pt.tsv");
    if (scalar(@twopt_files) > 0) {
        print "\nComputing 2-point function disorder average ...\n";
        $cmd = "$avg_y_script --num-key-columns 2 --output $twopt_file *-run*-2pt.tsv";
        execute($cmd);
    }
}

print "\nDone.\n";

exit 0;

sub get_one_file {
    my $pattern = shift;
    my @files = glob($pattern);
    die "Can't find files that match '$pattern'" if scalar(@files) == 0;
    return $files[0];
}

sub execute {
    my $cmd = shift @_;
    print "$cmd\n";

    if (!$dry_run) {
        system($cmd);
        die "Command failed" if $? >> 8;
    }
}
