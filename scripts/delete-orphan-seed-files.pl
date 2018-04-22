#!/usr/bin/env perl
#
# Delete lanczos seed files that don't have a corresponding state file.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
	print "Usage: $Script [--help] [-n]\n";
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

if (scalar(@ARGV) != 0) {
	usage();
	exit 1;
}

foreach my $seed_file (glob("lanc-*-seed")) {
    my $state_file = $seed_file;
    $state_file =~ s/-seed/-state/;
    
    if (!-f $state_file) {
        print "rm $seed_file\n";

        if (!$dry_run) {
            unlink $seed_file || die "Can't delete $seed_file";
        }
    }
}
