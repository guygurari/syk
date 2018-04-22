#!/usr/bin/perl
#
# Combine the log files xxx.log xxx.o123 xxx.e123 into single xxx.log files.
# 

use warnings;
use strict;

foreach my $log_file (glob("*.log")) {
#print "Combining into $log_file\n";
    combine_into_log($log_file, "o*");
    combine_into_log($log_file, "e*");
}

sub combine_into_log {
    my ($log_file, $suffix) = @_;
    my $base = $log_file;
    $base =~ s/\.log$// || die;

    my @ofiles = glob("$base.$suffix");

    if (scalar(@ofiles) > 1) {
        die "More than one alternative file for $log_file : " . join(' ', @ofiles);
    }
    elsif (scalar(@ofiles) == 1) {
        my $ofile = shift @ofiles;
        my $cmd = "cat $ofile >> $log_file";
#print "$cmd\n";
        system($cmd);
        die "Failed" if $? >> 8;
        unlink $ofile || die "Can't erase $ofile";
    }
}

