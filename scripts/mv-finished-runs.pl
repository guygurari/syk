#!/usr/bin/perl
#
# Move finished spectrum runs into a directory based on the prefix.
#

use strict;
use warnings;

foreach my $err (glob("*.e*")) {
    if (-s $err) {
        # stderr file has non-zero size, which means the run completed
        $err =~ /^((.*)-run\d+)/ || die "Bad filename: $err";

        my $base_name = $1;
        my $dir = $2;

        if (! -d $dir) {
            mkdir $dir || die "Can't create directory $dir";
        }

        
        my @products = glob("$base_name-*");

        if (scalar(@products) > 0) {
            execute("mv $base_name-* $base_name.* $dir");
        }
        else {
            print "No products found for run $base_name, deleting logs:\n";
            execute("rm $base_name.*");
        }
    }
    else {
        print "Not finished: $err\n";
    }
}

sub execute {
    my $cmd = shift;
    print "$cmd\n";
    system($cmd);
    die if $? >> 8;
}

