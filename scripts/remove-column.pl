#!/usr/bin/env perl
#
# Remove the given column name from the tab-separated files.
# There must be a header line starting with '#' that includes that column.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;

use warnings;
use strict;

sub usage {
    print "Usage: $Script [--help]  --column name  files ...\n";
}

my $help = 0;
my $col_name;

GetOptions(
    'help' => \$help,
    'column=s' => \$col_name,
);

if ($help) {
    usage();
    exit 0;
}

if (!defined $col_name) {
    die "Must specify --column";
}

my @file_names = @ARGV;

foreach my $fname (@file_names) {
    my $output_fname = "$fname.remove-column";

    my $in;
    my $out;

    if (is_compressed($fname)) {
        $in = IO::File->new("bzcat $fname|") || die;
    }
    else {
        $in = IO::File->new("< $fname") || die;
    }

    my $header = <$in>;
    chomp $header;
    die "No header line in $fname" unless $header =~ /^#/;
    my @header_cols = split /\t/, $header;

    my ($col_to_remove) = grep { $header_cols[$_] eq $col_name } 0..$#header_cols;

    if (!defined $col_to_remove) {
        print "Warning: Can't find column '$col_name' in file $fname\n";
        $in->close();
        next;
    }

    $out = IO::File->new("> $output_fname") || die;

    splice @header_cols, $col_to_remove, 1;
    my $new_header_line = join("\t", @header_cols) . "\n";
    $out->print($new_header_line);

    # The remaining lines don't include the initial column '#'
    $col_to_remove -= 1;
    #print "Column: $col_to_remove\n";

    while (<$in>) {
        chomp; 
        my @columns = split /\s+/;
        splice @columns, $col_to_remove, 1;
        my $new_line = join("\t", @columns) . "\n";
        #print $new_line;
        $out->print($new_line);
    }

    $in->close();
    $out->close();

    if (is_compressed($fname)) {
        unlink $fname || die "Cannot delete original file $fname";
        system("bzip2 -c $output_fname > $fname");
        die "Cannot compress $output_fname" if $? >> 8;
        unlink $output_fname;
    }
    else {
        system("cp $output_fname $fname");
        die "Can't cp $output_fname -> $fname" if $? >> 8;
        unlink $output_fname;
    }
}

sub is_compressed {
    my $fname = shift;
    return $fname =~ /\.bz2$/;
}

