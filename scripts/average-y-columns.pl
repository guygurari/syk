#!/usr/bin/env perl
#
# Compute the average of the Y columns: all columns, labelled by the first 
# one by default. Number of 'key' columns (= X columns) can be specified.
# in the given files.
#
# Input files may be bzip2 compressed.
#

use FindBin qw($Script);
use IO::File;
use Getopt::Long;
#use List::MoreUtils 'pairwise';

use warnings;
use strict;

sub usage {
    print "Usage: $Script [--help] [--num-key-columns n] --output out-file in-files ...\n";
}

# Copied from List::MoreUtils
sub pairwise (&\@\@) {
    my $op = shift;

    # Symbols for caller's input arrays
    use vars qw{ @A @B };
    local ( *A, *B ) = @_;

    # Localise $a, $b
    my ( $caller_a, $caller_b ) = do {
        my $pkg = caller();
        no strict 'refs';
        \*{$pkg.'::a'}, \*{$pkg.'::b'};
    };

    # Loop iteration limit
    my $limit = $#A > $#B? $#A : $#B;

    # This map expression is also the return value
    local( *$caller_a, *$caller_b );
    map {
        # Assign to $a, $b as refs to caller's array elements
        ( *$caller_a, *$caller_b ) = \( $A[$_], $B[$_] );

        # Perform the transformation
        $op->();
    }  0 .. $limit;
}

my $num_keys = 1;
my $outfilename;
my $help = 0;

GetOptions(
    'help' => \$help,
    'num-key-columns=s' => \$num_keys,
    'output=s' => \$outfilename,
);

if ($help) {
    usage();
    exit 0;
}

if (!defined $outfilename) {
    usage();
    exit 1;
}

# X -> {
#       cols => [col2,col3,...,colN],
#       n    => num_rows
#      }
my $data = {};

my $header;

# Collect the data
foreach my $infilename (@ARGV) {
    my $infile;
   
    if ($infilename =~ /\.bz2$/) {
        $infile = IO::File->new("bunzip2 -c $infilename|") 
            || die "Can't open compressed file $infilename";
    }
    else {
        $infile = IO::File->new("< $infilename") 
            || die "Can't open $infilename";
    }

    my $line_num = 0;
    
    while (<$infile>) {
        chomp;

        if (/^#/) {
            $header = "$_\n";
            next;
        }

        my @cells = split /\t/, $_;

        my @key_cells = @cells[0..($num_keys-1)];
        my @Y_cells = @cells[$num_keys..(scalar(@cells)-1)];

        my $x = join("---", @key_cells);

        if (!exists $data->{$x}) {
            $data->{$x} = { 
                line => $line_num,
                keys => \@key_cells, 
                cols => \@Y_cells, 
                n => 1 };
        }
        else {
            my $cols = $data->{$x}->{cols};

            no warnings;
            my @sum = pairwise { $a + $b } @$cols, @Y_cells;
            use warnings;

            #if ($x == 1) {
            #    print "$x: current cols:\t" . join(",",@$cols) . "\n";
            #    print "$x: new cols:\t" . join(",",@cells) . "\n";
            #    print "$x: result:\t" . join(",",@sum) . "\n";
            #}

            $data->{$x}->{cols} = \@sum;
            $data->{$x}->{n} += 1;
        }

        $line_num++;
    }

    $infile->close();
}

my $outfile = IO::File->new("> $outfilename") || die;

if (defined $header) {
    $outfile->print($header);
}

foreach my $x (sort {$data->{$a}->{line} <=> $data->{$b}->{line}} (keys %$data)) {
    my $key_cols = $data->{$x}->{keys};
    my $cols = $data->{$x}->{cols};
    my $n = $data->{$x}->{n};

    $outfile->print(join("\t", @$key_cols));

    foreach my $y (@$cols) {
        my $avg_y = $y / $n;
        $outfile->print("\t$avg_y");
    }

    $outfile->print("\n");
}

$outfile->close();



