#!/usr/bin/env python3
#
# Find good Lanczos eigenvalues, and separate high and low.
#

import os
import sys
import re
import numpy as np
import pandas as pd
import optparse

input_dir = 'data/lanczos'
processed_dir = 'data/lanczos-processed'
acceptable_err = 0.01
N_power = 4
# min_num_evs = 1000
# tolerance = 1e-8

def get_data_file(name):
    return "%s/%s" % (input_dir, name)

def is_bad_ev(ev):
    return ev['err'] > tolerance

def get_good_evs(evs_and_errs, min_evs, max_evs):
    """Extract good ('low error') eigenvalues based on heuristics.
These were obtained by analyzing the Lanczos errors empirically."""
    evs = evs_and_errs['ev'].values
    errs = evs_and_errs['err'].values

    diffs = evs[1:] - evs[:-1]

    # Only look at the bottom half of the spectrum
    evs = evs[:max_evs]
    errs = errs[:max_evs]
    diffs = diffs[:max_evs]

    assert len(evs) == len(diffs)

    # The worst error we will tolerate is a fraction of
    # the difference between the neighboring eigenvalues,
    # because we really want the level spacings to be
    # precise. The cutoff is this worst allowed error.
    cutoff = diffs * acceptable_err

    # Indicator function of which eigenvalues are 'bad',
    # namely have error greater than the cutoff.
    bad_ev_bool = errs > cutoff
    bad_ev_indicator = [1 if x else 0 for x in bad_ev_bool]

    # How many bad ev's did we see so far?
    bad_ev_cumsum = np.cumsum(bad_ev_indicator)

    # How many evs did we see so far in total?
    #N = evs_and_errs['idx'].values + 1
    N = evs_and_errs.index.values + 1
    N = N[:len(diffs)]

    # We will choose how many eigenvalues to use, starting at the first one,
    # by minimizing the following metric. It tries to balance minimizing the
    # number of bad evs, while maximizing the number of total evs, giving the
    # latter a greater weight.
    #
    # In practice, if we follow eigenvalues from the lowest one we see the errors
    # are very low, with occassional spikes, and then at some point they start
    # rising dramatically. We want to tolerate a few spikes, but stop using eigenvalues
    # once the error starts rising. This choice of metric seems to do this well.
    #
    # Here we add 1 to avoid division-by-zero, and to make N always meaningful in the metric
    metric = (bad_ev_cumsum + 1) / np.power(N, N_power)
    min_i = np.argmin(metric)

    if min_i == max_evs - 1:
        print("Warning: Found exactly max_evs=%d good eigenvalues, either bad sample of max_evs too low." % max_evs)
        #sys.exit(1)
        return None

    if min_i < min_evs - 1:
        print("Warning: Found less than min_evs=%d good eigenvalues, probably metric should be adjusted (increase N_power)." % min_evs)
        #sys.exit(1)
        return None
        
    return evs_and_errs[:min_i+1]


def save_good_evs(evs, filename, suffix):
    m = re.search('%s/(\w+-N\d+)/(\w+-N(\d+)-run\d+)\.tsv' % input_dir, filename)

    if m == None:
        print('Filename %s does not match regexp during save_good_evs' % filename)
        sys.exit(1)

    experiment = m.group(1)
    basename = m.group(2)
    N = m.group(3)
    output_dir = '%s/%s' % (processed_dir, experiment)
    out = '%s/%s-%s.tsv' % (output_dir , basename, suffix)

    try:
        os.mkdir(output_dir)
    except OSError:
        # directory already exists
        pass

    evs.to_csv(
        out,
        sep='\t',
        index=True,
        header=['ev'],
        index_label='# i'
    )

def process_evs_file(filename, options):
    evs_and_errs = pd.read_table(
        filepath_or_buffer = filename,
        sep = '\t',
        comment = '#',
        names = ['idx', 'ev', 'err'],
        )

    low_evs = get_good_evs(evs_and_errs, options.min_evs, options.max_evs)

    if low_evs is not None:
        print("Found %d good low evs" % len(low_evs))
        save_good_evs(low_evs['ev'], filename, "low-spectrum")

    flipped = evs_and_errs.sort_index(ascending=False).reset_index(drop=True)
    flipped['ev'] = flipped['ev'].apply(lambda x: -x)
    high_evs = get_good_evs(flipped, options.min_evs, options.max_evs)

    if high_evs is not None:
        print("Found %d good high evs" % len(high_evs))
        save_good_evs(high_evs['ev'], filename, "high-spectrum")

def main():
    parser = optparse.OptionParser()
    parser.add_option('--min-evs', dest='min_evs', default=0, type=int,
                      help='Minimum number of evs to consider')
    parser.add_option('--max-evs', dest='max_evs', default=0, type=int,
                      help='Maximum number of evs to consider')

    (options, args) = parser.parse_args()

    if options.min_evs == 0:
        print('Error: Must provide --min-evs')
        sys.exit(1)

    if options.max_evs == 0:
        print('Error: Must provide --max-evs')
        sys.exit(1)
    
    for filename in args:
        print("Processing %s" % filename)
        process_evs_file(filename, options)
    
if __name__ == '__main__':
    main()
