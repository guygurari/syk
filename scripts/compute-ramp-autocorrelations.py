#!/usr/bin/env python
#
# To use this in ipython:
#
# > ipython --profile=syk
# > run compute-ramp-autocorrelations.py
#

import sys
import glob
import math
import cmath
import numpy as np
import pandas as pd
import plotutils

# float accuracy
eps = 1e-5

def read_spectrum(filename, max_samples=None):
    """Read the multiple-sample spectrum and group by sample"""
    data = pd.read_table(
        filepath_or_buffer = filename,
        sep = '\t',
        comment = '#',
        names = ['sample', 'charge-parity', 'energy'],
        #usecols = ['sample', 'energy'],
    )

    if max_samples != None:
        data = data[data['sample'] <= max_samples]

    #return data.groupby('sample')
    return data

def compute_Z_by_sample(spectrum, beta, t):
    """Compute a DataFrame of Z(beta,t) indexed by sample."""
    exp_spectrum = pd.DataFrame({
        'sample' : spectrum['sample'],
        # This is much faster than using map(lambda)
        'Z' : np.exp(spectrum['energy'] * complex(-beta, t)),
        #'exp' : spectrum['energy'].map(
        #    lambda E: np.exp(complex(-beta, t) * E),
    })
    return exp_spectrum.groupby('sample').sum()

def compute_g_single_time(spectrum, beta, t):
    """Compute g(beta,t) = Z(beta,t) Z*(beta,t) by sample."""
    g_by_sample = np.square(np.abs(
        compute_Z_by_sample(spectrum, beta, t)))
    return pd.DataFrame(
        { 'g' : g_by_sample['Z'] },
        index = g_by_sample.index,
        columns = ['g'],
    )

def compute_g_by_sample(spectrum, beta, times):
    """Compute g(beta,t) for all times, indexed by sample. 
    The columns are the different times."""
    samples = spectrum['sample'].unique()
    g = pd.DataFrame(index=samples, columns=[])
    for t_ind in range(0,len(times)):
        g[t_ind] = compute_g_single_time(spectrum, beta, t=times[t_ind])
    return g

def compute_dataframe_mean(data):
    """Compute the mean of the given dataframe across samples.  The
    dataframe index is the sample number. It can contain different
    columns, which can correspond (for example) to different times.
    The answer includes one column, indexed by the dataframe's
    column headers."""
    return data.mean()

def combine_into_timeseries(data, times):
    """Combine data and times into a dataframe with two columns,
    't' and 'value'. 'data' and 'times' are dataframes with a single
    column, both indexed in the same way."""
    return pd.DataFrame({ 't' : times, 'value' : data })

def compute_g_mean(spectrum, beta, times):
    """Compute <g(t)>"""
    g_by_sample = compute_g_by_sample(spectrum, beta, times)
    return pd.DataFrame({ 't' : times, 'value' : g_by_sample.mean() })

def compute_gg_single_sample(g_single_sample, dt_ind):
    """Compute g(0)g(dt) at a single time point."""
    gg1 = g_single_sample[0]
    gg2 = g_single_sample[dt_ind]
    return gg1 * gg2

def compute_gg_mean(g, dt_ind):
    """Computes average of g(t)g(t+dt) for each sample in g.
    g should be indexed by sample number."""
    g0gt = g.apply(
        lambda g_single_sample : compute_gg_single_sample(g_single_sample, dt_ind),
        axis=1)
    return g0gt.mean()

def compute_fractional_g_variance(g_by_sample, times):
    """Compute (<g(t)^2> - <g(t)>^2) / <g(t)>^2"""
    gg_mean = g_by_sample.applymap(lambda x : x*x).mean()
    g_mean_sqr = np.square(g_by_sample.mean())
    return pd.DataFrame({
        't' : times,
        'value' : (gg_mean - g_mean_sqr) / g_mean_sqr,
    })

def get_dt_indices(dt, max_dt):
    return pd.Series(np.arange(0,int(max_dt/dt),1))

def compute_autocorrelation(spectrum, beta, times):
    g = compute_g_by_sample(spec, beta, times)

    # Compute the correlation with t=0
    autocorr = g.corrwith(g[0])

    # Get the times relative to the starting time
    dts = times - times[0]

    return pd.DataFrame({ 't' : dts, 'vlaue' : autocorr})

def get_filename(N, beta, label, extension):
    return ('data/autocorr/maj-N%d-b%d-%s.%s'
            % (N, int(beta), label, extension))

def get_figure_filename(N, beta, label):
    return ('plots/maj-N%d-b%d-%s.png' % (N, int(beta), label))

def get_tsv_filename(N, beta, label):
    return get_filename(N, beta, label, 'tsv')

def save_autocorrelation(autocorr, N, beta, label):
    autocorr.to_csv(
        get_tsv_filename(N,beta,"autocorrelation-" + label),
        sep='\t',
        header=['# dt', 'value'],
        index=False,
    )

def run_g():
    # Plot g over the whole range, useful for testing
    print "Computing g..."
    all_times = pd.Series(np.logspace(0,6,200))
    all_g = compute_g_by_sample(spec, beta, times=all_times)
    all_g_mean = pd.DataFrame({ 't' : all_times, 'value' : all_g.mean() })
    all_g_fractional_variance = compute_fractional_g_variance(
        all_g, all_times)

    # Save <g>, var(g)/<g>^2
    all_g_mean.to_csv(
        ('data/autocorr/maj-N%d-b%d-g.tsv' % (N, int(beta))),
        sep='\t',
        header=['# t', 'value'],
        index=False,
    )
    all_g_fractional_variance.to_csv(
        ('data/autocorr/maj-N%d-b%d-g-fractional-var.tsv' % (N, int(beta))),
        sep='\t',
        header=['# t', 'value'],
        index=False,
    )

    plotutils.plot_dataframe(
        all_g_mean, 
        title=("<g(t)>  N=%d beta=%.0f" % (N,beta)), 
        logx=True, logy=True)
    plotutils.plot_dataframe(
        all_g_fractional_variance, 
        title=("(<g(t)^2> - <g(t)>^2) / <g(t)>^2  N=%d beta=%.0f"
                % (N,beta)),
        logx=True,
        logy=True
    )

    plotutils.plot_two_dataframes(
        all_g_mean, all_g_fractional_variance,
        title=("g(t) mean and fractional variance  N=%d beta=%.0f" % (N,beta)),
        ylabel1='<g(t)>',
        ylabel2='frac. var.',
        logx=True, logy=True,
        save_to_file=get_figure_filename(N, beta, "g-frac-var")
    )

def run_early_autocorr():
    print "Computing early time autocorrelations..."
    early_times = pd.Series(np.arange(early_ti,early_tf,dt))
    early_autocorr = compute_autocorrelation(spec, beta, early_times)
    save_autocorrelation(early_autocorr, N, beta, 'early')
    plotutils.plot_dataframe(
        early_autocorr, 
        title=('Early autocorrelations N=%d beta=%.0f t0=%.0f'
            % (N,beta,early_times[0])),
    )

def run_ramp_autocorr():
    print "Computing ramp time autocorrelations..."
    ramp_times = pd.Series(np.arange(ramp_ti,ramp_tf,dt))
    ramp_autocorr = compute_autocorrelation(spec, beta, ramp_times)
    save_autocorrelation(ramp_autocorr, N, beta, 'ramp')
    plotutils.plot_dataframe(
        ramp_autocorr, 
        title=('Ramp autocorrelations N=%d beta=%.0f t0=%.0f'
            % (N,beta,ramp_times[0])),
    )

def run_plateau_autocorr():
    print "Computing plateau time autocorrelations..."
    plateau_times = pd.Series(np.arange(plateau_ti,plateau_tf,dt))
    plateau_autocorr = compute_autocorrelation(
        spec, beta, plateau_times)
    save_autocorrelation(plateau_autocorr, N, beta, 'plateau')
    plotutils.plot_dataframe(
        plateau_autocorr, 
        title=('Plateau autocorrelations N=%d beta=%.0f t0=%0.f'
            % (N,beta,plateau_times[0])),
    )

beta = 1

# N = 22
# ramp_ti = 300
# ramp_tf = ramp_ti + 50
# max_samp = None
# plateau_ti = 1e5
# plateau_tf = plateau_ti + 50

N = 26
ramp_ti = 500
ramp_tf = ramp_ti + 1
plateau_ti = 1e5
plateau_tf = plateau_ti + 2
max_samp = None

# N = 30
# ramp_ti = 1e3
# ramp_tf = ramp_ti + 50
# plateau_ti = 1e6
# plateau_tf = plateau_ti + 50
# max_samp = None

# N = 32
# ramp_ti = 1e3
# ramp_tf = ramp_ti + 50
# plateau_ti = 1e6
# plateau_tf = plateau_ti + 50
# max_samp = None

early_ti = 1
early_tf = 30

dt = 0.1

# Load the spectrum
print "Loading the spectrum..."
spec = read_spectrum(
    ('data/spectra/maj-N%d-spectrum.tsv' % N), 
    max_samples=max_samp)
# spec = read_spectrum(
#     'data/spectra/test-spectrum.tsv',
#     max_samples=max_samp)

# auto = compute_autocorrelation(spec, 1, pd.Series([1,10]))

run_g()

run_early_autocorr()
run_ramp_autocorr()
run_plateau_autocorr()
