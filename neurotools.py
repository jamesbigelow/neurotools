
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Toolbox for analyzing and plotting neuronal spike data ( e.g., PSTHs, MPHs, BSCs, STRFs, FRAs, CCGs, classifiers, information theoretic analyses, etc. ).

@author: jamesbigelow at gmail dot com  

"""

#### Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps
from scipy import fftpack as fft
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def bsc_drop_spikes( bsc, n_spikes ):

    '''
    
    Drop a single spike per trial until target n_spikes is reached (for spike-equated analyses).
        
    INPUT -------
    bsc : Binned spike count matrix, n trials x k time points (numpy array) 
    n_spikes : Desired number of spikes (bsc.sum())


    RETURN -------
    bsc: Spike-count reduced version of input BSC ( sum == n_spikes )
        
    '''
    
    assert bsc.sum() >= n_spikes
    
    n_spikes0 = bsc.sum()
    idx_trial = 0
    while n_spikes0 > n_spikes:
        idx_spike = np.nonzero( bsc[idx_trial,:] )[0]
        if idx_spike.size > 0:
           idx_drop_spike = int( idx_spike[ np.random.randint(idx_spike.size, size=1) ] )
           bsc[idx_trial,idx_drop_spike] -= 1
        
        n_spikes0 = bsc.sum()
           
        if idx_trial < bsc.shape[0]-1:
            idx_trial += 1
        else:
            idx_trial = 0
        
    return bsc   


def classify_responses( tspk, tref, stim_vals, t1, t2, binwidth, opt_plot=False ):
    
    '''

    Classify single-trial responses to a set of stimuli using leave-one-out cross validation.
    Each single trial response is compared to averaged responses for all stimuli in terms of Euclidean distance. 
    The response is classified as the stimulus with minimum Euclidean distance. 

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    stim_vals : stimulus value for each trial (same size as tref) 
    t1 : start of window 
    t2 : end of window (must be >t1)
    binwidth: temporal resolution for binned responses
    opt_plot: optional plot of confusion matrix (true/false)
    
    RETURN -------
    accuracy : classifier accuracy (proportion correct)
    cm : confusion matrix
    
    References:
        Bigelow, J., & Malone, B. (2020). Extracellular voltage thresholds for maximizing information extraction in primate auditory cortex: implications for a brain computer interface. Journal of neural engineering.
        Foffani, G., & Moxon, K. A. (2004). PSTH-based classification of sensory stimuli using ensembles of single neurons. Journal of neuroscience methods, 135(1-2), 107-120.

        
    '''

    # First, calc PSTHs for each unique stim val
    stim_vals_unique = np.unique( stim_vals )
    bin_edges = np.arange( t1, t2, binwidth )
    psth_all = np.zeros( (stim_vals_unique.size, bin_edges.size), dtype=np.float64 )
    for ii in range( stim_vals_unique.size ):
        tref_i = tref[ stim_vals_unique[ii] == stim_vals ]
        psth_all[ii,:] = get_psth( tspk, tref_i, t1, t2, binwidth )

    # Second, classify single-trial responses according to the nearest trial-averaged PSTH (Euclidean distance)
    cm = np.zeros( (stim_vals_unique.size, stim_vals_unique.size), dtype=np.float64 )
    for ii in range( stim_vals_unique.size ):
        psth_all_tmp = psth_all
        tref_i = tref[ stim_vals_unique[ii] == stim_vals ]
        for jj in range( tref_i.size ):
            tref_j = tref_i[jj]
            tref_i_tmp = np.setdiff1d( tref_i, tref_j )
            psth_j = get_psth( tspk, tref_j, t1, t2, binwidth ) # single-trial response
            psth_i = get_psth( tspk, tref_i_tmp, t1, t2, binwidth ) # PSTH recalculated without classified trial
            psth_all_tmp[ii,:] = psth_i
            # Calc Euclidean distance between response and all PSTHs
            euc_dist = np.zeros( stim_vals_unique.size, dtype=float )
            for kk in range( stim_vals_unique.size ):
                euc_dist[kk] = np.sqrt( sum( ( psth_j - psth_all_tmp[kk,:] ) ** 2 ) )
            cm[ ii, np.argmin( euc_dist ).min() ] += 1 # classify by min Euclidean distance
 
    accuracy = np.sum( np.diag( cm ) ) / np.sum(cm)
    
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( cm, cmap='gray_r', aspect='auto' )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Actual')
        plt.ylabel('Classified')
        plt.title( f'Proportion trials correct = %0.4f' % accuracy )
            
    return accuracy, cm   
     
   
def filt_1d( x, n, window='gaussian' ):
    
    '''
    
    1-dimensional smoothing filter.

    INPUT -------
    x : 2-D numpy array to be filtered 
    n : filter size in bins (uniform)
    window: Smoothing function, options == 'gaussian', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'

    RETURN -------
    y : filtered x
    
    '''
    
    # Define kernel
    if window == 'gaussian':
        w = sps.gaussian( n, std=np.floor(n/5) )
    else:
        w = eval( 'np.'+window+'(n)' )  
    x0 = np.r_[x[n-1:0:-1],x,x[-2:-n-1:-1]] # Pad edges
    y = np.convolve(w/w.sum(),x0,mode='same') # Filter
    y = y[n-1:-n+1] # Remove padded edges
    
    return y


def filt_2d_box( x, n ):
    
    '''
    
    Uniform 2-D boxcar filter.
    
    INPUT -------
    x : 2-D numpy array to be filtered 
    n : filter size in bins (uniform)

    RETURN -------
    y : filtered x
    
    Example: 
    x = np.array([[    17,    24,     1,     8,    15],
    [23,     5,     7,    14,    16],
     [4,     6,    13,    20,    22],
    [10,    12,    19,    21,     3],
    [11,    18,    25,     2,     9]])

    n = 3
    
    '''
    
    # Pad (replicate) edges to avoid edge effects
    n_pad = round((n-1)/2)
    x = np.vstack(( np.array([x[0,:],]*n_pad), x, np.array([x[-1,:],]*n_pad) ))
    x = np.hstack(( np.array([x[:,0],]*n_pad).transpose(), x, np.array([x[:,-1],]*n_pad).transpose() ))
    # Filter
    filt = np.ones((n,n))/(n*n)
    y = sps.convolve2d( x, filt, 'valid' )
    
    return y


def get_bsc( tspk, tref, t1, t2, binwidth, opt_plot=False ):
    
    '''
    
    Get binned spike count matrix from spike and event times.
    
    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of window 
    t2 : end of window (must be >t1)
    binwidth : binwidth for time bins 
    opt_plot : optional plot (true/false)

    RETURN -------
    bsc : Binned spike count matrix, n trials x k time points (numpy array) 
           
    Notes: 
        1) units of input variables must be the same (e.g., sec or ms)
        2) all trials must have equal duration ( t2 - t1 )
                                                              
    '''
    
    bin_edges = np.arange( t1, t2+binwidth, binwidth )
    bsc = np.zeros( (tref.size,bin_edges.size-1), dtype=np.float64)
    if tref.size > 1:
        for ii in range(tref.size):
            bsc[ii,:], bin_edges_i = np.histogram( tspk - tref[ii], bins = bin_edges )
    else:
        bsc, bin_edges_i = np.histogram( tspk - tref, bins = bin_edges )
        
    if opt_plot: 
        fig, ax = plt.subplots()
        im = ax.imshow( bsc, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(bin_edges), max(bin_edges), 0, bsc.shape[0] ] )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time (s)')
        plt.ylabel('Trial')
    
    return bsc


def get_ccg( x, y, maxlag ):
    
    '''
    
    Get cross-correlogram function. 
    For auto-correlogram x and y are the same.
    
    INPUT -------
    x : binned spike count vector for unit 1 
    y : binned spike count vector for unit 2, same size as x 
    maxlag : number of bins for analysis window (same units as x and y), usually substantially smaller than x & y
    opt_plot : optional plot of CCG function (true/false)

    RETURN -------
    ccg : cross-correlogram 
        
    Attribution: partially incorporates code modified from 
        https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy

    '''
    
    assert maxlag < x.size-1
    assert x.size-1 > 0 
    
    m = int( 2 ** np.ceil( np.log2( 2 * x.size -1 ) ) )
    X = fft.fft(x, m)
    Y = fft.fft(y, m)
    c = np.real( fft.ifft( X * np.conj(Y) ) )
    idx1 = np.arange(1, maxlag+1, 1) + (m - maxlag -1)
    idx2 = np.arange(1, maxlag+2, 1) - 1
    ccg = np.hstack( ( c[idx1], c[idx2] ) )
    
    if np.array_equal( x, y ):
        ccg[maxlag] = 0

    return ccg
    

def get_classifier_opt_binwidth( tspk, tref, stim_vals, t1, t2, binwidths, opt_plot ):
    
    '''
    
    Estimate optimal temporal encoding resolution by finding binwidth that maximizes classifier accuracy.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    stim_vals : stimulus value for each trial (same size as tref) 
    t1 : start of window (must be <0 for baseline calcs)
    t2 : end of window (must be >t1)
    binwidths : range of temporal resolutions for calculating psth 
    opt_plot: true/false for plot
    
    RETURN -------
    binwidth_opt :  reliability, mean of subsampled PSTH correlations
    accuracy_smooth : smoothed binwidth x accuracy function
        
    '''
    
    accuracy = np.zeros( binwidths.size, dtype=np.float64)
    for ii in range(accuracy.size):
        accuracy[ii], cm = classify_responses( tspk, tref, stim_vals, t1, t2, binwidths[ii] )
             
    # 3-point mean smoothing     
    f = np.ones(3)
    f = f/f.sum()
    accuracy_smooth = np.convolve( f, accuracy, mode='same' )
    binwidth_opt = binwidths[ np.argmax(accuracy_smooth) ]

    if opt_plot:
        fig, ax = plt.subplots()
        plt.plot( binwidths, accuracy_smooth, color = 'black', linewidth=1 )
        plt.plot( binwidths, accuracy, color = [0.5, 0.5, 0.5], marker='o', linewidth=0 )
        plt.plot( binwidth_opt, accuracy_smooth[ np.argmax(accuracy_smooth) ], color='red', marker='*', markersize=15, linewidth=0 )
        plt.xscale('log')
        plt.xlabel('Binwidth (s)')
        plt.ylabel('Classifier accuracy (p[correct])')
        plt.title( f'Opt binwidth = %0.4f s' % binwidth_opt )
   
    return binwidth_opt, accuracy_smooth
    

def get_cm_random( n_stim, n_trials ):
    
    '''
    
    Get confusion matrix populated by random guesses. 
    Useful null condition for comparing results from classify_responses and get_mutual_info_from_cm.

    INPUT -------
    n_stim : number of stimuli, i.e., classifier choices (dimension of confusion matrix) 
    n_trials : number of trials, i.e., random guesses to make (sum of confusion matrix)
    
    RETURN -------
    cm :  confusion matrix
        
    '''

    cm = np.zeros( ( n_stim, n_stim ), dtype=float )
    for ii in range( n_trials ):
        idx = np.random.randint(0, high=n_stim, size=2)
        cm[ idx[0], idx[1] ] += 1
        
    return cm

    
def get_empirical_nonlinearity( xprior, xposterior, dt, n_bins=13, opt_plot=False ):
    
    '''
    
    Get empirical nonliniearity (input/output function) from p(x) and p(x|spike) distributions. 
        
    INPUT -------
    xprior : prior distribution, p(x) 
    xposterior : posterior distribution, p(x|spike)
    dt : delta time (temporal bin width), s
    n_bins : number of bins for histogram calcs
    opt_plot: true/false for plot
 
    RETURN -------
    px : standardized prior distribution probability, p(x) 
    px_spk : standardized posterior distribution probability, p(x|spike)  
    pspk_x : IO function, spiking probability p(spike|x)
    pspk_x_hz : IO function, spikes/s
    nx : number of data points for each bin, px
    nx_spk : number of data points for each bin, px_spk 
    mean_hz : Mean firing rate (spikes/s)
    x_bin_centers : histogram bin centers for all output functions (standardized)
    x_bin_edges : histogram bin edges for all output functions (standardized)
    x_bin_centers_raw : histogram bin centers for all output functions (non-standardized)
    x_bin_edges_raw : histogram bin edges for all output functions (non-standardized)    

    '''
    
    # Standardize prior and posterior distributions 
    n_total = xprior.size
    n_spk = xposterior.size
    x_mean = xprior.mean()
    x_std  = xprior.std()
    x_prior_std = (xprior - x_mean) / x_std
    x_posterior_std = (xposterior - x_mean) / x_std
    x_bin_edges_raw = np.linspace( xprior.min(), xprior.max(), n_bins )
    x_bin_centers_raw = (x_bin_edges_raw[1:] + x_bin_edges_raw[:-1])/2
    
    # Define probability distributions and values
    x_bin_edges = np.linspace( x_prior_std.min(), x_prior_std.max(), n_bins )
    x_bin_centers = (x_bin_edges[1:] + x_bin_edges[:-1])/2
    nx = np.histogram( x_prior_std, x_bin_edges )[0] 
    px = nx / nx.sum() # p(x)
    nx_spk = np.histogram( x_posterior_std, x_bin_edges )[0]
    px_spk = nx_spk / n_spk # p(x|spk)
    pspk = n_spk / n_total
    pspk_x = px_spk * pspk / px # IO function: spiking probability
    pspk_x_hz = pspk_x / dt # IO function: spikes/s
    mean_hz = pspk / dt
    
    if opt_plot: 
        fig, axs = plt.subplots(2, 1)

        axs[0].plot( x_bin_centers, px, color=(0.5, 0.5, 0.5), marker='o', label='p(x)')
        axs[0].plot( x_bin_centers, px_spk, color='k', marker='o', label='p(x|spk)')
        axs[0].plot( x_bin_centers, pspk_x, color='r', marker='o', label='p(spk|x)')  
        axs[0].plot([0, 0], axs[0].get_ylim() , 'k:')
        axs[0].legend()
        axs[0].set_xlabel('STRF-stim projection (SD)')
        axs[0].set_ylabel('Probability')
        
        axs[1].plot( x_bin_centers, pspk_x_hz, color='r', marker='o')        
        axs[1].plot([0, 0], axs[1].get_ylim() , 'k:')
        axs[1].plot(axs[1].get_xlim(), [mean_hz, mean_hz], 'r:')
        axs[1].set_xlabel('STRF-stim projection (SD)')
        axs[1].set_ylabel('Firing rate (Hz)')                
        
        fig.tight_layout()
        plt.show()
        
    return px, px_spk, pspk_x, pspk_x_hz, nx, nx_spk, mean_hz, x_bin_centers, x_bin_edges, x_bin_centers_raw, x_bin_edges_raw


def get_fra( tspk, tref, freq, atten, t1, t2, opt_plot=False ): 
    
    '''
    
    Frequency response area function: Spike count x Frequency x Attenuation. 
    
    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    freq: tone frequency for each trial (same size as tref)
    atten: tone attenuation for each trial (same size as tref) 
    t1 : start of window 
    t2 : end of window (must be >t1)
    opt_plot: true/false for plot
    
    RETURN -------
    fra : Frequency-Response Area function, i.e., mean firing rate evoked by tones of each freq-atten combination
       
    '''
    
    freq_unique = np.unique( freq )
    atten_unique = np.unique( atten )
    fra = np.zeros( ( atten_unique.size, freq_unique.size) )
    for ff in range(freq_unique.size):
        for aa in range(atten_unique.size): 
            idx_fa = np.nonzero( (freq == freq_unique[ff]) & (atten == atten_unique[aa]) )[0]
            nspk = np.zeros( (idx_fa.size) )
            for tt in range(idx_fa.size):
                nspk[tt] = tspk[ (tspk > tref[ idx_fa[tt] ]+t1 ) & ( tspk <= tref[ idx_fa[tt] ]+t2 ) ].size
            fra[aa,ff] = np.nanmean( nspk )
    fra[ np.isnan(fra)] = 0
        
    if opt_plot: 
        fig, ax = plt.subplots()
        im = ax.imshow( fra, cmap='inferno_r', aspect='auto', extent=[ min(freq_unique), max(freq_unique), max(atten_unique), min(atten_unique) ] )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Attenuation (dB)')

    return fra 


def get_fra_params( fra, faxis, aaxis ):
    
    '''
    
    FRA params 
   
    INPUT -------
    fra : Frequency-Response Area, n freq x n atten bins (see get_fra)
    faxis: frequency axis of FRA in kHz
    aaxis: attenuation axis of FRA in dB
    
    RETURN -------
    p : dictionary with FRA params 
    
    '''
    
    # Specify cutoff and freq atten binwidths 
    cutoff_val = 0.75 # Specify cutoff
    
    # Define freq atten binwidths    
    n_oct = round( np.log2( max(faxis) / min(faxis) ), ndigits=2 )
    n_freq_per_oct = round( faxis.size / n_oct, ndigits=2 )
    n_oct_per_bin = 1/( faxis.size / n_oct )
    n_db_per_bin = aaxis[1] - aaxis[0]

    # Freq and atten marginals
    fmarginal = np.sum( fra, axis=0 )
    amarginal = np.sum( fra, axis=1 )
    
    # Basic params - - - - - - - - -
    p = {}   
    p['numel'] = np.nonzero( abs(fra) > 0 )[0].size
    p['sum'] = fra.sum()
    if p['numel'] > 0:
        p['mean'] = np.mean( fra[abs(fra)>0])
        p['median'] = np.median( fra[abs(fra)>0])
        p['max'] = np.max( fra[abs(fra)>0])
        p['min'] = np.min( fra[abs(fra)>0])
        p['range'] = p['max'] - p['min'] + 1
    else:
        p['mean'] = float('nan')
        p['median'] = float('nan')
        p['max'] = float('nan')
        p['min'] = float('nan')
        p['range'] = float('nan')
        
    # Freq params - - - - - - - - -  
    xvec = fmarginal 
    if sum( abs( xvec ) ) > 0:
        idx_max = np.argmax( xvec ).min()
        idx_submax = np.nonzero( xvec < max(xvec)*cutoff_val )[0]
        if np.nonzero( idx_submax < idx_max )[0].size == 0:
            xmin = 0
        else:
            xmin = max( idx_submax[ np.nonzero( idx_submax <= idx_max )[0] ] )
        if np.nonzero( idx_submax > idx_max )[0].size == 0:
            xmax = xvec.size-1
        else:
            xmax = min( idx_submax[ np.nonzero( idx_submax >= idx_max )[0] ] )
        p['best_freq'] = faxis[idx_max]
        p['bandwidth'] = ( xmax - xmin + 1 ) * n_oct_per_bin
        p['bandwidth_total'] = ( np.nonzero( xvec >= max(xvec)*cutoff_val )[0].size + 2 ) * n_oct_per_bin
    else: 
        p['best_freq'] = float('nan')
        p['bandwidth'] = float('nan')
        p['bandwidth_total'] = float('nan')
        
    # Atten params - - - - - - - - - 
    xvec = amarginal
    if sum( abs( xvec ) ) > 0:
        idx_max = np.argmax( xvec ).min()
        idx_submax = np.nonzero( xvec < max(xvec)*cutoff_val )[0]
        if np.nonzero( idx_submax < idx_max )[0].size == 0:
            xmin = 0
        else:
            xmin = max( idx_submax[ np.nonzero( idx_submax <= idx_max )[0] ] )
        if np.nonzero( idx_submax > idx_max )[0].size == 0:
            xmax = xvec.size-1
        else:
            xmax = min( idx_submax[ np.nonzero( idx_submax >= idx_max )[0] ] )
        p['best_atten'] = aaxis[idx_max]
        p['threshold'] = aaxis[xmax]
    else: 
        p['best_atten'] = float('nan')
        p['threshold'] = float('nan')
        
    return p 


def get_fra_reliability( tspk, tref, freq, atten, t1, t2, n_iter, opt_plot=False ): 
    
    '''
    
    Estimate reliability of FRA by calculating correlation coefficient between FRAs calculated from random trial halves.
    
    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    freq: tone frequency for each trial (same size as tref)
    atten: tone attenuation for each trial (same size as tref) 
    t1 : start of window 
    t2 : end of window (must be >t1)
    n_iter : number of iterations to repeat FRA calcs from random trial subsets
    opt_plot: true/false for plot
    
    RETURN -------
    r_mean : mean correlation coefficient between STRFs calculated from random trial subsets
    r_zscore : r_mean relative to null distribution ( [r_mean - r_mean_null] / r_std_null )
    p_val : proportion of difference distribution (reliability - null) falling at or below zero
       
    '''

    # Subsampled FRA correlation distributions      
    n_iter = int(n_iter)
    r_dist = np.zeros(n_iter,dtype=np.float64) 
    r_dist_null = np.zeros(n_iter,dtype=np.float64) 
    for ii in range(n_iter):
        idx_a = np.random.choice( tref.size, int( np.ceil(tref.size/2)), replace=0 )
        idx_b = np.setdiff1d( np.arange(0,tref.size), idx_a )
        idx_ar = np.random.choice( idx_a.size, int( np.ceil(idx_a.size)), replace=0 )
        idx_br = np.random.choice( idx_b.size, int( np.ceil(idx_b.size)), replace=0 )

        fra_a = get_fra(tspk, tref[idx_a], freq[idx_a], atten[idx_a], t1, t2 )
        fra_b = get_fra(tspk, tref[idx_b], freq[idx_b], atten[idx_b], t1, t2 )

        fra_a_null = get_fra(tspk, tref[idx_a[idx_ar]], freq[idx_a], atten[idx_a], t1, t2 )
        fra_b_null = get_fra(tspk, tref[idx_b[idx_br]], freq[idx_b], atten[idx_b], t1, t2 )
               
        r_dist[ii] = np.corrcoef( np.reshape( fra_a, fra_a.size ), np.reshape( fra_b, fra_b.size ) )[0,1]
        r_dist_null[ii] = np.corrcoef( np.reshape( fra_a_null, fra_a_null.size ), np.reshape( fra_b_null, fra_b_null.size ) )[0,1]
    
    # FRA reliability
    r_mean = np.nanmean(r_dist)           
    r_mean_null = np.nanmean(r_dist_null) 
    r_zscore = ( r_mean - r_mean_null ) / np.nanstd( r_dist_null )
    p_val = (np.nonzero( r_dist_null >= r_mean )[0].size) / n_iter 
    # p_val = (np.nonzero(((r_dist-r_dist_null)<=0) )[0].size) / n_iter # alt method 
    
    if opt_plot:
        fig, ax = plt.subplots()
        ax.hist( r_dist, bins=np.arange(-0.5,1.05,0.025), color='r', alpha = 0.5, label='data' )
        ax.hist( r_dist_null, bins=np.arange(-0.5,1.05,0.025), color='k', alpha = 0.5, label='null' )
        plt.ylim(plt.ylim())
        ax.plot([r_mean, r_mean], ax.get_ylim() , 'r:')
        ax.plot([r_mean_null, r_mean_null], ax.get_ylim() , 'k:')
        plt.xlabel('Subsampled FRA correlations')
        plt.ylabel('Count')
        ax.legend()

    return r_mean, r_zscore, p_val        
    

def get_mph( tspk, tref, t1, t2, mod_rate, n_bins, opt_plot=False ):
    
    '''
    
    Modulation period histogram of spike event times.    

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of stimulus window 
    t2 : end of stimulus window (must be >t1)
    mod_rate : stimulus modulation rate (Hz)
    n_bins: number of temporal bins for calculating mph 
    opt_plot: optional MPH plot (true/false)
    
    RETURN -------
    mph : modulation period histogram
        
    '''
    
    n_periods = int((t2-t1)*mod_rate)                
    x = np.empty([0],dtype=np.float64)
    for ii in range(tref.size):
        for kk in range(n_periods):
            trig0 = tref[ii]+t1+(kk*(1/mod_rate))
            x = np.concatenate( (x, tspk[ (tspk > trig0 ) & ( tspk <= trig0+(1/mod_rate) ) ] - trig0 ) )
                                                                                                           
    binwidth = 1/mod_rate/n_bins       
    mph, bin_edges = np.histogram( x, bins = np.arange( 0, (1/mod_rate) +binwidth, binwidth ) )
    mph = mph / ( tref.size * n_periods )
      
    # bin_edges = np.arange( 0, (1/mod_rate)+binwidth, binwidth )
                                                
    if opt_plot: 
        
        fig, axs = plt.subplots(2, 1)
        clr = [0.1, 0.1, 0.1]

        # BSC
        bsc = np.zeros([ tref.size*n_periods, bin_edges.size ], dtype=np.float64 )
        for ii in range(tref.size):
            for kk in range(n_periods):
                tref0 = tref[ii]+t1+(kk*(1/mod_rate))
                bsc[(ii*n_periods)+kk,:-1], bin_edges0 = np.histogram( tspk - tref0, bins = bin_edges )
        bsc = bsc[:,:-1] # correct edge matching for last bin  
        axs[0].imshow( bsc, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(bin_edges), max(bin_edges), 0, bsc.shape[0] ] )
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Period')
        
        # MPH
        clr = [0.1, 0.1, 0.1]
        mph_sem = np.std( bsc, axis=0 ) / np.sqrt(bsc.shape[0])
        axs[1].plot( bin_edges[1:], mph, color=clr )        
        axs[1].fill_between( bin_edges[1:], mph-mph_sem, mph+mph_sem, alpha = 0.25, facecolor=clr )
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Spikes/period')  
        axs[1].set_xlim(( min(bin_edges), max(bin_edges) ))              
        
        fig.tight_layout()
        plt.show()
    
    return mph


def get_mph_simple( tspk, tref, t1, t2, mod_rate, n_bins ):
    
    '''
    
    Modulation period histogram of spike event times.    
    Simpler, faster version of get_mph useful for large datasets that don't require plotting.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of stimulus window 
    t2 : end of stimulus window (must be >t1)
    mod_rate : stimulus modulation rate (Hz)
    n_bins: number of temporal bins for calculating mph 
    
    RETURN -------
    mph : modulation period histogram
        
    '''
    
    n_periods = int((t2-t1)*mod_rate)                
    x = np.empty([0],dtype=np.float64)
    for ii in range(tref.size):
        for kk in range(n_periods):
            trig0 = tref[ii]+t1+(kk*(1/mod_rate))
            x = np.concatenate( (x, tspk[ (tspk > trig0 ) & ( tspk <= trig0+(1/mod_rate) ) ] - trig0 ) )
                                                                                                           
    binwidth = 1/mod_rate/n_bins       
    mph, bin_edges = np.histogram( x, bins = np.arange( 0, (1/mod_rate) +binwidth, binwidth ) )
    mph = mph / ( tref.size * n_periods )
    
    return mph


def get_mph_reliability( tspk, tref, t1, t2, mod_rate, n_bins, n_iter=100, opt_plot=False ):
    
    '''
    
    Estimate reliability of MPH by calculating correlation coefficient between MPHs calculated from random trial halves.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of stimulus window 
    t2 : end of stimulus window (must be >t1)
    mod_rate : stimulus modulation rate (Hz)
    n_bins: number of temporal bins for calculating mph 
    n_iter : number of subsample iterations
    opt_plot : true/false for plot of subsampled MPH correlations
    
    RETURN -------
    r_mean :  reliability, mean of subsampled PSTH correlations
    r_zscore : reliability z-score relative to null mean and SD
    p_val : reliability p-value, proportion of null dist > r_mean 
        
    '''
    
    # Binned spike count matrices
    n_periods = int((t2-t1)*mod_rate)  
    binwidth = 1/mod_rate/n_bins 
    bin_edges = np.arange( 0, (1/mod_rate)+binwidth, binwidth )
    bsc = np.zeros([ tref.size*n_periods, bin_edges.size ], dtype=np.float64)
    bsc_null = np.zeros([ tref.size*n_periods, bin_edges.size ], dtype=np.float64)
    for ii in range(tref.size):
        for kk in range(n_periods):
            tref0 = tref[ii]+t1+(kk*(1/mod_rate))
            bsc[(ii*n_periods)+kk,:-1], bin_edges0 = np.histogram( tspk - tref0, bins = bin_edges )
            bsc_null[(ii*n_periods)+kk,:] = np.roll( bsc[(ii*n_periods)+kk,:], np.random.randint( low = 0, high = bin_edges.size ) )
  
    # Subsampled MPH correlation distributions      
    n_iter = int(n_iter)
    r_dist = np.zeros(n_iter,dtype=np.float64) 
    r_dist_null = np.zeros(n_iter,dtype=np.float64) 
    for ii in range(n_iter):
        idx_a = np.random.choice( bsc.shape[0], int(np.ceil(tref.size/2)), replace=False )
        idx_b = np.setdiff1d( np.arange(0,bsc.shape[0]), idx_a )
        r = np.corrcoef( np.mean( bsc[idx_a,:], axis=0 ), np.mean( bsc[idx_b,:], axis=0 ) )
        r_dist[ii] = r[0,1]
        r_null = np.corrcoef( np.mean( bsc_null[idx_a,:], axis=0 ), np.mean( bsc_null[idx_b,:], axis=0 ) )
        r_dist_null[ii] = r_null[0,1]
    
    # Reliability
    r_mean = np.nanmean(r_dist)           
    r_mean_null = np.nanmean(r_dist_null) 
    r_zscore = ( r_mean - r_mean_null ) / np.nanstd( r_dist_null )
    p_val = (np.nonzero( r_dist_null >= r_mean )[0].size) / n_iter
    
    if opt_plot:
        fig, ax = plt.subplots()
        ax.hist( r_dist, bins=np.arange(-0.5,1.05,0.025), color='r', alpha = 0.5, label='data' )
        ax.hist( r_dist_null, bins=np.arange(-0.5,1.05,0.025), color='k', alpha = 0.5, label='null' )
        plt.ylim(plt.ylim())
        ax.plot([r_mean, r_mean], ax.get_ylim() , 'r:')
        ax.plot([r_mean_null, r_mean_null], ax.get_ylim() , 'k:')
        plt.xlabel('Subsampled MPH correlations')
        plt.ylabel('Count')
        ax.legend()
    
    return r_mean, r_zscore, p_val

    
def get_mod_rate_depth_psth_data( tspk, tref, mod_rates, mod_depths, t1, t2, binwidth, n_iter ):
    
    '''
    
    Get PSTH firing rate and reliability data from set of modulation rates.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    mod_rates : vector of modulation rate values for each trial (Hz)
    mod_depths : vector of modulation depth values for each trial (Hz)
    t1 : start of analysis window 
    t2 : end of analysis window (must be >t1)
    binwidth: temporal resolution for calculating psth 
    n_iter : number of subsample iterations
    
    RETURN -------
    rate_mean : mean firing rate for each modulation rate and depth (matrix)
    r_mean :  reliability for each modulation rate and depth (matrix)
    r_zscore : reliability z-score for each modulation rate and depth (matrix)
    p_val : reliability p-value for each modulation rate and depth (matrix) 
        
    '''
    
    mod_rate_unique = np.unique( mod_rates )
    mod_depth_unique = np.unique( mod_depths )
    rate_mean = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    r_mean = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    r_zscore = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    p_val = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    for ii in range( mod_depth_unique.size ):
        for jj in range( mod_rate_unique.size ): 
            idx_stim = np.nonzero( (mod_rates == mod_rate_unique[jj]) & (mod_depths == mod_depth_unique[ii]) )[0]
            r_mean[ii,jj], r_zscore[ii,jj], p_val[ii,jj] = get_psth_reliability( tspk, tref[idx_stim], t1, t2, binwidth, n_iter )
            psth = get_psth( tspk, tref[idx_stim], t1, t2, binwidth, False )
            rate_mean[ii,jj] = np.nansum( psth )
            
    rate_mean = np.squeeze(rate_mean)        
    r_mean = np.squeeze(r_mean)        
    r_zscore = np.squeeze(r_zscore)        
    p_val = np.squeeze(p_val)  
      
    return rate_mean, r_mean, r_zscore, p_val


def get_mod_rate_depth_mph_data( tspk, tref, mod_rates, mod_depths, t1, t2, n_bins, n_iter ):
    
    '''
    
    Get MPH firing rate and reliability data from set of modulation rates.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    mod_rates : vector of modulation rate values for each trial (Hz)
    mod_depths : vector of modulation depth values for each trial (Hz)
    t1 : start of analysis window 
    t2 : end of analysis window (must be >t1)
    n_bins: number of temporal bins for calculating mph 
    n_iter : number of subsample iterations
    
    RETURN -------
    rate_mean : mean firing rate for each modulation rate and depth (matrix)
    r_mean :  reliability for each modulation rate and depth (matrix)
    r_zscore : reliability z-score for each modulation rate and depth (matrix)
    p_val : reliability p-value for each modulation rate and depth (matrix) 
        
    '''
    
    mod_rate_unique = np.unique( mod_rates )
    mod_depth_unique = np.unique( mod_depths )
    r_mean = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    r_zscore = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    p_val = np.zeros( (mod_depth_unique.size, mod_rate_unique.size), dtype=np.float64 )
    for ii in range( mod_depth_unique.size ):
        for jj in range( mod_rate_unique.size ): 
            idx_stim = np.nonzero( (mod_rates == mod_rate_unique[jj]) & (mod_depths == mod_depth_unique[ii]) )[0]
            r_mean[ii,jj], r_zscore[ii,jj], p_val[ii,jj] = get_mph_reliability( tspk, tref[idx_stim], t1, t2, mod_rate_unique[jj], n_bins, n_iter )
            
    r_mean = np.squeeze(r_mean)        
    r_zscore = np.squeeze(r_zscore)        
    p_val = np.squeeze(p_val)      
    
    return r_mean, r_zscore, p_val


def get_mutual_info_extrap( xprior, xposterior, dt=0.005, n_bins=13, opt_plot=False ):
    
    '''
    
    Similar to get_mutual_info, but extrapolated to infinite dataset by calculating info from portions of xprior, xposterior distributions
    
    INPUT -------
    xprior : prior distribution, p(x) 
    xposterior : posterior distribution p(x|spike)
    dt : delta time (temporal bin width), sec
    n_bins : number of bins for histogram calcs
    opt_plot: true/false for plot
  
    RETURN -------
    info_extrap : Extrapolated Shannon mutual information (bits/spike)
    
    Reference: Atencio, C. A., & Schreiner, C. E. (2016). Functional congruity in local auditory cortical microcircuits. Neuroscience, 316, 402-419.


    '''
    
    data_pct = np.array([90, 92.5, 95, 97.5, 99, 100]) # Calc info using these percentages of full dataset
    n_iter = 20 # number of times to repeat info calc from each data percentage (random draw each iter)  
    n_spikes = xposterior.size

    # Store info for each data pct, iter - - - - - - - - - - 
    info_data_pct_iter = np.zeros((n_iter,data_pct.size),dtype=np.float64)
    for ii in range(data_pct.size):
        n_spikes_pct = int( np.round( data_pct[ii]/100 * n_spikes) )
        for jj in range(n_iter):
            xposterior_pct = np.random.choice( xposterior, n_spikes_pct, replace=False ) # get subset of the posterior distribution
            px, px_spk, pspk_x, pspk_x_hz, nx, nx_spk, mean_hz, x_bin_centers, x_bin_edges, x_bin_centers_raw, x_bin_edges_raw = get_empirical_nonlinearity( xprior, xposterior_pct, dt, n_bins )
            info_data_pct_iter[jj,ii] = get_mutual_info( px, px_spk )

    # Extrapolate - - - - - - - - - -  
    info_mean = np.mean(info_data_pct_iter,axis=0)
    x = 1 / data_pct
    y = info_mean 
    beta = np.polyfit(x,y,1)
    info_extrap = beta[1]

    return info_extrap


def get_mutual_info_from_cm( cm ):
    
    '''
    
    Estimate mutual information from the response classifier confusion matrix. 
    See classify_responses.
    
    INPUT -------
    cm : confusion matrix
 
    RETURN -------
    info : Shannon mutual information (bits)
    
    Reference: 
        Kayser, C., Logothetis, N. K., & Panzeri, S. (2010). Millisecond encoding precision of auditory cortex neurons. Proceedings of the National Academy of Sciences, 107(39), 16976-16981.
    
    '''
    
    cmi = np.zeros( cm.shape, dtype=float)
    n_trials = np.sum( cm )
    for rr in range(cm.shape[0]):
        for cc in range(cm.shape[0]):
            pxy = cm[rr,cc] / n_trials
            px = np.sum( cm[:,cc] ) / n_trials
            py = np.sum( cm[rr,:] ) / n_trials
            if pxy==0 or px==0 or py==0:
                cmi[rr,cc] = pxy * np.log2( pxy/(px*py) )
            else:
                cmi[rr,cc] = pxy * np.log2( pxy/(px*py) )
    info = np.nansum( cmi )
    
    return info


def get_mutual_info( px, px_spk ):
    
    '''
    
    Estimate mutual information between stimulus and response. 
    Calculated from p(x) and p(x|spike) distributions as calculated by get_strf_stimulus_projections.
    
    INPUT -------
    px : standardized prior distribution probability, p(x) 
    px_spk : standardized posterior distribution probability, p(x|spike)
 
    RETURN -------
    info : Shannon mutual information (bits/spike)
    
    Reference: 
        Atencio, C. A., & Schreiner, C. E. (2016). Functional congruity in local auditory cortical microcircuits. Neuroscience, 316, 402-419.

    '''
    
    idx = np.where( (px > 0) & (px_spk > 0) )[0] # avoid dividing by zero
    info = sum( px_spk[idx] * np.log2( px_spk[idx] / px[idx] ) )
    
    return info


def get_nonlinearity_data( strf, stim, bsc, opt_plot=False ): 
    
    '''
    
    Get dictionary containing all information relevant to empirical nonlinearity from STRF, stimulus, and response (BSC).

    INPUT -------
    strf : [Filter] Spectro-temporal receptive field aka stimulus filter, n freq x n time bins (numpy array) 
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    opt_plot: true/false for plot    
  
    RETURN -------
    d : dictionary with nonlinearity data (probability distributions, mutual info, etc.)
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: strf.shape
        Out: (32, 40) 
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)
    
    '''
    
    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]
    assert stim.shape[0] == strf.shape[0]
    
    # Hard code some args for now ###
    strf_temp_res = 0.005 # sec
    n_bins_info_hist = 10

    xprior, xposterior = get_strf_stimulus_projections( strf, stim, bsc )
    px, px_spk, pspk_x, pspk_x_hz, nx, nx_spk, mean_hz, x_bin_centers, x_bin_edges, x_bin_centers_raw, x_bin_edges_raw = get_empirical_nonlinearity( xprior, xposterior, strf_temp_res, n_bins_info_hist, opt_plot )
    info = get_mutual_info( px, px_spk )
    info_extrap = get_mutual_info_extrap( px, px_spk )
       
    d = {}
    d['px'] = px
    d['px_spk'] = px_spk
    d['pspk_x'] = pspk_x
    d['pspk_x_hz'] = pspk_x_hz
    d['nx'] = nx
    d['nx_spk'] = nx_spk
    d['mean_hz'] = mean_hz
    d['x_bin_centers'] = x_bin_centers
    d['x_bin_edges'] = x_bin_edges
    d['x_bin_centers_raw'] = x_bin_centers_raw
    d['x_bin_edges_raw'] = x_bin_edges_raw
    d['info'] = info
    d['info_extrap'] = info_extrap

    return d 


def get_psth( tspk, tref, t1, t2, binwidth, opt_plot=False ):
    
    '''

    Peristimulus-time histogram of spike event times.    

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of window (must be <0 for baseline calcs)
    t2 : end of window (must be >t1)
    binwidth: temporal resolution for calculating psth 
    opt_plot: true/false for plot
    
    RETURN -------
    psth : peristimulus-time histogram
        
    '''

    tspk_aligned = np.zeros([0],dtype=np.float64)
    if tref.size > 1:
        for ii in range(tref.size):     
            tspk_aligned = np.concatenate( (tspk_aligned, tspk[ (tspk > tref[ii]+t1 ) & ( tspk <= tref[ii]+t2 ) ] - tref[ii] ) )
    else:
        tspk_aligned = np.concatenate( (tspk_aligned, tspk[ (tspk > tref+t1 ) & ( tspk <= tref+t2 ) ] - tref ) )
    psth, bin_edges = np.histogram( tspk_aligned, bins = np.arange( t1, t2+binwidth, binwidth) )
    psth = psth / tref.size

    if opt_plot:

        fig, axs = plt.subplots(2, 1)

        # BSC
        bsc = get_bsc( tspk, tref, t1, t2, binwidth )
        axs[0].imshow( bsc, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(bin_edges), max(bin_edges), 0, bsc.shape[0] ] )
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Trial')
        
        # PSTH
        clr = [0.1, 0.1, 0.1]
        psth_sem = np.std( bsc, axis=0 ) / np.sqrt(bsc.shape[0])
        axs[1].plot( bin_edges[1:], psth, color=clr )        
        axs[1].fill_between( bin_edges[1:], psth-psth_sem, psth+psth_sem, alpha = 0.25, facecolor=clr )
        # axs[1].plot( [0, 0], axs[1].get_ylim(), 'k:')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Spikes/trial')  
        axs[1].set_xlim(( min(bin_edges), max(bin_edges) ))              
        
        fig.tight_layout()
        plt.show()
 
    return psth


def get_psth_latency( tspk, tref, t1, t2, binwidth ):
    
    '''
    
    Estimate onset, offset, and peak latency from smoothed PSTH. 
    Analysis window defined by t1 and t2 should include baseline time, i.e., t1 should be negative to include time before stimulus onset.
    
    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of window (must be <0 for baseline calcs)
    t2 : end of window (must be >t1)
    binwidth: temporal resolution for calculating psth 
    
    RETURN -------
    onset_latency : first time point exceeding baseline + 2.5 SD
    offset_latency : first time point returning to baseline + 2.5 SD range
    peak_latency : time point at which PSTH reaches max value 
        
    '''
    
    # Hard code a couple args
    cutoff_std = 2.5
    cutoff_bins = 4
    
    bin_edges = np.arange( t1, t2, binwidth )
    psth = get_psth( tspk, tref, t1, t2, binwidth )   
    psth = sps.savgol_filter( psth, window_length = 5, polyorder = 3 ) # smooth
    psth_pos = psth[bin_edges>0]
    bin_edges_pos = bin_edges[bin_edges>0]
    
    bl_mean = np.mean( psth[bin_edges<0] )
    bl_std = np.std( psth[bin_edges<0] )
    
    idx_above = np.nonzero( psth_pos > bl_mean+bl_std*cutoff_std )[0]
    idx_below = np.nonzero( psth_pos <= bl_mean+bl_std*cutoff_std )[0]
    
    onset_latency = bin_edges_pos[ idx_above.min() ]
    offset_latency = bin_edges_pos[ min( idx_below[ idx_below > idx_above.min() ] ) ]
    peak_latency = bin_edges_pos[ np.argmax( psth_pos ) ]    
    
    return onset_latency, offset_latency, peak_latency


def get_psth_opt_binwidth( tspk, tref, t1, t2, binwidths, n_iter, opt_plot=False ):
    
    '''
    
    Estimate optimal temporal encoding resolution by finding PSTH binwidth that maximizes reliability z-score (difference from null) 

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of window (must be <0 for baseline calcs)
    t2 : end of window (must be >t1)
    binwidths : range of temporal resolutions for calculating psth 
    n_iter : number of subsample iterations
    opt_plot: true/false for plot
    
    RETURN -------
    binwidth_opt :  reliability, mean of subsampled PSTH correlations
    r_zscore_smooth : smoothed binwidth x reliability z-score function
        
    '''
    
    r_zscore = np.zeros( binwidths.size, dtype=np.float64)
    for ii in range(r_zscore.size):
        r_mean, r_zscore[ii], p_val = get_psth_reliability( tspk, tref, t1, t2, binwidths[ii], n_iter )
      
    # 3-point mean smoothing     
    f = np.ones(3)
    f = f/f.sum()
    r_zscore_smooth = np.convolve( f, r_zscore, mode='same' )
    binwidth_opt = binwidths[ np.argmax(r_zscore_smooth) ]

    if opt_plot:
        fig, ax = plt.subplots()
        plt.plot( binwidths, r_zscore_smooth, color = 'black', linewidth=1 )
        plt.plot( binwidths, r_zscore, color = [0.5, 0.5, 0.5], marker='o', linewidth=0 )
        plt.plot( binwidth_opt, r_zscore_smooth[ np.argmax(r_zscore_smooth) ], color='red', marker='*', markersize=15, linewidth=0 )
        plt.xscale('log')       
        plt.xlabel('Binwidth (s)')
        plt.ylabel('PSTH reliability (z-score)')
        plt.title( f'Opt binwidth = %0.4f s' % binwidth_opt )
   
    return binwidth_opt, r_zscore_smooth


def get_psth_reliability( tspk, tref, t1, t2, binwidth, n_iter=100, opt_plot=False ):
    
    '''
    
    Estimate reliability of PSTH by calculating correlation coefficient between PSTHs calculated from random trial halves.

    INPUT -------
    tspk : spike times 
    tref : reference event times, 0 relative to t1 and t2
    t1 : start of window (must be <0 for baseline calcs)
    t2 : end of window (must be >t1)
    binwidth : temporal resolution for calculating psth 
    n_iter : number of subsample iterations
    opt_plot : true/false for plot of subsampled PSTH correlations
    
    RETURN -------
    r_mean :  reliability, mean of subsampled PSTH correlations
    r_zscore : reliability z-score relative to null mean and SD
    p_val : reliability p-value, proportion of null dist > r_mean 
        
    '''
    
    # Binned spike count matrices 
    bin_edges = np.arange( t1, t2+binwidth, binwidth )
    bsc = np.zeros( (tref.size,bin_edges.size-1), dtype=np.float64)
    bsc_null = np.zeros( (tref.size,bin_edges.size-1),dtype=np.float64)
    for ii in range(tref.size):
        bsc[ii,:], bin_edges_i = np.histogram( tspk - tref[ii], bins = bin_edges )
        bsc_null[ii,:] = np.roll( bsc[ii,:], np.random.randint( low = 0, high = bin_edges.size ) )

    # Subsampled PSTH distributions      
    n_iter = int(n_iter)
    r_dist = np.zeros(n_iter,dtype=np.float64) 
    r_dist_null = np.zeros(n_iter,dtype=np.float64) 
    for ii in range(n_iter):
        idx_a = np.random.choice( tref.size, int(np.ceil(tref.size/2)), replace=False )
        idx_b = np.setdiff1d( np.arange(0,tref.size), idx_a )
        r = np.corrcoef( np.mean( bsc[idx_a,:], axis=0 ), np.mean( bsc[idx_b,:], axis=0 ) )
        r_dist[ii] = r[0,1]
        r_null = np.corrcoef( np.mean( bsc_null[idx_a,:], axis=0 ), np.mean( bsc_null[idx_b,:], axis=0 ) )
        r_dist_null[ii] = r_null[0,1]
    
    # Reliability 
    r_mean = np.nanmean(r_dist)           
    r_mean_null = np.nanmean(r_dist_null) 
    r_zscore = ( r_mean - r_mean_null ) / np.nanstd( r_dist_null )
    p_val = (np.nonzero( r_dist_null >= r_mean )[0].size) / n_iter
    
    if opt_plot:
        fig, ax = plt.subplots()
        ax.hist( r_dist, bins=np.arange(-0.5,1.05,0.025), color='r', alpha = 0.5, label='data' )
        ax.hist( r_dist_null, bins=np.arange(-0.5,1.05,0.025), color='k', alpha = 0.5, label='null' )
        plt.ylim(plt.ylim())
        ax.plot([r_mean, r_mean], ax.get_ylim() , 'r:')
        ax.plot([r_mean_null, r_mean_null], ax.get_ylim() , 'k:')
        plt.xlabel('Subsampled PSTH correlations')
        plt.ylabel('Count')
        ax.legend()
    
    return r_mean, r_zscore, p_val
  

def get_psth_sparseness( psth ):
    
    '''
    
    PSTH lifetime spareseness of PSTH, i.e., how many spikes go into how many bins
    Sparseness is high when many spikes are concentrated in few bins 
    Reference: Vinje, W. E., & Gallant, J. L. (2000). Sparse coding and decorrelation in primary visual cortex during natural vision. Science, 287(5456), 1273-1276.


    INPUT -------
    psth : Peristimulus time histogram vector (see get_psth) 

    
    RETURN -------
    sparseness : PSTH lifetime spareseness value 
        
    '''
    
    psth_mean = np.nanmean( psth )
    psth_std = np.nanstd( psth )
    n = (psth_mean**2) / ( (psth_mean**2) + (psth_std**2) )
    d = 1 - (1/psth.size)
    sparseness = 1 - (n/d)

    return sparseness 
    
   
def get_spike_triggered_sum( bsc, stim, n_bins_pre, n_bins_post, opt_plot=False ):

    '''
    
    Cumulative sum of windowed stimulus values preceding each spike in BSC.
    
    INPUT -------
    bsc : binned spike count vector, i.e., spike counts binned in n time bins
    stim : stimulus matrix, n time bins x m freq bins
    n_bins_pre : pre-spike bins to include in caclulation
    n_bins_post : post-spike bins to include in calculation 
    opt_plot: true/false for plot
    
    RETURN -------
    strf : stimulus filter, n freq x n time bins
    n_spikes : total spike count  
    
    
    '''

    assert stim.shape[1] == bsc.shape[0]
    
    n_spikes = 0 
    spike_triggered_sum = np.zeros( [stim.shape[0], n_bins_pre + n_bins_post], dtype=np.float64 )
    for ii in np.arange(n_bins_pre-1,bsc.size-n_bins_post): 
        # print(f' %1.0f\n' % ii )
        if bsc[ii] != 0: 
            n_spikes += bsc[ii]
            spike_triggered_sum = spike_triggered_sum + bsc[ii] * stim[ :,np.arange( ii-n_bins_pre+1,ii+n_bins_post+1) ]
    
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( spike_triggered_sum, cmap='RdBu_r', origin='lower')
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time bin')
        plt.ylabel('Frequency bin')
        
    return spike_triggered_sum, n_spikes


def get_strf( stim, bsc, opt_null=False, opt_plot=False, opts_strf_calc=False ):
    
    '''
    
    Spectro-temporal receptive field: time-frequency encoding model 

    INPUT -------
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    opt_null: true/false for null STRF version reflecting time-reversed stimulus
    opt_plot: true/false for plot    
    opts_strf_calc: options for STRF calc, default if not provided
    
    RETURN -------
    strf : stimulus filter, n freq x n time bins. Mean-subtracted & smoothed response-triggered mean of stim
    n_spikes : total spike count  
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)
    
    '''
    
    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]
    
    # Unpack calc opts
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005 # sec
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3
    
    strf_temp_res = opts_strf_calc['strf_temp_res']
    n_bins_pre = opts_strf_calc['n_bins_pre']
    n_bins_post = opts_strf_calc['n_bins_post']
    n_bins_filt = opts_strf_calc['n_bins_filt'] # must be odd integer 
    
    # Calc spike-triggered sum
    spike_triggered_sum = np.zeros( ( stim.shape[0], n_bins_pre + n_bins_post), dtype=np.float64 )
    n_spikes = 0
    for ii in range(bsc.shape[0]):
        stimulus = stim[:,:,ii]
        stimulus = stimulus-stimulus.mean() # Mean subtraction 
        if opt_null:
            stimulus = np.fliplr( stimulus )
        spike_triggered_sum_i, n_spikes_i = get_spike_triggered_sum( bsc[ii,:], stimulus, n_bins_pre, n_bins_post )
        spike_triggered_sum += spike_triggered_sum_i
        n_spikes += n_spikes_i

    # Divide by spike count and smooth 
    strf = spike_triggered_sum / n_spikes
    strf = filt_2d_box( strf, n_bins_filt )
    
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( strf, cmap='RdBu_r', origin='lower', aspect='auto')
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time bin')
        plt.ylabel('Frequency bin')
        m = np.max(np.abs(strf))
        im.set_clim(-m,m)
        
    return strf, n_spikes


def get_strf_data( stim, bsc, opt_null=False, opts_strf_calc=False ):
    
    '''
    
    Get data relevant to STRF, mutual info, empirial nonlinearity, etc.
    
    INPUT -------
    
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    opt_null: true/false for null STRF version reflecting time-reversed stimulus
    opts_strf_calc: options for STRF calc, default if not provided
    
    RETURN -------
    d : dictionary with strf data 
    
    '''
    
    # Unpack calc opts
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005 # sec
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3
    
    strf_temp_res = opts_strf_calc['strf_temp_res']
    n_bins_pre = opts_strf_calc['n_bins_pre']
    n_bins_post = opts_strf_calc['n_bins_post']
    n_bins_filt = opts_strf_calc['n_bins_filt'] # must be odd integer 

    n_bins_info_hist = 10
   
    opt_plot = False
    strf, n_spikes = get_strf_hz( stim, bsc, opt_null, opt_plot, opts_strf_calc )
    xprior, xposterior = get_strf_stimulus_projections( strf, stim, bsc )   
    px, px_spk, pspk_x, pspk_x_hz, nx, nx_spk, x_bin_centers = get_empirical_nonlinearity( xprior, xposterior, strf_temp_res, n_bins_info_hist, opt_plot )
    info = get_mutual_info( px, px_spk )
    
    d = {}
    d['strf'] = strf
    d['n_spikes'] = n_spikes
    d['strf_temp_res'] = strf_temp_res
    d['n_bins_pre'] = n_bins_pre
    d['n_bins_post'] = n_bins_post
    d['info'] = info
    d['px'] = px
    d['nx'] = nx
    d['pspk_x_hz'] = pspk_x_hz
    d['x_bin_centers'] = x_bin_centers
        
    return d 


def get_strf_difference_reliability( stim_0, bsc_0, stim_1, bsc_1, n_iter=100, opt_plot=False, opts_strf_calc=False ):
    
    '''

    Estimate reliability of difference STRFs (difference between two conditions) by calculating correlation coefficient between difference STRFs calculated from random trial halves.

    INPUT -------
    stim_0 : [Stimulus] - Condition 0: Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc_0 : [Response] - Condition 0: Binned spike count matrix, n trials x k time points (numpy array) 
    stim_1 : [Stimulus] - Condition 1: Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc_1 : [Response] - Condition 1: Binned spike count matrix, n trials x k time points (numpy array) 
    n_iter : number of iterations to repeat STRF calcs from random trial subsets
    opt_plot: true/false for plot    
    
    RETURN -------
    r_mean : mean correlation coefficient between difference STRFs calculated from random trial subsets
    r_zscore : r_mean relative to null distribution ( [r_mean - r_mean_null] / r_std_null )
    p_val : proportion of difference distribution (reliability - null) falling at or below zero
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)
    
    '''
    
    assert stim_0.shape[-1] == bsc_0.shape[0], stim_0.shape[1] == bsc_0.shape[1]
    assert stim_1.shape[-1] == bsc_1.shape[0], stim_1.shape[1] == bsc_1.shape[1]
    
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3
    
    # Difference STRF reliability distributions      
    n_iter = int(n_iter)
    r_dist = np.zeros(n_iter,dtype=np.float64) 
    r_dist_null = np.zeros(n_iter,dtype=np.float64) 
    for ii in range(n_iter):
        
        idx_a = np.random.choice( bsc_0.shape[0], int( np.ceil(bsc_0.shape[0]/2)), replace=0 )
        idx_b = np.setdiff1d( np.arange(0,bsc_0.shape[0]), idx_a )
        
        # STRFs condition 0 ----------
        strf_0_a, n_spikes = get_strf_hz( stim_0[:,:,idx_a], bsc_0[idx_a,:] )
        strf_0_b, n_spikes = get_strf_hz( stim_0[:,:,idx_b], bsc_0[idx_b,:] )
        strf_0_a_null, n_spikes =  get_strf_hz( stim_0[:,:,idx_a], bsc_0[idx_a,:], opt_null=1 )
        strf_0_b_null, n_spikes =  get_strf_hz( stim_0[:,:,idx_b], bsc_0[idx_b,:], opt_null=1 )
        
        # STRFs condition 1 ----------
        # idx_a = np.random.choice( bsc_1.shape[0], int( np.ceil(bsc_1.shape[0]/2)), replace=0 )
        # idx_b = np.setdiff1d( np.arange(0,bsc_1.shape[0]), idx_a )
        strf_1_a, n_spikes = get_strf_hz( stim_1[:,:,idx_a], bsc_1[idx_a,:] )
        strf_1_b, n_spikes = get_strf_hz( stim_1[:,:,idx_b], bsc_1[idx_b,:] )
        strf_1_a_null, n_spikes =  get_strf_hz( stim_1[:,:,idx_a], bsc_1[idx_a,:], opt_null=1 )
        strf_1_b_null, n_spikes =  get_strf_hz( stim_1[:,:,idx_b], bsc_1[idx_b,:], opt_null=1 )
        
        # Difference STRFs: condition 1 - 0 ----------
        strf_a = strf_1_a - strf_0_a
        strf_b = strf_1_b - strf_0_b
        strf_a_null = strf_1_a_null - strf_0_a_null
        strf_b_null = strf_1_b_null - strf_0_b_null
        
        r_dist[ii] = np.corrcoef( np.reshape( strf_a, strf_a.size ), np.reshape( strf_b, strf_b.size ) )[0,1]
        r_dist_null[ii] = np.corrcoef( np.reshape( strf_a_null, strf_a_null.size ), np.reshape( strf_b_null, strf_b_null.size ) )[0,1]
    
    r_mean = np.nanmean(r_dist)           
    r_mean_null = np.nanmean(r_dist_null) 
    r_zscore = ( r_mean - r_mean_null ) / np.nanstd( r_dist_null )
    p_val = (np.nonzero( r_dist_null >= r_mean )[0].size) / n_iter 
    # p_val = (np.nonzero(((r_dist-r_dist_null)<=0) )[0].size) / n_iter # alt method 
    
    if opt_plot:
        fig, ax = plt.subplots()
        ax.hist( r_dist, bins=np.arange(-0.5,1.05,0.025), color='r', alpha = 0.5, label='data' )
        ax.hist( r_dist_null, bins=np.arange(-0.5,1.05,0.025), color='k', alpha = 0.5, label='null' )
        ax.plot([r_mean, r_mean], ax.get_ylim() , 'r:')
        ax.plot([r_mean_null, r_mean_null], ax.get_ylim() , 'k:')
        plt.xlabel('Subsampled Difference STRF correlations')
        plt.ylabel('Count')
        ax.legend()

    return r_mean, r_zscore, p_val


def get_strf_ei_balance( strf ):
    
    '''
    
    Calc STRF excitation-inhibition balance. 
    Difference / sum ratio of positive and negative STRF values.
    Excitation- and Inhibition- dominated STRFs have values near 1 and -1, respectively, and balanced STRFs have values near 0. 

    INPUT -------
    strf : stimulus filter, n freq x n time bins (may be raw or sig version of STRF)

    RETURN -------
    ei_balance : excitation-inhibition balance value 
    
    '''
    
    
    strf_exc = np.zeros(strf.shape,dtype=np.float64)
    strf_exc[:] = strf[:]
    strf_exc[strf_exc<0]= 0
    
    strf_inh = np.zeros(strf.shape,dtype=np.float64)
    strf_inh[:] = strf[:]
    strf_inh[strf_inh>0] = 0 
    strf_inh = abs(strf_inh)

    ei_balance = ( strf_exc.sum() - strf_inh.sum() ) / ( strf_exc.sum() + strf_inh.sum() )

    return ei_balance
    
    
def get_strf_hz( stim, bsc, opt_null=False, opt_plot=False, opts_strf_calc=False ):
    
    '''
    
    Same as get_strf except time-freq bins expressed in firing rate units (hz). 

    INPUT -------
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    opt_null: true/false for null STRF version reflecting time-reversed stimulus
    opt_plot: true/false for plot 
    opts_strf_calc: options for STRF calc, default if not provided
    
    RETURN -------
    strf : stimulus filter expressed in firing rate (hz), n freq x n time bins. 
    n_spikes : total spike count  
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)
        
    References: 
    DeCharms, R. C., Blake, D. T., & Merzenich, M. M. (1998). Optimizing sound features for cortical neurons. Science, 280(5368), 1439-1444.
    Rutkowski, R. G., Shackleton, T. M., Schnupp, J. W., Wallace, M. N., & Palmer, A. R. (2002). Spectrotemporal receptive field properties of single units in the primary, dorsocaudal and ventrorostral auditory cortex of the guinea pig. Audiology and Neurotology, 7(4), 214-227.

    
    '''
    
    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]

    # Unpack calc opts
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3
    
    strf_temp_res = opts_strf_calc['strf_temp_res'] # sec
    n_bins_pre = opts_strf_calc['n_bins_pre']
    n_bins_post = opts_strf_calc['n_bins_post']
    n_bins_filt = opts_strf_calc['n_bins_filt'] # must be odd integer 
    
    spike_triggered_sum = np.zeros( ( stim.shape[0], n_bins_pre + n_bins_post), dtype=np.float64 )
    n_spikes = 0
    n_time_bins = 0
    psf = np.zeros( stim.shape[0], dtype=np.float64 ) # p(stim[freq]), probability of stimulus frequency
    for ii in range(bsc.shape[0]):
        stimulus = stim[:,:,ii]
        if opt_null:
            stimulus = np.fliplr( stimulus )
        spike_triggered_sum_i, n_spikes_i = get_spike_triggered_sum( bsc[ii,:], stimulus, n_bins_pre, n_bins_post )
        spike_triggered_sum += spike_triggered_sum_i
        n_spikes += n_spikes_i
        n_time_bins += stimulus.shape[1] - n_bins_pre - n_bins_post + 1
        psf += np.sum(stimulus[:,np.arange( n_bins_pre-1,stimulus.shape[1]-n_bins_post)],axis=1)

    n_sec = (n_time_bins * strf_temp_res)   
    mean_rate = n_spikes / n_sec; 
    psft = np.tile( psf / n_time_bins, (spike_triggered_sum.shape[1],1)).T # p(stim[freq,time]), probability of stimulus frequency @ time bin
    psft_spk = spike_triggered_sum / n_spikes # Spike-triggered average: p(stim[freq,time]|spike) Probability of stimulus frequency @ time bin given that a spike has occurred. Eq. (1) in Rutkowski et al.
    pspk = n_spikes / n_time_bins  # p(spike|x), probability of a spike occurring in a bin
    pspk_sft = psft_spk * pspk / psft # p(spike|stim[freq,time]) Probability of spike given that frequency @ time bin has occurred Eq. (2) in Rutkowski et al.
    strf = ( pspk_sft / strf_temp_res ) - mean_rate # Divide by binwidth to express in spikes/sec, subtract mean rate to reflect deviation above/below mean 
    strf = filt_2d_box( strf, n_bins_filt )
    
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( strf, cmap='RdBu_r', origin='lower', aspect='auto' )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time bin')
        plt.ylabel('Frequency bin')
        m = np.max(np.abs(strf))
        im.set_clim(-m,m)
        
    return strf, n_spikes


def get_strf_params( strf, taxis, faxis ):
    
    '''
    
    Get various time-frequency params from STRF (tuning, latency, bandwidth, etc.).
   
    INPUT -------
    strf : **absolute valued** STRF, i.e., stimulus filter, n freq x n time bins (may be raw or sig version of STRF)
    taxis: time axis of STRF in milliseconds. Time before spike is positive, e.g., taxis ranges between 200 - 0 for STRF calculated using 0.2 sec window before spike
    faxis: frequency axis of STRF in Hz
    
    RETURN -------
    p : dictionary with STRF params 
 
    Note: This function is intended to be used with absolute-valued STRFs or STRF subfields. Not intended for raw STRFs with +/- values. 
    e.g., 
    
        strf_abs = np.zeros(strf.shape,dtype=np.float64)
        strf_abs[:] = abs( strf_sig[:] )
        d_params_abs = nt.get_strf_params( strf_abs, taxis, faxis ) 

        strf_exc = np.zeros(strf.shape,dtype=np.float64)
        strf_exc[:] = strf_sig[:]
        strf_exc[strf_exc<0]= 0
        d_params_exc = nt.get_strf_params( strf_exc, taxis, faxis )
        
        strf_inh = np.zeros(strf.shape,dtype=np.float64)
        strf_inh[:] = strf_sig[:]
        strf_inh[strf_inh>0] = 0 
        strf_inh = abs(strf_inh)
        d_params_inh = nt.get_strf_params( strf_inh, taxis, faxis ) 
        
        strf = np.zeros(strf_sig.shape,dtype=np.float64)
        strf[:] = strf_abs[:]
        strf[:] = strf_exc[:]
        strf[:] = strf_inh[:]
    
    '''
    
    assert np.nonzero( strf<0 )[0].size == 0
        
    # Set cutoff
    cutoff_val = 0.5
    
    # Drop post-spike time
    strf = strf[ :, taxis >= 0 ]
    
    # Temporal and frequency marginals
    tmarginal = np.sum( strf, axis=0 )
    fmarginal = np.sum( strf, axis=1 )
    
    n_oct = round( np.log2( max(faxis) / min(faxis) ), ndigits=2 )
    n_freq_per_oct = round( faxis.size / n_oct, ndigits=2 )
    n_oct_per_bin = 1/( faxis.size / n_oct )
    n_ms_per_bin = taxis[0] - taxis[1]

    # Basic params - - - - - - - - -
    p = {}   
    p['numel'] = np.nonzero( abs(strf) > 0 )[0].size
    p['sum'] = strf.sum()
    if p['numel'] > 0:
        p['mean'] = np.mean( strf[abs(strf)>0])
        p['median'] = np.median( strf[abs(strf)>0])
        p['max'] = np.max( strf[abs(strf)>0])
        p['min'] = np.min( strf[abs(strf)>0])
        p['range'] = p['max'] - p['min'] + 1
    else:
        p['mean'] = float('nan')
        p['median'] = float('nan')
        p['max'] = float('nan')
        p['min'] = float('nan')
        p['range'] = float('nan')
        
    # Temp params - - - - - - - - - 
    xvec = tmarginal
    if sum( abs( xvec ) ) > 0:
        idx_max = np.argmax( xvec ).min()
        idx_submax = np.nonzero( xvec < max(xvec)*cutoff_val )[0]
        if np.nonzero( idx_submax < idx_max )[0].size == 0:
            xmin = 0
        else:
            xmin = max( idx_submax[ np.nonzero( idx_submax <= idx_max )[0] ] )
        if np.nonzero( idx_submax > idx_max )[0].size == 0:
            xmax = xvec.size-1
        else:
            xmax = min( idx_submax[ np.nonzero( idx_submax >= idx_max )[0] ] )
        p['latency_peak'] = taxis[idx_max]
        p['latency_onset'] = taxis[xmax]
        p['latency_offset'] = taxis[xmin]
        p['integration_time'] = ( xmax - xmin + 1 ) * n_ms_per_bin
        p['integration_time_total'] = ( np.nonzero( xvec >= max(xvec)*cutoff_val )[0].size + 2 ) * n_oct_per_bin
    else: 
        p['latency_peak'] = float('nan')
        p['latency_onset'] = float('nan')
        p['latency_offset'] = float('nan')
        p['integration_time'] = float('nan')
        p['integration_time_total'] = float('nan')
        
    # Freq params - - - - - - - - -  
    xvec = fmarginal 
    if sum( abs( xvec ) ) > 0:
        idx_max = np.argmax( xvec ).min()
        idx_submax = np.nonzero( xvec < max(xvec)*cutoff_val )[0]
        if np.nonzero( idx_submax < idx_max )[0].size == 0:
            xmin = 0
        else:
            xmin = max( idx_submax[ np.nonzero( idx_submax <= idx_max )[0] ] )
        if np.nonzero( idx_submax > idx_max )[0].size == 0:
            xmax = xvec.size-1
        else:
            xmax = min( idx_submax[ np.nonzero( idx_submax >= idx_max )[0] ] )
        p['best_freq'] = faxis[idx_max]
        p['bandwidth'] = ( xmax - xmin + 1 ) * n_oct_per_bin
        p['bandwidth_total'] = ( np.nonzero( xvec >= max(xvec)*cutoff_val )[0].size + 2 ) * n_oct_per_bin
    else: 
        p['best_freq'] = float('nan')
        p['bandwidth'] = float('nan')
        p['bandwidth_total'] = float('nan')
        
    return p 


def get_strf_reliability( stim, bsc, n_iter=100, opt_plot=False, opts_strf_calc=False ):
    
    '''
    
    Estimate reliability of STRF by calculating correlation coefficient between STRFs calculated from random trial halves.

    INPUT -------
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    n_iter : number of iterations to repeat STRF calcs from random trial subsets
    opt_plot: true/false for plot    
    opts_strf_calc: options for STRF calc, default if not provided
   
    RETURN -------
    r_mean : mean correlation coefficient between STRFs calculated from random trial subsets
    r_zscore : r_mean relative to null distribution ( [r_mean - r_mean_null] / r_std_null )
    p_val : proportion of difference distribution (reliability - null) falling at or below zero
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)
        
    Reference: 
    Escabí, M. A., Read, H. L., Viventi, J., Kim, D. H., Higgins, N. C., Storace, D. A., ... & Cohen, Y. E. (2014). A high-density, high-channel count, multiplexed μECoG array for auditory-cortex recordings. Journal of neurophysiology, 112(6), 1566-1583.

    
    '''
    
    strf_hz_model = True
    
    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]
    
    # Default calc opts
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005 # sec
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3

    # Subsampled STRF correlation distributions      
    n_iter = int(n_iter)
    r_dist = np.zeros(n_iter,dtype=np.float64) 
    r_dist_null = np.zeros(n_iter,dtype=np.float64) 
    opt_plot0 = False
    for ii in range(n_iter):
        idx_a = np.random.choice( bsc.shape[0], int( np.ceil(bsc.shape[0]/2)), replace=0 )
        idx_b = np.setdiff1d( np.arange(0,bsc.shape[0]), idx_a )
        
        if strf_hz_model:
            opt_null = False
            strf_a, n_spikes = get_strf_hz( stim[:,:,idx_a], bsc[idx_a,:], opt_null, opt_plot0, opts_strf_calc )
            strf_b, n_spikes = get_strf_hz( stim[:,:,idx_b], bsc[idx_b,:], opt_null, opt_plot0, opts_strf_calc )
            opt_null = True
            strf_a_null, n_spikes =  get_strf_hz( stim[:,:,idx_a], bsc[idx_a,:], opt_null, opt_plot0, opts_strf_calc )
            strf_b_null, n_spikes =  get_strf_hz( stim[:,:,idx_b], bsc[idx_b,:], opt_null, opt_plot0, opts_strf_calc )
        else:
            opt_null = False
            strf_a, n_spikes = get_strf( stim[:,:,idx_a], bsc[idx_a,:], opt_null, opt_plot0, opts_strf_calc )
            strf_b, n_spikes = get_strf( stim[:,:,idx_b], bsc[idx_b,:], opt_null, opt_plot0, opts_strf_calc )
            opt_null = True
            strf_a_null, n_spikes =  get_strf( stim[:,:,idx_a], bsc[idx_a,:], opt_null, opt_plot0, opts_strf_calc )
            strf_b_null, n_spikes =  get_strf( stim[:,:,idx_b], bsc[idx_b,:], opt_null, opt_plot0, opts_strf_calc )
               
        r_dist[ii] = np.corrcoef( np.reshape( strf_a, strf_a.size ), np.reshape( strf_b, strf_b.size ) )[0,1]
        r_dist_null[ii] = np.corrcoef( np.reshape( strf_a_null, strf_a_null.size ), np.reshape( strf_b_null, strf_b_null.size ) )[0,1]
    
    # STRF reliability
    r_mean = np.nanmean(r_dist)           
    r_mean_null = np.nanmean(r_dist_null) 
    r_zscore = ( r_mean - r_mean_null ) / np.nanstd( r_dist_null )
    p_val = (np.nonzero( r_dist_null >= r_mean )[0].size) / n_iter 
    # p_val = (np.nonzero(((r_dist-r_dist_null)<=0) )[0].size) / n_iter # alt method 
    
    if opt_plot:
        fig, ax = plt.subplots()
        ax.hist( r_dist, bins=np.arange(-0.5,1.05,0.025), color='r', alpha = 0.5, label='data' )
        ax.hist( r_dist_null, bins=np.arange(-0.5,1.05,0.025), color='k', alpha = 0.5, label='null' )
        plt.ylim(plt.ylim())
        ax.plot([r_mean, r_mean], ax.get_ylim() , 'r:')
        ax.plot([r_mean_null, r_mean_null], ax.get_ylim() , 'k:')
        plt.xlabel('Subsampled STRF correlations')
        plt.ylabel('Count')
        ax.legend()

    return r_mean, r_zscore, p_val


def get_strf_response_prediction( strf, stim, pspk_x_hz, x_bin_edges_raw ):
    
    '''
    
    Predicted response to novel stimulus given by STRF filter. 

    INPUT -------
    strf : [Filter] - Spectrotemporal receptive field, n freq x n time bins (numpy array) 
    stim : [Stimulus] - Binned stimulus matrices, n freq bins x m time bins x k stimuli (numpy array) 
    pspk_x_hz : IO function, spikes/s, from get_empirical_nonlinearity
    x_bin_edges_raw : histogram bin edges for pspk_x_hz (non-standardized), from get_empirical_nonlinearity 
    
    RETURN -------
    pred : predicted response, spikes/s, i.e., STRF-Stim projections multiplied by IO spike rate (pspk_x_hz), x m time bins x k stimuli (numpy array)

    '''

    assert stim.shape[0] == strf.shape[0]

    if len( stim.shape ) == 2:
        stim = stim.reshape( stim.shape + (1,) )

    pred = np.zeros( (stim.shape[-1],stim.shape[1]), dtype=np.float64)
    for ii in range(stim.shape[-1]):
        stimulus = stim[:,:,ii]
        stimulus = stimulus-stimulus.mean() # Mean subtraction 
        xprior_i, xposterior_i = get_strf_stimulus_projection( strf, stimulus, np.zeros( stim.shape[1], dtype=np.float64) ) # We don't need xposterior here, so use zero vector as dummy 'response' 
        idx = np.digitize( xprior_i, x_bin_edges_raw, right=True )
        idx[idx > pspk_x_hz.size] = pspk_x_hz.size ### Round down values > max(x_bin_edges_raw)  
        pred[ii,:] = pspk_x_hz[idx-1]
        
    pred = np.squeeze(pred)
        
    return pred 


def get_strf_response_prediction_data( strf, stim, bsc, mean_hz, pspk_x_hz, x_bin_edges_raw, opt_plot=False, opts_strf_calc=False ):
    
    '''

    Get response-prediction correlation coefficient.

    INPUT -------
    strf : [Filter] Spectro-temporal receptive field aka stimulus filter, n freq x n time bins (numpy array) 
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    mean_hz : Mean firing rate (spikes/s), from get_empirical_nonlinearity
    pspk_x_hz : IO function, spikes/s, from get_empirical_nonlinearity
    x_bin_edges_raw : histogram bin edges for pspk_x_hz (non-standardized), from get_empirical_nonlinearity 
    opt_plot : optional plot of example trial prediction + response (true/false)
    opts_strf_calc : options for STRF calc, default if not provided

    RETURN -------
    cc : Prediction-response correlation coefficient

    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: strf.shape
        Out: (32, 40) 
        In: stim.shape
        Out: (32, 3000, 40)
        In: bsc.shape
        Out: (40, 3000)

    '''

    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]
    assert stim.shape[0] == strf.shape[0]
    
    # Unpack calc opts
    if opts_strf_calc == False:
        opts_strf_calc = {}
        opts_strf_calc['strf_temp_res'] = 0.005 # sec
        opts_strf_calc['n_bins_pre'] = 40
        opts_strf_calc['n_bins_post'] = 0
        opts_strf_calc['n_bins_filt'] = 3
    
    strf_temp_res = opts_strf_calc['strf_temp_res']
    n_bins_pre = opts_strf_calc['n_bins_pre']
    n_bins_post = opts_strf_calc['n_bins_post']
    n_bins_filt = opts_strf_calc['n_bins_filt'] # must be odd integer
    
    n_bins_info_hist = 10
    n_bins_smooth_pred = 51
      
    # Get response prediction
    pred = get_strf_response_prediction( strf, stim, pspk_x_hz, x_bin_edges_raw )
    
    # Correlation between prediction and spike density function - - - - - - - - - 
    # Define x as smoothed prediction (firing rate ratio relative to mean)
    x = np.zeros( pred.shape, dtype=np.float64 )
    for ii in range(pred.shape[0]):
        x[ii,:] = filt_1d( pred[ii,:], n_bins_smooth_pred )
    x = x[:,n_bins_pre-1:-1] # Drop time at beginning < STRF time bins

    # Define y as smoothed spike density function (firing rate ratio relative to mean)
    y = np.zeros( bsc.shape, dtype=np.float64 )
    for ii in range(y.shape[0]):
        y[ii,:] = filt_1d( bsc[ii,:]/strf_temp_res, n_bins_smooth_pred )
    y = y[:,n_bins_pre-1:-1] # Drop time at beginning < STRF time bins
    
    x = np.reshape( x, x.size )
    y = np.reshape( y, y.size )
    cc = np.corrcoef( x, y )[0,1]

    if opt_plot: 
        
        pred = get_strf_response_prediction( strf, stim, pspk_x_hz, x_bin_edges_raw )

        # Which trial to plot
        ii = 0   
        
        # Time range within trial to plot
        idx_a = 0
        idx_b = pred.shape[1]
        idx_plot = np.arange(idx_a,idx_b)
        
        # Pred/resp data and correlation for example trial 
        bsc_i = bsc[ii,:]
        # bin_i[bin_i>1] = 1 # force binary response 0/1
        pred_smooth = filt_1d( pred[ii,:], n_bins_smooth_pred,'gaussian' )
        resp_smooth = filt_1d( bsc[ii,:]/strf_temp_res, n_bins_smooth_pred,'gaussian'  )
        taxis = np.arange( 0, strf_temp_res* bsc.shape[1], strf_temp_res )
        r = np.corrcoef( pred_smooth, resp_smooth )[0,1]

        # Plot 
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches((8,4))

        j = 0
        axs[j].bar( taxis[idx_plot], bsc_i[idx_plot], width = strf_temp_res, color='k', label='Response' )
        # axs[j].legend()
        axs[j].set_xlabel('Time (s)')
        axs[j].set_ylabel('Binned spike count')
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)
            
        j = 1
        axs[j].plot( taxis[idx_plot], resp_smooth[idx_plot], color=(0.5, 0.5, 0.5), label='Response')
        # axs[j].legend()
        axs[j].set_xlabel('Time (s)')
        axs[j].set_ylabel('Observed spk/s')
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)

        j = 2
        axs[j].plot( taxis[idx_plot], pred_smooth[idx_plot], color='r', label='Prediction')
        # axs[j].legend()
        axs[j].set_xlabel('Time (s)')
        axs[j].set_ylabel('Predicted spk/s')
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)
        
        for jj in range(axs.size):
            axs[jj].set_xlim(( min(taxis[idx_plot]), max(taxis[idx_plot])))
                
        fig.suptitle( f'r = %0.4f' % r, fontweight='bold' )
        fig.tight_layout()
        plt.show()

    return cc  


def get_strf_sig( strf, strf_null, pval = 0.01, opt_plot=False ):
    
    '''
    
    Version of STRF limited to significant time-frequency bins.
    
    INPUT -------
    strf : stimulus filter, n freq x n time bins (may be raw or sig version of STRF)
    strf_null : strf calculated using time-reversed stim (time-frequency bin values expected by chance)
    pval : Significance level for defining significant regions of STRF

    RETURN -------
    strf_sig : strf with values below chance cutoff set to zero  
    
    '''
    
    strf_sig_cutoff = np.percentile( abs(strf_null.reshape(strf_null.size)), 100-(pval/2*100) )
    strf_sig = np.zeros(strf.shape,dtype=np.float64)
    strf_sig[:] = strf[:]
    strf_sig[ abs(strf_sig) < strf_sig_cutoff ] = 0
    
    if opt_plot: 
        fig, ax = plt.subplots()
        im = ax.imshow( strf_sig, cmap='RdBu_r', origin='lower', aspect='auto')
        # im = ax.imshow( strf, cmap='RdBu_r', origin='lower')
        # ct = ax.contour( strf, strf_sig_cutoff, colors='k',extent=extent)
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time bin')
        plt.ylabel('Frequency bin')
        m = np.max(np.abs(strf))
        im.set_clim(-m,m)        
        
    return strf_sig 


def get_strf_stimulus_projection( strf, stimulus, binned_spike_counts ):
    
    '''
        
    INPUT -------
    strf : stimulus filter, n freq x n time bins 
    binned_spike_counts : spike counts binned in n time bins
    stimulus : stimulus matrix, n time bins x m freq bins
    
    RETURN -------
    xprior : prior distribution, n(x) - the result of STRF-Stim convolution
    xposterior : posterior distribution n(x|spike) - the xprior values associated with a spike
    
    '''

    assert(strf.shape[0] == stimulus.shape[0])
    
    nc = strf.shape[1] # time bins
    xprior = np.concatenate(( np.zeros(nc-1), sps.convolve2d( stimulus, np.rot90(strf,2), 'valid').squeeze() ))
    xposterior = np.zeros(0)
    if sum(binned_spike_counts)>0:
        for ii in np.arange(1,max(binned_spike_counts)+1):
            xposterior = np.concatenate( (xposterior, np.tile(xprior[binned_spike_counts==ii],int(ii)) ) )
    
    return xprior, xposterior


def get_strf_stimulus_projections( strf, stim, bsc ):
    
    '''
    
    Get projection value (convolution) distributions.  

    INPUT -------
    strf : stimulus filter, n freq x n time bins (numpy array) 
    stim : [Stimulus] - Binned stimulus matrices, 3-D array of n freq bins x m time bins x k stimuli (numpy array). 
    bsc : [Response] - Binned spike count matrix, n trials x k time points (numpy array) 
    
    RETURN -------
    xprior : prior distribution, n(x) - the result of STRF-Stim convolution
    xposterior : posterior distribution n(x|spike) - the xprior values associated with a spike 
    
    Note: assertions in place to ensure that the number of time bins and stimuli (trials) are equal for stim and bsc. 
    E.g., For an experiment with 40 stimuli, 3000 time bins, and 32 freq bins:
        In: strf.shape
        Out: (32, 40) 
        
        In: stim.shape
        Out: (32, 3000, 40)
        
        In: bsc.shape
        Out: (40, 3000)
    
    '''
    
    assert stim.shape[-1] == bsc.shape[0]
    assert stim.shape[1] == bsc.shape[1]
    assert stim.shape[0] == strf.shape[0]

    xprior = np.array([])
    xposterior = np.array([])
    for ii in range(bsc.shape[0]):
        stimulus = stim[:,:,ii]
        stimulus = stimulus-stimulus.mean() # Mean subtraction 
        xprior_i, xposterior_i = get_strf_stimulus_projection( strf, stimulus, bsc[ii,:] )
        xprior = np.concatenate( (xprior, xprior_i) )
        xposterior = np.concatenate( (xposterior, xposterior_i) )
        
    return xprior, xposterior


def get_waveform_peak_trough_info( waveform, fs=3e4 ):
    
    '''
    
    Get trough-peak delay and peak-trough ratio from waveform. 
            
    INPUT -------
    waveform : Neuron waveform (single channel), usually averaged across many instances 
    fs: sample rate at which neuron waveform was acquired

    RETURN -------
    tp_delay: trough-to-peak time, sec 
    pt_ratio: peak / trough ratio, absolute valued
        
    '''
    
    idx_trough = np.argmin( waveform )
    if idx_trough < waveform.size-1:
        idx_peak = np.argmax( waveform[idx_trough:-1] ) + idx_trough
        tp_delay = (idx_peak - idx_trough) / fs
        pt_ratio = waveform[idx_peak] / abs( waveform[idx_trough] )
    else: 
        tp_delay = float('nan')
        pt_ratio = float('nan')
        
    return tp_delay, pt_ratio

def plot_bsc( bsc, bin_edges ):
    
    '''
    
    Get binned spike count matrix from spike and event times.
    
    INPUT -------
    bsc : Binned spike count matrix, n trials x k time points (numpy array) 
    bin_edges : time axis for bsc

                                                         
    '''
        
    fig, ax = plt.subplots()
    im = ax.imshow( bsc, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(bin_edges), max(bin_edges), 0, bsc.shape[0] ] )
    fig.colorbar(im, ax=ax)
    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
        
        
def plot_fra( fra, faxis, aaxis ):
    
    '''
    
    Plot frequency-response area function. 
            
    INPUT -------
    fra : Frequency-Response Area function, i.e., mean firing rate evoked by tones of each freq-atten combination
    faxis: frequency axis of FRA in kHz
    aaxis: attenuation axis of FRA in dB
        
    '''    
    
    fig, ax = plt.subplots()
    im = ax.imshow( fra, cmap='inferno_r', aspect='auto', extent=[ min(faxis), max(faxis), max(aaxis), min(aaxis) ] )
    fig.colorbar(im, ax=ax)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Attenuation (dB)')    


def plot_nonlinearity( x_bin_centers, px, px_spk, pspk_x, pspk_x_hz, mean_hz ):
    
    '''
    
    Plot empirical nonlinearity including stimulus and spike probability distributions. 
    See get_empirical_nonlinearity
            
    INPUT -------
    px : standardized prior distribution probability, p(x) 
    px_spk : standardized posterior distribution probability, p(x|spike)  
    pspk_x : IO function, spiking probability p(spike|x)
    pspk_x_hz : IO function, spikes/s
    mean_hz : Mean firing rate (spikes/s)
    x_bin_centers : histogram bin centers for all output functions (standardized)

    ''' 

    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot( x_bin_centers, px, color=(0.5, 0.5, 0.5), marker='o', label='p(x)')
    axs[0].plot( x_bin_centers, px_spk, color='k', marker='o', label='p(x|spk)')
    axs[0].plot( x_bin_centers, pspk_x, color='r', marker='o', label='p(spk|x)')  
    axs[0].plot([0, 0], axs[0].get_ylim() , 'k:')
    axs[0].legend()
    axs[0].set_xlabel('STRF-stim projection (SD)')
    axs[0].set_ylabel('Probability')
    
    axs[1].plot( x_bin_centers, pspk_x_hz, color='r', marker='o')        
    axs[1].plot([0, 0], axs[1].get_ylim() , 'k:')
    axs[1].plot(axs[1].get_xlim(), [mean_hz, mean_hz], 'r:')
    axs[1].set_xlabel('STRF-stim projection (SD)')
    axs[1].set_ylabel('Firing rate (Hz)')  
              
    fig.tight_layout()
    plt.show()


def plot_strf( strf ):
    
    '''
    
    Plot spectro-temporal receptive field. 
            
    INPUT -------
    strf : spectro-temporal receptive field, n freq x n time bins. 

        
    '''    
    
    fig, ax = plt.subplots()
    im = ax.imshow( strf, cmap='RdBu_r', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Time bin')
    plt.ylabel('Frequency bin')
    m = np.max(np.abs(strf))
    im.set_clim(-m,m)

