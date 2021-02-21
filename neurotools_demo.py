# -*- coding: utf-8 -*-
"""

Demo of various neurotools functions.

@author: jamesbigelow at gmail dot com


"""

#### Import modules 
import os, glob
import numpy as np
import neurotools as nt
from scipy import io
import matplotlib.pyplot as plt


#%% PSTH - peristimulus time histogram 

'''

This section documents usage of several functions for calculating PSTHs and related analyses, 
including PSTH reliability, optimal binwidth estimation, lifetime sparseness, and binned spike count matrix calc.

In this example, the stimulus is a mouse vocalization repeated 100x.

'''

#### Load data needed for PSTH calcs 

fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_spike_times_2020-02-20_14-47-56_ch28.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_stim_times_2020-02-20_16-51-32.mat' ) )[0] # stim times

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 


#### Calc PSTH data 

# PSTH - - - - - - - - - - - - - - - - -
tref = on_times
t1, t2 = -0.2, 1.15 + 0.2 # 0.2 s before and after stim period
binwidth = 0.02
opt_plot = True
psth = nt.get_psth( tspk, tref, t1, t2, binwidth, opt_plot )

# BSC - - - - - - - - - - - - - - - - -
# If we want BSC (trial x spike count) as plotted by get_psth we can use:
bsc = nt.get_bsc( tspk, tref, t1, t2, binwidth, opt_plot )

# PSTH reliability - - - - - - - - - - - - - - - - - 
n_iter = 1000
r_mean, r_zscore, p_val = nt.get_psth_reliability( tspk, tref, t1, t2, binwidth, n_iter, opt_plot )

# PSTH optimal binwidth - - - - - - - - - - - - - - - - - 
binwidths = np.logspace(-4, 0, 25 )
binwidth_opt, r_zscore_smooth = nt.get_psth_opt_binwidth( tspk, tref, t1, t2, binwidths, n_iter, opt_plot )

# PSTH lifetime sparseness - - - - - - - - - - - - - - - - - 
sparseness = nt.get_psth_sparseness( psth )


#%% MPH - Modulation period histogram

'''

This section documents usage of several functions for calculating MPHs and related analyses, including MPH reliability. 
These functions assume temporally modulated stimuli with regular periods (e.g., 4 Hz) such as click trains and sinusoidal modulated sounds. 

In this example, the stimuli are sinusoidal amplidute modulated chords with a range of modulation rates and depths, each repeated 20x. 

'''

#### Load data needed for MPH calcs 

fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_spike_times_2020-02-20_14-47-56_ch28.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_stim_times_2020-02-20_16-40-42.mat' ) )[0] # stim times
fid_stim_vals = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_stim_vals_2020-02-20_16-40-42.mat' ) )[0] # stim values per trial

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 

# Get SAM rate & depth for each trial - - - - - - - - - - - - - - - - -
stim_info = io.loadmat( fid_stim_vals, squeeze_me=True )
stim_fids = stim_info['stim_fids']
trial_order = stim_info['trial_order']
sam_rate = np.empty( (trial_order.size) )
sam_depth = np.empty( (trial_order.size) )
for ii in range(trial_order.size):  
    stim_str = stim_fids[ int(trial_order[ii])-1 ]['name'].split('_')
    sam_rate[ii] = int( stim_str[2][4:] )
    sam_depth[ii] = int( stim_str[3][5:-4] )

# Drop mod depths <100 - - - - - - - - - - - - - - - - -
# Note: This expriment has 3 mod depths (25, 50, and 100%), and 6 mod rates (2, 4, 8, 16, 32, and 64 Hz)
# For this demo, we'll analyze responses to mod rate == 4 and depth rate == 100 
tref = on_times[ (sam_depth==100) & (sam_rate==4) ]

#### Calc MPH data for single rate/depth

# MPH - - - - - - - - - - - - - - - - -
t1, t2 = 0.2, 1.2 # Drop first 0.2 s, which is non-modulated 
mod_rate = 4
n_bins = 50 
opt_plot=True
mph = nt.get_mph( tspk, tref, t1, t2, mod_rate, n_bins, opt_plot )

# MPH reliability - - - - - - - - - - - - - - - - - 
n_iter = 1000 
r_mean, r_zscore, p_val = nt.get_mph_reliability( tspk, tref, t1, t2, mod_rate, n_bins, n_iter, opt_plot )

# Compare to PSTH - - - - - - - - - - - - - - - - -
t1, t2 = 0.2, 1.2 # 0.3 s before and after stim period
binwidth = 0.02
psth = nt.get_psth( tspk, tref, t1, t2, binwidth, opt_plot )


#### Calc MPH data across full set of rates x depths

tref = on_times
n_iter = 10 
r_mean, r_zscore, p_val = nt.get_mod_rate_depth_mph_data( tspk, tref, sam_rate, sam_depth, t1, t2, n_bins, n_iter )

# Plot reliability x mod rate x mod depth 
fig, ax = plt.subplots()
im = ax.imshow( r_mean, cmap='gray_r', origin='lower', aspect='auto', extent=[ 0, np.unique(sam_rate).size, 0, np.unique(sam_depth).size ] )
plt.xticks( np.arange(0.5,np.unique(sam_rate).size+0.5,1), np.unique(sam_rate))
plt.yticks( np.arange(0.5,np.unique(sam_depth).size+0.5,1), np.unique(sam_depth))
fig.colorbar(im, ax=ax)
plt.xlabel('Modulation rate (Hz)')
plt.ylabel('Modulation depth (%)')
        
        
# Compare to PSTH version - - - - - - - - - - - - - - - - -
rate_mean, r_mean_psth, r_zscore_psth, p_val_psth = nt.get_mod_rate_depth_psth_data( tspk, tref, sam_rate, sam_depth, t1, t2, binwidth, n_iter )

# Plot reliability x mod rate x mod depth 
fig, ax = plt.subplots()
im = ax.imshow( r_mean_psth, cmap='gray_r', origin='lower', aspect='auto', extent=[ 0, np.unique(sam_rate).size, 0, np.unique(sam_depth).size ] )
plt.xticks( np.arange(0.5,np.unique(sam_rate).size+0.5,1), np.unique(sam_rate))
plt.yticks( np.arange(0.5,np.unique(sam_depth).size+0.5,1), np.unique(sam_depth))
fig.colorbar(im, ax=ax)
plt.xlabel('Modulation rate (Hz)')
plt.ylabel('Modulation depth (%)')


#%% STRF - spectrotemporal receptive field

'''

This section documents usage of several functions for calculating STRFs and related analyses, 
including STRF reliability, significant STRF time-frequency bins, empirical nonlinearities (input/output functions), 
mutual information, and predicting responses to novel stimuli using STRFs and nonlinearities.

In this example, the stimuli are non-repeating segments (15 s) of a random double frequency sweep.   

'''

#### Load data needed for STRF calcs 

# Filenames for spike and stim data - - - - - - - - - - - - - - - - -
fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_1_spike_times_2019-08-14_18-11-44_unit74.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_1_stim_times_2019-08-14_18-32-18.mat' ) )[0] # stim times
fid_stim_vals = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_1_stim_vals_2019-08-14_18-32-18.mat' ) )[0] # stim values per trial
fid_stim_full = glob.glob( os.path.join( os.getcwd(), 'demo data', 'rds_stim_array_temp_5ms_freq_8oct.mat' ) )[0] # stimulus (time-frequency values)

# Load stim - - - - - - - - - - - - - - - - - 
stimulus_matrices = io.loadmat( fid_stim_full )['stim_array']

# STRF calc opts - - - - - - - - - - - - - - - - - 
strf_temp_res = 0.005 # sec
n_bins_pre = 20
n_bins_post = 0
n_bins_filt = 3
opts_strf_calc = {}
opts_strf_calc['strf_temp_res'] = strf_temp_res
opts_strf_calc['n_bins_pre'] = n_bins_pre
opts_strf_calc['n_bins_post'] = n_bins_post
opts_strf_calc['n_bins_filt'] = n_bins_filt
n_bins_info_hist = 10
taxis = np.linspace( 95, 0,  num=n_bins_pre )
faxis = np.logspace( np.log10(4e3), np.log10(64e3), stimulus_matrices.shape[0] )

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times_all = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 
off_times_all = on_times_all + 15 # constant stim duration for this experiment
if io.loadmat(fid_stim_vals)['stim_vals'].shape == (2,1): 
    stim_vals = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][0] )
    stim_condition = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][1][0] )
elif io.loadmat(fid_stim_vals)['stim_vals'].shape == (1,2): 
    stim_vals = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][0] )
    stim_condition = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][1] )

# Isolate trials from a single condition - - - - - - - - - - - - - - - - - 
# Note: This experiment has two conditions, 0 and 1, which both use the same STRF estimation stimulus (random double sweep [RDS]). 
# Condition 0: auditory stimulus alone (RDS)
# Condition 1: auditory + visual stimulus (RDS + contrast modulated visual noise)
# For this demo, we use auditory trials only 
which_idx = stim_condition == 0
on_times, off_times = on_times_all[which_idx], off_times_all[which_idx]  
stim = stimulus_matrices[:,:,stim_vals[which_idx]-1]
# Sort by stim val 
idx = np.argsort( stim_vals[which_idx]-1 )
on_times, off_times = on_times[idx], off_times[idx]            
stim = stim[:,:,idx]    


#### Calc STRF data 

# Calc STRF - - - - - - - - - - - - - - - - -
tref = on_times
opt_plot = True
opt_null = False
bsc = nt.get_bsc( tspk, on_times, 0, 15, binwidth=strf_temp_res )
bsc[:,0:n_bins_pre] = 0 # Drop onset tranient time to reduce bias
strf, n_spikes = nt.get_strf_hz( stim, bsc, opt_null, opt_plot, opts_strf_calc )

# We can also plot PSTH to get a sense for time-averaged firing rate changes
t1, t2 = -3, 18 # 3 s before and after stim period
binwidth = 0.25
opt_plot=True
psth = nt.get_psth( tspk, tref, t1, t2, binwidth, opt_plot )

# Estimate STRF reliability - - - - - - - - - - - - - - - - -
n_iter = 100
strf_reliability_mean, strf_reliability_zscore, strf_reliability_pval = nt.get_strf_reliability( stim, bsc, n_iter, opt_plot, opts_strf_calc )

# Calc null STRF - - - - - - - - - - - - - - - - -
opt_null = True
strf_null, n_spikes_null = nt.get_strf_hz( stim, bsc, opt_null, opt_plot, opts_strf_calc )

# Define significant STRF time-frequency bins - - - - - - - - - - - - - - - - -
pval = 0.001
strf_sig = nt.get_strf_sig( strf, strf_null, pval, opt_plot )


# Calc various STRF params - - - - - - - - - - - - - - - - -
# Note: the get_strf_params function expects the STRF input to be absolute-valued. 
# This is so that subfields can be characterized independently or together, e.g.,:
 
# Excitatory subfield
strf_exc = np.zeros(strf.shape,dtype=np.float64)
strf_exc[:] = strf_sig[:]
strf_exc[strf_exc<0]= 0
d_params_exc = nt.get_strf_params( strf_exc, taxis, faxis )

# Inhibitory subfield
strf_inh = np.zeros(strf.shape,dtype=np.float64)
strf_inh[:] = strf_sig[:]
strf_inh[strf_inh>0] = 0 
strf_inh = abs(strf_inh)
d_params_inh = nt.get_strf_params( strf_inh, taxis, faxis ) 

# Both subfields together
strf_abs = np.zeros(strf.shape,dtype=np.float64)
strf_abs[:] = abs( strf_sig[:] )
d_params_abs = nt.get_strf_params( strf_abs, taxis, faxis ) 

# Excitation-Inhibition balance 
ei_balance = nt.get_strf_ei_balance( strf_sig )
    

#### Calc response nonlinearity

# Use 'sig' STRF for nonlinearity and mutual info calcs - - - - - - - - - - - - - - - - -
d_nonlinearity = nt.get_nonlinearity_data( strf_sig, stim, bsc )

# Plot nonlinearity - - - - - - - - - - - - - - - - -
nt.plot_nonlinearity( d_nonlinearity['x_bin_centers'], d_nonlinearity['px'], d_nonlinearity['px_spk'], d_nonlinearity['pspk_x'], d_nonlinearity['pspk_x_hz'], d_nonlinearity['mean_hz'] )

        
#### Predict response to novel stimulus 

# (1/2) Predict single trial --------------------------------

# Use first n-1 trials to fit STRF and nonlinearity, predict final trial - - - - - - - - - - - - - - - - -   
bsc_train = bsc[0:-1,:]
stim_train = stim[:,:,0:-1]
bsc_test = bsc[-1,:]
stim_test = stim[:,:,-1]

# Calc STRF and nonlinearity from training set - - - - - - - - - - - - - - - - - 
strf_train, n_spikes_train = nt.get_strf_hz( stim_train, bsc_train )
d_nonlinearity_train = nt.get_nonlinearity_data( strf_train, stim_train, bsc_train )

# Use STRF and nonlinearity to predict response to test trial - - - - - - - - - - - - - - - - - 
prediction = nt.get_strf_response_prediction( strf_train, stim_test, d_nonlinearity_train['pspk_x_hz'], d_nonlinearity_train['x_bin_edges_raw'] )

# Smooth pred/resp and calc correlation - - - - - - - - - - - - - - - - - 
# Note: since we're predicting a single trial using a non-repeated stimulus, we use a fairly wide smoothing kernel to capture increases and decreases in firing rate within the smoothing time window
pred_smooth = nt.filt_1d( prediction, 51,'gaussian' )
resp_smooth = nt.filt_1d( bsc_test/strf_temp_res, 51,'gaussian'  )
axis_time = np.arange( 0, strf_temp_res * prediction.size, strf_temp_res )
r = np.corrcoef( pred_smooth, resp_smooth )[0,1]

# Plot prediction/response - - - - - - - - - - - - - - - - - 
fig, axs = plt.subplots(3, 1)
fig.set_size_inches((12,5))
# Raw response (binned spike count vector)
axs[0].bar( axis_time, bsc_test, width = strf_temp_res, color='k', label='Response' )
axs[0].set_ylabel('Spike count')
# Smoothed response
axs[1].plot( axis_time, resp_smooth, color=(0.5, 0.5, 0.5), label='Response')
axs[1].legend()
axs[1].set_ylabel('Observed spk/s')
# Smoothed prediction
axs[2].plot( axis_time, pred_smooth, color='r', label='Prediction')
axs[2].legend()
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Predicted spk/s')
fig.suptitle( f'Prediction/response correlation: r = %0.4f' % r, fontweight='bold' )
fig.tight_layout()
        

# (2/2) Estimate prediction accuracy across trials by iteratively resampling training/test data sets --------------------------------

n_trials = bsc.shape[0]
n_iter_train = 10 # number of training/test data split iterations
p_train = 0.75 # proportion of data set to use for model training (STRF, nonlinearity)
pred_corrs = np.zeros(n_iter_train,dtype=np.float64) # initialize array for storing prediction correlations x iter 
opt_plot = False
for kk in range(n_iter_train):

    # Define train/test sets for stim and response - - - - - - - - - - - - - - - - - 
    idx_train = np.random.choice( n_trials, int(np.round(p_train * n_trials)), replace=False )
    idx_test = np.setdiff1d( np.arange(0,n_trials), idx_train )        
    bsc_train = bsc[idx_train,:]
    stim_train = stim[:,:,idx_train]
    bsc_test = bsc[idx_test,:]
    stim_test = stim[:,:,idx_test]
    
    # Calc STRF and nonlinearity from training set - - - - - - - - - - - - - - - - - 
    strf_train, n_spikes_train = nt.get_strf_hz( stim_train, bsc_train )
    d_nonlinearity_train = nt.get_nonlinearity_data( strf_train, stim_train, bsc_train )
    
    # Use STRF and nonlinearity to predict response to test set - - - - - - - - - - - - - - - - - 
    pred_corrs[kk] = nt.get_strf_response_prediction_data( strf_train, stim_test, bsc_test, d_nonlinearity_train['mean_hz'], d_nonlinearity_train['pspk_x_hz'], d_nonlinearity_train['x_bin_edges_raw'], opt_plot, opts_strf_calc )

pred_corr = np.nanmean( pred_corrs ) # estimate predicition accuracy as mean across iterations



#%% Difference STRF

'''

This section documents usage of functions for calculating STRFs in two conditions and estimating difference STRF reliability. 

In this example, the stimuli are non-repeating segments (15 s) of a random double frequency sweep.   

'''

#### Load data needed for STRF calcs 

# Filenames for spike and stim data - - - - - - - - - - - - - - - - -
fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_2_spike_times_2020-05-08_10-38-17_unit40.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_2_stim_times_2020-05-08_11-40-03.mat' ) )[0] # stim times
fid_stim_vals = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_2_stim_vals_2020-05-08_11-40-03.mat' ) )[0] # stim values per trial
fid_stim_full = glob.glob( os.path.join( os.getcwd(), 'demo data', 'rds_stim_array_temp_5ms_freq_8oct.mat' ) )[0] # stimulus (time-frequency values)

# Load stim - - - - - - - - - - - - - - - - - 
stimulus_matrices = io.loadmat( fid_stim_full )['stim_array']

# STRF calc opts - - - - - - - - - - - - - - - - - 
strf_temp_res = 0.005 # sec
n_bins_pre = 20
n_bins_post = 0
n_bins_filt = 3
opts_strf_calc = {}
opts_strf_calc['strf_temp_res'] = strf_temp_res
opts_strf_calc['n_bins_pre'] = n_bins_pre
opts_strf_calc['n_bins_post'] = n_bins_post
opts_strf_calc['n_bins_filt'] = n_bins_filt
n_bins_info_hist = 10
taxis = np.linspace( 95, 0,  num=n_bins_pre )
faxis = np.logspace( np.log10(4e3), np.log10(64e3), stimulus_matrices.shape[0] )

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times_all = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 
off_times_all = on_times_all + 15 # constant stim duration for this experiment
if io.loadmat(fid_stim_vals)['stim_vals'].shape == (2,1): 
    stim_vals = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][0] )
    stim_condition = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][1][0] )
elif io.loadmat(fid_stim_vals)['stim_vals'].shape == (1,2): 
    stim_vals = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][0] )
    stim_condition = np.squeeze( io.loadmat(fid_stim_vals)['stim_vals'][0][1] )

# Separate by condition - - - - - - - - - - - - - - - - - 
# Note: This experiment has two conditions, 0 and 1, which both use the same STRF estimation stimulus (random double sweep [RDS]). 
# Condition 0: auditory stimulus alone (RDS) --------
which_idx = stim_condition == 0
on_times_0, off_times_0 = on_times_all[which_idx], off_times_all[which_idx]  
stim_0 = stimulus_matrices[:,:,stim_vals[which_idx]-1]
# Sort by stim val 
idx = np.argsort( stim_vals[which_idx]-1 )
on_times_0, off_times_0 = on_times_0[idx], off_times_0[idx]            
stim_0 = stim_0[:,:,idx]
# Condition 1: auditory + visual stimulus (RDS + contrast modulated visual noise) --------
which_idx = stim_condition == 1
on_times_1, off_times_1 = on_times_all[which_idx], off_times_all[which_idx]  
stim_1 = stimulus_matrices[:,:,stim_vals[which_idx]-1]
# Sort by stim val 
idx = np.argsort( stim_vals[which_idx]-1 )
on_times_1, off_times_1 = on_times_1[idx], off_times_1[idx]            
stim_1 = stim_1[:,:,idx]   


#### Calc STRF data 

# Calc STRFs for each condition - - - - - - - - - - - - - - - - -
opt_plot = False
opt_null = False
bsc_0 = nt.get_bsc( tspk, on_times_0, 0, 15, binwidth=strf_temp_res )
bsc_0[:,0:n_bins_pre] = 0 # Drop onset tranient time
strf_0, n_spikes_0 = nt.get_strf_hz( stim_0, bsc_0, opt_null, opt_plot, opts_strf_calc )

bsc_1 = nt.get_bsc( tspk, on_times_1, 0, 15, binwidth=strf_temp_res )
bsc_1[:,0:n_bins_pre] = 0 # Drop onset tranient time
strf_1, n_spikes_1 = nt.get_strf_hz( stim_1, bsc_1, opt_null, opt_plot, opts_strf_calc )

# Difference STRF is simply the difference between conditions:
strf_d = strf_1 - strf_0 

# Plot STRFs - - - - - - - - - - - - - - - - -
m = np.max( ( np.max(np.abs(strf_0)), np.max(np.abs(strf_1)) ) )
md = np.max(np.abs(strf_d))
opts = {'vmin': -m, 'vmax': m}
opts_d = {'vmin': -md, 'vmax': md}

fig, axs = plt.subplots( 1, 3 )
fig.set_size_inches((12,4))
       
im = axs[0].imshow( strf_1, cmap='RdBu_r', origin='lower', aspect='auto', **opts )
axs[0].set_xlabel('Time bin')
axs[0].set_ylabel('Frequency bin')
axs[0].title.set_text('Condition 1')
fig.colorbar(im, ax=axs[0])

im = axs[1].imshow( strf_0, cmap='RdBu_r', origin='lower', aspect='auto', **opts )
axs[1].set_xlabel('Time bin')
axs[1].set_ylabel('Frequency bin')
axs[1].title.set_text('Condition 0')
fig.colorbar(im, ax=axs[1])

im = axs[2].imshow( strf_d, cmap='PuOr_r', origin='lower', aspect='auto', **opts_d  )
axs[2].set_xlabel('Time bin')
axs[2].set_ylabel('Frequency bin')
axs[2].title.set_text('Difference (1-0)')
fig.colorbar(im, ax=axs[2])

fig.tight_layout()
plt.show()

# Estimate Difference STRF reliability - - - - - - - - - - - - - - - - -
n_iter = 100
opt_plot = True
strf_reliability_mean_d, strf_reliability_zscore_d, strf_reliability_pval_d = nt.get_strf_difference_reliability( stim_0, bsc_0, stim_1, bsc_1, n_iter, opt_plot, opts_strf_calc )


#%% FRA - frequency response area

'''

This section documents usage of functions for calculating FRA and FRA params. 

These functions assume stimuli are tone pips with variable frequencies and attenuations.

'''

#### Load data needed for FRA calc 

# Filenames for spike and stim data - - - - - - - - - - - - - - - - -
fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_3_spike_times_2019-12-18_15-21-22_ch18.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_3_stim_times_2019-12-18_15-52-07.mat' ) )[0] # stim times
fid_stim_vals = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_3_stim_vals_2019-12-18_15-52-07.mat' ) )[0] # stim values per trial

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 

# Get frequency and attenuation values for each trial - - - - - - - - - - - - - - - - -  
stim_info = io.loadmat( fid_stim_vals, squeeze_me=True )
stim_fids = stim_info['stim_fids']
trial_order = stim_info['trial_order']
freq = np.zeros( (trial_order.size) )
freq[:] = np.NaN
atten = np.zeros( (trial_order.size) )
atten[:] = np.NaN
for ii in range(trial_order.size):  
    stim_str = stim_fids[ int(trial_order[ii])-1 ]['name'].split('_')
    freq[ii] = float( stim_str[2] )
    atten[ii] = float( stim_str[4][:-4] )
freq = freq / 1e3 # express in kHz


#### Calc FRA

tref = on_times
t1 = 0
t2 = 0.3     
opt_plot = True
fra = nt.get_fra( tspk, tref, freq, atten, t1, t2, opt_plot )

# Estimate FRA reliability - - - - - - - - - - - - - - - - -
n_iter = 100
fra_reliability_mean, fra_reliability_zscore, fra_reliability_pval = nt.get_fra_reliability( tspk, tref, freq, atten, t1, t2, n_iter, opt_plot )

# Calc FRA params - - - - - - - - - - - - - - - - -  
faxis = np.unique( freq )   
aaxis = np.unique( atten )   
fra_params = nt.get_fra_params( fra, faxis, aaxis )

# Tone onset latency - - - - - - - - - - - - - - - - - 
t1 = -0.1
t2 = 0.3  
binwidth = 0.002
onset_latency, offset_latency, peak_latency = nt.get_psth_latency( tspk, tref, t1, t2, binwidth )


#%% CCG - crosscorrelogram

'''

This section demonstrates calculation of cross- and auto-correlograms.  


'''

#### Load spike times for each unit  

# Filenames for spike and stim data - - - - - - - - - - - - - - - - -
fid_tspk_a = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_4a_spike_times_2020-03-05_10-50-59_unit49.mat' ) )[0] # spike times for unit a
fid_tspk_b = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_4b_spike_times_2020-03-05_10-50-59_unit54.mat' ) )[0] # spike times for unit b

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk_a = io.loadmat( fid_tspk_a, squeeze_me=True )['spiketimes']
tspk_b = io.loadmat( fid_tspk_b, squeeze_me=True )['spiketimes']

#### Calc CCG
maxlag = 100 # number of bins
binwidth = 0.001 # seconds
bin_edges = np.arange( min( ( min(tspk_a), min(tspk_b) ) ), max( ( max(tspk_a), max(tspk_b) ) ), binwidth )
bsc_a, bin_edges = np.histogram( tspk_a, bins = bin_edges )
bsc_b, bin_edges = np.histogram( tspk_b, bins = bin_edges )
ccg = nt.get_ccg( bsc_a, bsc_b, maxlag )

# Plot CCG - - - - - - - - - - - - - - - - - 
fig, ax = plt.subplots()
bin_edges_ccg = np.arange( -maxlag*binwidth, maxlag*binwidth + binwidth, binwidth )
plt.plot( bin_edges_ccg, ccg, color='k')
plt.xlim((-maxlag*binwidth,maxlag*binwidth))
plt.ylim(plt.ylim())
plt.plot( [0,0], plt.ylim(), 'k:' )

# Calc and plot autocorrelogram - - - - - - - - - - - - - - - - - 
fig, ax = plt.subplots()
acg = nt.get_ccg( bsc_a, bsc_a, maxlag )
plt.plot( bin_edges_ccg, acg, color='k')
plt.xlim((-maxlag*binwidth,maxlag*binwidth))
plt.ylim(plt.ylim())
plt.plot( [0,0], plt.ylim(), 'k:' )


#%% Classifier analysis 

'''

This section documents usage of functions for classifying which stimulus among a set elicited a single-trial response. May be used with arbitrary stimuli. 

In this example, the stimuli are sinusoidal amplidute modulated chords with variable modulation rates and depths, each repeated 20x. 


'''

#### Load data needed for classifier analysis 

fid_tspk = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_spike_times_2020-02-20_14-47-56_ch28.mat' ) )[0] # spike times
fid_tref = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_stim_times_2020-02-20_16-40-42.mat' ) )[0] # stim times
fid_stim_vals = glob.glob( os.path.join( os.getcwd(), 'demo data', 'example_unit_5_stim_vals_2020-02-20_16-40-42.mat' ) )[0] # stim values per trial

# Load spike and stim data - - - - - - - - - - - - - - - - - 
tspk = io.loadmat( fid_tspk, squeeze_me=True )['spiketimes']
on_times = io.loadmat( fid_tref, squeeze_me=True )['on_times'] # stim onset times 

# Get SAM rate & depth for each trial - - - - - - - - - - - - - - - - -
stim_info = io.loadmat( fid_stim_vals, squeeze_me=True )
stim_fids = stim_info['stim_fids']
trial_order = stim_info['trial_order']
sam_rate = np.empty( (trial_order.size) )
sam_depth = np.empty( (trial_order.size) )
for ii in range(trial_order.size):  
    stim_str = stim_fids[ int(trial_order[ii])-1 ]['name'].split('_')
    sam_rate[ii] = int( stim_str[2][4:] )
    sam_depth[ii] = int( stim_str[3][5:-4] )

# Drop mod depths <100 - - - - - - - - - - - - - - - - -
# Note: This expriment has 4 mod depths (0, 25, 50, 100), and 6 mod rates (2 4 8 16 32 64)
# For this demo, we'll classify responses among the six mod rates at depth rate == 100 
tref = on_times[ sam_depth==100 ]
stim_vals = sam_rate[ sam_depth==100 ]
stim_vals_unique = np.unique( stim_vals )
n_stim = np.unique( stim_vals ).size

# First, visualize responses to each mod rate with PSTH function
t1, t2 = -0.3, 1.6 # 0.3 s before and after stim period
binwidth = 0.02
opt_plot = True
for ii in range(n_stim):
    psth = nt.get_psth( tspk, on_times[ (sam_depth==100) & (sam_rate==stim_vals_unique[ii]) ], t1, t2, binwidth, opt_plot )


#### Get classifier data  

# Define classifier analysis window - - - - - - - - - - - - - - - - - 
# Note: The stimuli used in this experiment have uniform non-modulated lead of 0.2 s across all stimuli, 
# to allow adaptation to stimulus onset. A 1-s modulated period follows, which is the focus of the experiment.  
t1 = 0.2
t2 = t1 + 1

# Classifier analysis - - - - - - - - - - - - - - - - - 
binwidth = 0.02 
opt_plot = True
accuracy, cm = nt.classify_responses( tspk, tref, stim_vals, t1, t2, binwidth, opt_plot )

# Classifier optimal binwidth - - - - - - - - - - - - - - - - - 
binwidths = np.logspace(-4, 0, 25 )
binwidth_opt, accuracy_smooth = nt.get_classifier_opt_binwidth( tspk, tref, stim_vals, t1, t2, binwidths, opt_plot )

# Mutual information from confusion matrix - - - - - - - - - - - - - - - - - 
info = nt.get_mutual_info_from_cm( cm )

# We can also estimate chance mutual info from random confusion matrix
n_trials = stim_vals.size
cm_rand = nt.get_cm_random( n_stim, n_trials )
info_rand = nt.get_mutual_info_from_cm( cm_rand )
