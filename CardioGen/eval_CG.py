# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:52:08 2022

@author: agarwal.270a
"""


import os
import sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.signal import detrend
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle
import glob
import pandas as pd
import copy
import neurokit2 as nk
#import seaborn as sns
import scipy as sp
import time
import math
#import matplotlib

#matplotlib.rc_file_defaults()
rng = default_rng(seed=1)
#gpus = tf.config.experimental.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(gpus[0], True)


from tensorflow.keras.activations import relu

import datetime

from lib.data import load_data_wesad_eval as load_data
from lib.data.load_data_wesad import Rpeaks2RRint, sliding_window_fragmentation
from lib.utils import set_plot_font


#from pathlib import Path
#curr_dir=Path.cwd()
#root_dir=curr_dir.parents[1]

import Augmentor_ecg_st_id as augmentor
data_path='../data/pre-training/WESAD/'
save_dir='../data/post-training/results/eval/h30_m28_dsamp2'
os.makedirs(save_dir,exist_ok=True)

augmentor.data_path=data_path
save_name='WESAD_synth_h30_m28/eval'
aug_suffix='s_c'
proj_path='.'
ver=12
win_len_analyse=32
Fs_ppg_new,Fs_ecg_new=25,100

#win_len_s,step_s,bsize=augmentor.win_len_s,augmentor.step_s,augmentor.bsize

tf.keras.backend.set_floatx('float32')
#plt.style.use('default')
#sns.set_style("paper")
#sns.set()
plt.rcParams['text.usetex'] = True

# stresses={k:v for v,k in 
#           enumerate(['trash','baseline', 'stress', 'amusement','meditation'])}
stresses=['trash','baseline', 'stress', 'amusement','meditation']



#tf.keras.backend.clear_session()
#TODO: Added to run only on CPU when needed
#tf.config.set_visible_devices([], 'GPU')
#%% Helper functions for loading data


def Rpeak2avgHR(arr_pks,Fs=Fs_ecg_new):
    RR_ints_NU, _=Rpeaks2RRint(arr_pks.flatten(),Fs)
    return np.mean(RR_ints_NU**-1)

def Rpeak2avgRR(arr_pks,Fs=Fs_ecg_new):
    RR_ints_NU, _=Rpeaks2RRint(arr_pks.flatten(),Fs)
    return np.mean(RR_ints_NU)

def seq_format_function_eval(avghrv2ecg_aug,cond_HRV,arr_tacho_synth,
                            arr_pk_synth,ecg_synth):
    #stress,class,smooth_tacho,tacho
    seq_hrv=np.stack([np.argmax(cond_HRV[:,1:6],axis=-1),
                     np.argmax(cond_HRV[:,6:21],axis=-1),
                     cond_HRV[:,0],arr_tacho_synth],axis=-1)
    #pks,ecg
    seq_morph=np.stack([arr_pk_synth,ecg_synth[:,0]],axis=-1)
    
    # hrv_wins.shape=[N,160,4]
    hrv_wins=load_data.sliding_window_fragmentation([seq_hrv],
                    win_len_analyse*avghrv2ecg_aug.Fs_tacho,
                    win_len_analyse*avghrv2ecg_aug.Fs_tacho)
    # morph_wins.shape=[N,3200,2]
    morph_wins=load_data.sliding_window_fragmentation([seq_morph],
                    win_len_analyse*avghrv2ecg_aug.Fs_out,
                    win_len_analyse*avghrv2ecg_aug.Fs_out)
    return hrv_wins,morph_wins

def get_drop_mask(arr_out):
    # drop all trash classes
    drop_mask=arr_out[:,0].astype(bool) 
    print(np.mean(drop_mask))
    #TODO: Commented out this clipping as 0.5%ile extremeties will handle it
    # # drop all out-of-range HR windows
    # drop_mask=(((arr_out[:,2]>=np.min(HR_bins)) & 
    #             (arr_out[:,2]<=np.max(HR_bins))) & drop_mask)
    # print(np.mean(drop_mask))
    # # drop all out-of-range RR windows
    # drop_mask=(((arr_out[:,3]>=np.min(RR_bins)) & 
    #             (arr_out[:,3]<=np.max(RR_bins))) & drop_mask)
    # print(np.mean(drop_mask))
    return drop_mask

def get_real_stress_weights(arr_out):
    '''
    

    Parameters
    ----------
    arr_out : np.float of shape [N,2]

    Returns
    -------
    real_stress_weights : dict(class:lists) with list len=5

    '''
    real_stress_weights={k:[] for k in list(all_class_ids.keys())[:]}
    #real_stress_weights={}
    for class_name in list(load_data.class_ids.keys())[:]:
        id_mask=(arr_out[:,1]==all_class_ids[class_name])
        SC_ratio_list=[]
        for k in range(len(stresses)):
            stress_mask=((arr_out[:,0]==k) & id_mask)
            
            SC_ratio=100*(np.sum(stress_mask)/np.sum(id_mask))
            SC_ratio_list.append(SC_ratio)
        SC_ratios=np.array(SC_ratio_list)
        SC_ratios/=np.max(SC_ratios) #normalize such that baseline stress is 1
        real_stress_weights[class_name]=SC_ratios
    return real_stress_weights

def stress_weighted_dsample(arr_out,real_stress_weights):
    '''
    Parameters
    ----------
    hrv_wins : np.float array of shape [N,160,4]
    morph_wins : np.float array [N,3200,2]
    real_stress_weights : dict(class:lists) with list len=5

    Returns
    -------
    hrv_wins : np.float array [M,160,4]
    morph_wins : np.float array [M,3200,2]
    idxs_dsamp : np.int array of dasmpled indices
    M<=N
    '''
    
    idxs=np.arange(len(arr_out))
    idxs_dsamp_list=[]
    for class_name in list(load_data.class_ids.keys())[:]:
        id_mask=(arr_out[:,1]==all_class_ids[class_name])
        rsw=real_stress_weights[class_name]
        for k in range(len(stresses))[:]:
            stress_mask=((arr_out[:,0]==k) & id_mask)
            
            idxs_SC=idxs[stress_mask]
            n_select=int(rsw[k]*len(idxs_SC))
            idxs_SC_dsamp=rng.permutation(idxs_SC)[:n_select]
            idxs_dsamp_list.append(idxs_SC_dsamp)
    
    idxs_dsamp=np.concatenate(idxs_dsamp_list)
    return idxs_dsamp

def load_data_helper(Dsplit_mask_dict,musig_dict,data_path,augmentor,
                     seq_format_function_eval,
                     dsampling_factor_aug=1):
    '''
    

    Parameters
    ----------
    Dsplit_mask_dict : TYPE
        DESCRIPTION.
    musig_dict : TYPE
        DESCRIPTION.
    data_path : TYPE
        DESCRIPTION.
    augmentor : TYPE
        DESCRIPTION.
    seq_format_function_eval : TYPE
        DESCRIPTION.
    dsampling_factor_aug : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    real_data : list(np.arrays)
        real data [(arr_pks,ecg),(st,id,HR)] with shapes [[N,Fs_ecg*T,2],[N,3]]
    synth_data : list(np.arrays)
        synth data [(arr_pks,ecg),(st,id,HR)] with shapes 
        [[N,Fs_ecg*T,2],[N,3]]
    gen_ability_data : list(np.arrays)
        synth data [hrv_wins,morph_wins]=[(smooth_tacho,tacho),(arr_pks,ecg)] 
        with shapes [[N,Fs_tacho*T,2],[N,Fs_ecg*T,2]]
        
        T,Fs_tacho,Fs_ecg=32,5,100 => Fs_tacho*T=160 & Fs_ecg*T=3200
    '''

    # load_data.class_ids={k:all_class_ids[k] 
    #                       for k in list(all_class_ids.keys())[10:12]}
    #input=[pks,ecg].shape=[N,3200,2];output=[st,id].shape=[N,2]
    input_list,output_list,_=load_data.get_train_data(data_path,
                                    win_len_s=win_len_analyse,
                                    step_s=win_len_analyse,
                                    Dsplit_mask_dict=Dsplit_mask_dict,
                                    musig_dict=musig_dict,mode='eval')
    arr_in=np.concatenate(input_list,axis=0)
    arr_out=np.concatenate(output_list,axis=0)
    
    mean_HR=[Rpeak2avgHR(pks,Fs=Fs_ecg_new) for pks in arr_in[:,:,0]]
    mean_HR=np.expand_dims(np.stack(mean_HR,axis=0),axis=-1)
    mean_RR=[Rpeak2avgRR(pks,Fs=Fs_ecg_new) for pks in arr_in[:,:,0]]
    mean_RR=np.expand_dims(np.stack(mean_RR,axis=0),axis=-1)
    arr_out=np.concatenate([arr_out,mean_HR,mean_RR],axis=1)
    
    #remove undesired windows
    drop_mask=get_drop_mask(arr_out)#, HR_bins, RR_bins)
    arr_in,arr_out=arr_in[drop_mask],arr_out[drop_mask]
    real_data=[arr_in,arr_out]
    
    #  Get synthetic Data, [N,160,4], [N,3200,2]
    hrv_wins,morph_wins=augmentor.main(seq_format_function_eval,
                        save_name=f'{save_name}',
                        show_plots=False,suffix=aug_suffix)
    
    #At first Naive dsampling choosing every other sample
    hrv_wins=hrv_wins[::dsampling_factor_aug]
    morph_wins=morph_wins[::dsampling_factor_aug]
    
    mean_HR_synth=[Rpeak2avgHR(pks,Fs=Fs_ecg_new) for pks in morph_wins[:,:,0]]
    mean_HR_synth=np.expand_dims(np.stack(mean_HR_synth,axis=0),axis=-1)
    mean_RR_synth=[Rpeak2avgRR(pks,Fs=Fs_ecg_new) for pks in morph_wins[:,:,0]]
    mean_RR_synth=np.expand_dims(np.stack(mean_RR_synth,axis=0),axis=-1)
    
    arr_out_synth=np.concatenate([hrv_wins[:,0,0:2],mean_HR_synth,
                                  mean_RR_synth],axis=1)

    #remove undesired windows
    drop_mask=get_drop_mask(arr_out_synth)#, HR_bins, RR_bins)
    arr_out_synth=arr_out_synth[drop_mask]
    morph_wins,hrv_wins=morph_wins[drop_mask],hrv_wins[drop_mask]
    
    
    #form all gen_ability data before next stress dsampling
    gen_ability_data=[hrv_wins[:,:,2:4],morph_wins]
    
    #stress weighted dsampling for feat data
    synth_stress_weights=get_real_stress_weights(arr_out_synth)#to verify later
    real_stress_weights=get_real_stress_weights(arr_out)
    idxs_dsamp=stress_weighted_dsample(arr_out_synth,real_stress_weights)
    arr_out_synth=arr_out_synth[idxs_dsamp]
    hrv_wins,morph_wins=hrv_wins[idxs_dsamp],morph_wins[idxs_dsamp]
    
    synth_stress_weights=get_real_stress_weights(arr_out_synth)#verify
    print(real_stress_weights,synth_stress_weights)

    synth_data=[morph_wins,arr_out_synth]
    
    return real_data,synth_data,gen_ability_data,idxs_dsamp

def pks2tacho(arr_pks,arr_pks_ecg,fs):
    #arr_pks,arr_pks_ecg=arr_pks_in[delta:-delta],arr_pks_ecg[delta:-delta]
    RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks,Fs_pks)
    # Uniformly interpolate
    t=np.cumsum(RR_ints_NU)+(RR_extreme_idx[0]/Fs_pks)
    f_interpol = sp.interpolate.interp1d(t, RR_ints_NU,'cubic',axis=0)
    t_start,t_end=t[0],t[-1] #helps get integers on uniform grid
    
    RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks_ecg,Fs_pks)
    # Uniformly interpolate
    t=np.cumsum(RR_ints_NU)+(RR_extreme_idx[0]/Fs_pks)
    f_interpol_ecg = sp.interpolate.interp1d(t, RR_ints_NU,'cubic',axis=0)
    t_start,t_end=max(t_start,t[0]),min(t_end,t[-1]) #helps get integers on uniform grid
    t_start,t_end=np.ceil(t_start),np.floor(t_end)#helps resolve floating pt err
    
    t_interpol = np.arange(t_start,t_end, 1/fs)
    #Had to add explicit clipping due to floating point errors here.
    #t_interpol[-1]=min(t[-1],t_interpol[-1])
    #t_interpol[0]=max(t[0],t_interpol[0])
    nn_interpol = f_interpol(t_interpol)
    nn_interpol_ecg=f_interpol_ecg(t_interpol)

    #print(len(RR_ints))
    #Had to reduce filter order due to smaller windows. Reasonable results
    try:
        tacho=load_data.tacho_filter(nn_interpol, fs,f_cutoff=0.5,
                                     order=40-1,show_plots=False,margin=0.15)
        tacho_ecg=load_data.tacho_filter(nn_interpol_ecg,fs,f_cutoff=0.5,
                                     order=40-1,show_plots=False,margin=0.15)
    except ValueError:
        tacho=load_data.tacho_filter(nn_interpol, fs,f_cutoff=0.5,
                                     order=30-1,show_plots=False,margin=0.15)
        tacho_ecg=load_data.tacho_filter(nn_interpol_ecg,fs,f_cutoff=0.5,
                                     order=30-1,show_plots=False,margin=0.15)
    return tacho,tacho_ecg

def set_yticks(n_ticks=4,margin=0.2,precision_up=1,positive=False):
    ylocs, ylabels = plt.yticks()
    factr=ylocs[1]-ylocs[0]
    ylocs/=factr
    mn,mx=np.min(ylocs)*(1-margin),np.max(ylocs)*(1+margin)
    if positive: mn=max(0,mn)
    ylocs_new=np.round(np.linspace(mn,mx,n_ticks),precision_up)*factr#.astype(int)
    if ylocs_new[-1]%1.==0: ylocs_new=ylocs_new.astype(int) #clip decimal if ints
    prec_str=np.abs(min(0,int(f'{factr:e}'.split('e')[-1])))
    plt.yticks(ylocs_new,[f'{b:.{prec_str}f}' for b in ylocs_new])
#%% Main functions to get analysis data

#Get Conditional Generation Ability error data
def get_CGA_err_data(gen_ability_data,Fs_tacho):
    #[[N,160,2],[N,3200,2]]
    hrv_wins,morph_wins=gen_ability_data
    
    #get tacho from r-peaks and ECG peaks to compare
    delta=50 #helps clip noisy peak detection at the terminals
    avgHRV_wins=hrv_wins[:,:,0]
    tacho_wins=hrv_wins[:,:,1]
    pks_wins,ecg_wins=morph_wins[:,:,0],morph_wins[:,:,1]
    hrv_err,morph_err=np.zeros(len(ecg_wins)),np.zeros(len(ecg_wins))
    start_time=time.time()
    for i in range(len(ecg_wins)):
        #get smooth HR from tacho and compare
        avgHRV_in,tacho=avgHRV_wins[i,:],tacho_wins[i,:]
        #Had to reduce filter order due to smaller windows. Reasonable results.
        avgHRV_tacho=(load_data.tacho_filter(tacho.reshape(-1,1),Fs_tacho,
                        f_cutoff=0.125,
                        order=50-1,show_plots=False,margin=0.075)).flatten()
        #plt.figure();plt.plot(avgHRV_in);plt.plot(avgHRV_tacho)
        #hrv_err[i]=rmse(60*avgHRV_in**-1,60*avgHRV_tacho**-1)
        hrv_err[i]=rmse(1000*avgHRV_in,1000*avgHRV_tacho) #changed to ms.
        
        arr_pks_in,ecg=pks_wins[i,:],ecg_wins[i,:]
        arr_pks_ecg, _=load_data.find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        #plt.figure();plt.plot(arr_pks_in);plt.plot(arr_pks_ecg);plt.plot(ecg)
        
        tacho_in,tacho_ecg=pks2tacho(arr_pks_in[delta:-delta],
                                     arr_pks_ecg[delta:-delta],fs=Fs_tacho)
        #plt.figure();plt.plot(tacho_in);plt.plot(tacho_ecg)
        morph_err[i]=rmse(1000*tacho_in,1000*tacho_ecg) #changed to ms.
        
        if (i%100)==0:
            print(f'elapsed time {time.time()-start_time} for {100*i/len(ecg_wins)}%')
            
    return hrv_err,morph_err


# Get hrv and morph features data
def get_feat_data(real_data,synth_data,hrv_foi,units_foi):
    #Find common HR bins
    
    # all_HR=np.concatenate([real_data[1][:,2],synth_data[1][:,2]])
    # mn,mx=np.min(all_HR),np.max(all_HR)
    # bins = np.linspace(mn, mx, n_bins + 1, endpoint=True)
    # adj = (mx - mn) * 0.01  # 0.1% of the range
    # bins[0] -= adj
    # bins[-1] += adj
    # bins=np.round(bins,2)
        
    #HRV analysis using arr_pks
    #TODO: May need to edit these nk functions to directly input rri/tacho
    
    #define HRV params
    hrv_feat_dict={'Real':{k:[] for k in list(all_class_ids.keys())[:]},
                   'Synthetic':{k:[] for k in list(all_class_ids.keys())[:]}}
    # all_data=[real_data,synth_data]
    # all_data_labels=list(hrv_feat_dict.keys())
    
    #define Morph params
    #skip_pks=1 #MUST be int>0
    T_steps=real_data[0].shape[1]
    possible_pk_idxs=np.arange(T_steps)
    
    morph_feat_dict={'Real':{k:[] for k in list(all_class_ids.keys())[:]},
                   'Synthetic':{k:[] for k in list(all_class_ids.keys())[:]}}
    all_data=[real_data,synth_data]
    all_data_labels=list(morph_feat_dict.keys())
    
    for class_name in list(all_class_ids.keys())[:]:
        #print(f'processing class={class_name}\n')
        #bins=n_bins
        #HR_list=[]
        #QT_list=[]
        #all_HR=np.concatenate([real_data[1][:,2],synth_data[1][:,2]])
        for i in range(len(all_data))[:]:
            print(f'processing class={class_name} {all_data_labels[i]} data\n')
            data=all_data[i]
            #hrv_feat_dict_data=hrv_feat_dict[all_data_labels[i]]
            id_mask=(data[1][:,1]==all_class_ids[class_name])
            #stress_mask=(data[1][:,0]==stresses[k])
            data4class=[data[0][id_mask],data[1][id_mask]]
    
            #get hrv features of interest
            #TODO: Changed min_len requirement in hrv_freq-->sig_power-->sig_pow_instant and suppressed warn in ...-->signal_psd
            hrv_features_list=[nk.hrv(data4class[0][j,:,0],sampling_rate=Fs_ecg,
                                      show=False)[hrv_foi] 
                               for j in range(len(data4class[0]))]
            hrv_feat_dict[all_data_labels[i]][class_name]=pd.concat(
                                            hrv_features_list,ignore_index=True)
            #HR_list.append(data4class[1][:,2])
            
            #get morph features of interest
            pk_idxs_list=[(possible_pk_idxs[data4class[0][j,:,0].astype(bool)
                                                  ][1:-1]+j*T_steps) 
                                for j in range(len(data4class[0]))]
            #Match RR_ints_list and pk_idxs_list aptly
            # RR_ints_list=[(np.diff(pk_idx)[:-1]/Fs_ecg_new) 
            #               for pk_idx in pk_idxs_list]
            RR_ints_list=[(np.diff(pk_idx)[1:]/Fs_ecg_new) 
                          for pk_idx in pk_idxs_list]
            pk_idxs_list=[pk_idx[1:-1] for pk_idx in pk_idxs_list]
            RR_ints=np.concatenate(RR_ints_list,axis=0)
            pk_idxs=np.concatenate(pk_idxs_list,axis=0)
            
            #pk_idxs=sum(pk_idxs_list,[])
            ecg_unroll=data4class[0][:,:,1].flatten()
            
    
            #for class_name in list(all_class_ids.keys())[:]:
            #Check ecg morphs
            signal_peak, morph_features = nk.ecg_delineate(ecg_unroll[:], 
                                    pk_idxs, sampling_rate=Fs_ecg_new,
                                    show=False,method='peaks', show_type='peaks')
            
            #plt.title(class_name)
            morph_features['RR_ints']=RR_ints
            morph_features['QT']=(np.array(morph_features['ECG_T_Peaks'])-
                      np.array(morph_features['ECG_Q_Peaks']))/Fs_ecg_new
            
            morph_feat_dict[all_data_labels[i]][class_name]=morph_features
            #QT_list.append(morph_features['QT'])
            
            # if (i%100)==0:
            #     print(f'elapsed time {time.time()-start_time} for {100*i/len(ecg_wins)}%')
            
    return hrv_feat_dict,morph_feat_dict

#%% Initialize
Fs_pks=Fs_ecg_new
Fs_tacho=5
Fs_ecg=100
#n_bins=10
save_data=True
rmse=lambda y,y_hat:np.sqrt(np.mean((y.reshape(-1)-y_hat.reshape(-1))**2))
all_class_ids=copy.deepcopy(load_data.class_ids)
all_class_names=list(all_class_ids.keys())

#create meaningful and realisitc equi-spaced bins
# HR_bins=np.arange(0.5,4+0.35,0.35)
# RR_bins=np.arange(0.25,2+0.175,0.175)
# mn=min(HR_bins[0],RR_bins[-1]**-1)
# mx=max(HR_bins[-1],RR_bins[0]**-1)

hrv_foi=['HRV_RMSSD','HRV_SDNN','HRV_LF','HRV_HF','HRV_SD1','HRV_SD2']
units_foi=2*['$ms$.']+2*['$ms^2$.']+2*['$Ms$.']
#hrv_err_new,morph_err_new=copy.deepcopy(hrv_err),copy.deepcopy(morph_err)

#%% Load Aug data
#class_name_list=['S7'
#load_data.class_ids={class_name:all_class_ids[class_name]}
# load_data.class_ids={k:all_class_ids[k] 
#                       for k in list(all_class_ids.keys())[10:11]}
# Load Dsplit_mask
Dsplit_filename = (f'{proj_path}/../data/pre-training/'
            f'WESAD_musig_Dsplit_w{augmentor.win_len_s}s{augmentor.step_s}'
            f'b{augmentor.bsize}.pickle')
if os.path.isfile(Dsplit_filename):
    with open (Dsplit_filename, 'rb') as fp:
        musig_dict,Dsplit_mask_dict = pickle.load(fp)
else:
    assert False, ('Could not find existing Dsplit_mask_dict. '
                    'Run get_train_data in R2S mode first.')
    
d_list=load_data_helper(Dsplit_mask_dict,musig_dict,data_path,augmentor,
                        seq_format_function_eval,
                        dsampling_factor_aug=2)


real_data,synth_data,gen_ability_data,idxs_dsamp=d_list


#Drop outlier data and form qbins
all_RR=np.concatenate([real_data[1][:,3],synth_data[1][:,3]],axis=0)
#Remove w% from begining and end to account for outliers
w=0.5
q_w=np.round(np.percentile(all_RR,w),2)
q_nw=np.round(np.percentile(all_RR,(100-w)),2)
all_RR=all_RR[((all_RR>q_w) & (all_RR<q_nw))]
_,new_RR_bins=pd.cut(all_RR,10,labels=False,retbins=True,precision=2)
#round down, round up and round
new_RR_bins[0]=np.floor(new_RR_bins[0]*1e2)/1e2
new_RR_bins[-1]=np.ceil(new_RR_bins[-1]*1e2)/1e2
new_RR_bins=np.round(new_RR_bins,2)
RR_bins=new_RR_bins
filename = f'{save_dir}/extracted_data_RR({np.min(RR_bins):.1f},{np.max(RR_bins):.1f}).pickle'


#Get err data from disc or produce new
if os.path.isfile(filename):
    with open (filename, 'rb') as fp:
       idxs_synth,hrv_err,morph_err,hrv_feat_dict,morph_feat_dict = pickle.load(fp)
       assert (np.mean(idxs_dsamp==idxs_synth)==1),'existing idxs != saved idxs'
       
else:
    #Get CGA err data
    hrv_err,morph_err=get_CGA_err_data(gen_ability_data,Fs_tacho)
    
    hrv_feat_dict,morph_feat_dict=get_feat_data(real_data,synth_data,
                                                    hrv_foi,units_foi)
    if save_data:
        # Save data
        with open(filename, 'wb') as handle:
            pickle.dump([idxs_dsamp,hrv_err,morph_err,hrv_feat_dict,
                         morph_feat_dict], handle)

all_data=[real_data,synth_data]
all_data_labels=list(morph_feat_dict.keys())
#%% Create CGA plots

#hrv_err,morph_err=1000*(hrv_err/60)**(-1),1000*(morph_err/60)**(-1)
#hrv_err,morph_err=hrv_err_new,morph_err_new

#hrv_err,morph_err=hrv_err[plot_mask],morph_err[plot_mask]
_,morph_wins=gen_ability_data
# mean_HR_synth=[Rpeak2avgHR(pks,Fs=Fs_ecg_new) for pks in morph_wins[:,:,0]]
# mean_HR_synth=np.stack(mean_HR_synth,axis=0)
mean_RR_synth=[Rpeak2avgRR(pks,Fs=Fs_ecg_new) for pks in morph_wins[:,:,0]]
mean_RR_synth=np.stack(mean_RR_synth,axis=0)

#avoid Nan's by clipping first
sel_mask_bin=((mean_RR_synth>=RR_bins[0]) & (mean_RR_synth<=RR_bins[-1]))
mean_RR_synth=mean_RR_synth[sel_mask_bin]
hrv_err,morph_err=hrv_err[sel_mask_bin],morph_err[sel_mask_bin]

# bin data
cuts,bins=pd.cut(mean_RR_synth,RR_bins,labels=False,retbins=True,
               precision=2,include_lowest=True)
# cuts,bins=pd.cut(synth_data[1][:,2],10,labels=False,retbins=True,
#                precision=2)
#cuts=cuts[plot_mask]

assert ~np.isnan(cuts).any(), 'cuts has nan values, Check plot_mask'
#all_cuts[i]=cuts

bin_freq=np.array([100*(np.mean((cuts==n))) for n in range(len(bins)-1)])
bin_hrv_err_list=np.array([np.mean(hrv_err[(cuts==n)]) 
                           for n in range(len(bins)-1)])
bin_morph_err_list=np.array([np.mean(morph_err[(cuts==n)])
                             for n in range(len(bins)-1)])
bin_centers=1000*((bins[:-1] + bins[1:]) / 2) #scaling s-->ms

set_plot_font(6,9,10)

fig=plt.figure()

ax1=plt.subplot(311)
plt.plot(bin_centers,bin_hrv_err_list,'b-o')
plt.ylabel('HRV RMSE ($ms$.)')
plt.grid(True)
plt.margins(y=0.2)
plt.ylim(bottom=0)
#set_yticks(n_ticks=5,margin=0.2,precision_up=0,positive=True)

plt.subplot(312,sharex=ax1)
plt.plot(bin_centers,bin_morph_err_list,'r-o')
plt.ylabel('Morph RMSE ($ms$.)')
plt.grid(True)
#set_yticks(n_ticks=5,margin=0.2,precision_up=0,positive=True)
plt.margins(y=0.2)
plt.ylim(bottom=0)

plt.subplot(313,sharex=ax1)
plt.bar(bin_centers, bin_freq,width=1000*np.diff(bins),linewidth=0.5,
        align='center',color='g')
#plt.hist(bin_freq,bin_centers,color='g')
plt.ylabel("Bin Data ($\%$)")
plt.grid(True)
plt.xlabel('Average RR ($ms$.)')
plt.margins(y=0.2)
plt.ylim(bottom=0)
#set_yticks(n_ticks=5,margin=0.2,precision_up=0,positive=True)

# ax = plt.gca()    # Get current axis
# ax2 = ax.twinx()  # make twin axis based on x
# ax2.plot(bin_centers, bin_freq,'g--o',label='bin_data %')
# ax2.set_ylabel("Bin Data %")
# plt.legend(loc='upper right')

#plt.xticks(range(len(bin_centers)),[str(b) for b in bin_centers])
plt.xticks(1000*bins,[f'{1000*b:.0f}' for b in bins])
# Adjust spacings w.r.t. figsize
fig.tight_layout()
# drop_nan_mask=~np.isnan(bin_morph_err_list)
# plt.yticks(bin_morph_err_list[drop_nan_mask],
#            [f'{b:.2f}' for b in bin_morph_err_list[drop_nan_mask]])
#fig.tight_layout()
#plt.savefig(f'{save_dir}/gen_ability_eqbins.pdf')
plt.savefig(f'{save_dir}/gen_ability.png',dpi=300,bbox_inches="tight")

print(f'Overall mean of hrv_err={np.mean(hrv_err):.2f}, morph_err={np.mean(morph_err):.2f}')
#%% Find deviation in features classwise
bins_hrv=RR_bins*1
bins_morph=RR_bins*1
bin_hrv_feat_dict={'Real':{k:[] for k in list(all_class_ids.keys())[:]},
                   'Synthetic':{k:[] for k in list(all_class_ids.keys())[:]}}

for class_name in list(all_class_ids.keys())[:]:
    #bin data
    for i in range(len(all_data))[:]:
        data=all_data[i]
        id_mask=(data[1][:,1]==all_class_ids[class_name])
        #stress_mask=(data[1][:,0]==stresses[k])
        
        
        data4class=[data[0][id_mask],data[1][id_mask]]
        #avoid Nan's by clipping first
        sel_mask_bin=((data4class[1][:,3]>=RR_bins[0]) & 
                      (data4class[1][:,3]<=RR_bins[-1]))
        mean_RR_data=data4class[1][:,3][sel_mask_bin]
        
        #hrv_err,morph_err=hrv_err[sel_mask_bin],morph_err[sel_mask_bin]
        hrv_feat_dict[all_data_labels[i]][class_name]=(
                hrv_feat_dict[all_data_labels[i]][class_name][sel_mask_bin])
        
        #bin hrv data
        cuts=pd.cut(mean_RR_data,bins_hrv,labels=False,include_lowest=True)#,retbins=True)
        #data4class[1][np.arange(len(cuts))[np.isnan(cuts)],3]
        assert ~np.isnan(cuts).any(), 'cuts has nan values, Reconsider bins'
        #all_cuts[i]=cuts
        
        RMSSD=hrv_feat_dict[all_data_labels[i]][class_name]['HRV_RMSSD']
        
        
        bin_RMSSD=[np.sqrt(np.nanmean(RMSSD[(cuts==n)]**2)) 
                   for n in range(len(bins_hrv)-1)]
        bin_hrv_features_list=[(hrv_feat_dict[all_data_labels[i]][class_name]
                                [(cuts==n)]).mean(skipna=True) 
                               for n in range(len(bins_hrv)-1)]
        n_bin_hrv=np.array([np.sum(cuts==n) for n in range(len(bins_hrv)-1)])

        bin_hrv_features=pd.concat(bin_hrv_features_list,axis=1).T
        bin_hrv_features.index=1000*((bins_hrv[:-1] + bins_hrv[1:]) / 2)#s-->ms
        bin_hrv_features['HRV_RMSSD']=bin_RMSSD #replace with correct mean(RMSSD)
        bin_hrv_features['HRV_n_bin']=n_bin_hrv
        
        bin_hrv_feat_dict[all_data_labels[i]][class_name]=bin_hrv_features
        #bin_err[n]=((real_err-synth_err)/real_err)*100
        
        #bin morph data
        RR_ints=morph_feat_dict[all_data_labels[i]][class_name]['RR_ints']
        QT=1000*morph_feat_dict[all_data_labels[i]][class_name]['QT']#s-->ms
        
        #avoid Nan's by clipping first
        sel_mask_bin=((RR_ints>=RR_bins[0]) & (RR_ints<=RR_bins[-1]))
        RR_ints,QT=RR_ints[sel_mask_bin],QT[sel_mask_bin]
        
        cuts=pd.cut(RR_ints,bins_morph,labels=False,include_lowest=True)#,retbins=True)
        drop_nan_mask=~((np.isnan(cuts)) | (np.isnan(QT)))
        cuts,QT=cuts[drop_nan_mask],QT[drop_nan_mask]
        assert ~np.isnan(cuts).any(), 'cuts has nan values, Reconsider bins'
        #all_cuts[i]=cuts
        bin_QT=np.array([np.mean(QT[cuts==n]) for n in range(len(bins_morph)-1)])
        n_bin_QT=np.array([np.sum(cuts==n) for n in range(len(bins_morph)-1)])
        
        morph_feat_dict[all_data_labels[i]][class_name]['bin_QT']=bin_QT
        morph_feat_dict[all_data_labels[i]][class_name]['n_bin_QT']=n_bin_QT

        morph_feat_dict[all_data_labels[i]][class_name]['morph_bins']=1000*bins_morph
            

#%%


#make class-wise plots
plt.close('all')
#set_plot_font(8,9,10)
marker_list=['-o',':o']
#HRV plots
for k in [0,3]:
    feat=hrv_foi[k].split('_')[-1]
    unit=units_foi[k]
    
    for i in range(3):
        fig=plt.figure(i)
        
        for j in range(5):
            class_id=i*5+j
            class_name=all_class_names[class_id]
            if ((i+j)==0): ax1=plt.subplot(511+j)
            else: plt.subplot(511+j,sharex=ax1,sharey=ax1)
    
            for d in range(len(all_data))[:]:
                plt.plot(bin_hrv_feat_dict[all_data_labels[d]][class_name]
                         ['HRV_'+feat],marker_list[d],label=all_data_labels[d],
                         markersize=4)
            if j==0: plt.legend(loc='upper right')
            if j==2: plt.ylabel(f'{feat} ({unit})')
            if j==4: plt.xlabel('Average RR ($ms$.)')
            plt.title(class_name)
            plt.grid(True)
            
        # err=np.round(rmse(hrv_feat_dict['Real'][class_name][feat],
        #                   hrv_feat_dict['Synthetic'][class_name][feat]),2)
        # plt.suptitle(f'RMSE between Real and Synthetic {feat}={err} ({unit})')
        #width = 0.9 * (bins[1] - bins[0])
        #print(bin_err)
        fig.tight_layout()
    #ax1.margins(y=0.2)
    xlocs=(1000*bins_hrv).astype(int)
    plt.xticks(xlocs,[f'{b}' for b in xlocs])
    set_yticks(n_ticks=4,margin=0.2,positive=True)
    #plt.ylim(bottom=0)
        
    for i in range(3):
        plt.figure(i)
        #plt.savefig(f'{save_dir}/hrv_{feat}_{i}_eqbins.pdf')
        plt.savefig(f'{save_dir}/hrv_{feat}_{i}.png',dpi=300,
                    bbox_inches="tight")
    plt.close('all')
    
#morph_plots
plt.close('all')
feat='bin_QT'
unit='$ms$.'
for i in range(3):
    fig=plt.figure(i)
    
    for j in range(5):
        class_id=i*5+j
        class_name=all_class_names[class_id]
        if ((i+j)==0): ax1=plt.subplot(511+j)
        else: plt.subplot(511+j,sharex=ax1,sharey=ax1)
        QT_bins=morph_feat_dict['Real'][class_name]['morph_bins']
        bin_centers=(QT_bins[:-1] + QT_bins[1:]) / 2
        
        for d in range(len(all_data))[:]:
            plt.plot(bin_centers,
                     morph_feat_dict[all_data_labels[d]][class_name][feat],
                     marker_list[d],label=all_data_labels[d],markersize=4)
            
        if j==0: plt.legend(loc='upper right')
        if j==2: plt.ylabel('QT length ($ms$.)')
        if j==4: plt.xlabel('RR ($ms$.)')
        plt.title(class_name)
        plt.grid(True)
        
    # Adjust spacings w.r.t. figsize
    fig.tight_layout()
#ax1.margins(y=0.2)
xlocs=(1000*bins_morph).astype(int)
plt.xticks(xlocs,[f'{b}' for b in xlocs])
set_yticks(n_ticks=4,margin=0.2,positive=True)
#plt.ylim(bottom=0);

    
for i in range(3):
    fig=plt.figure(i)
    #plt.savefig(f'{save_dir}/morph_{feat}_{i}.pdf')
    plt.savefig(f'{save_dir}/morph_{feat}_{i}.png',dpi=300,bbox_inches="tight")
plt.close('all')
#%% Make unified plot
set_plot_font(8,9,10)

QT_bins=morph_feat_dict['Real']['S2']['morph_bins']
bin_centers=(QT_bins[:-1] + QT_bins[1:]) / 2
plt.figure()

for d in range(len(all_data)):
    all_n_bin_sum=np.zeros(len(QT_bins)-1)
    all_bin_QT_sum=np.zeros(len(QT_bins)-1)
    for class_name in list(all_class_ids.keys())[:]:
        bin_QT=morph_feat_dict[all_data_labels[d]][class_name]['bin_QT']
        n_bin_QT=morph_feat_dict[all_data_labels[d]][class_name]['n_bin_QT']
        for i in range(len(n_bin_QT)):
            QT_sum=n_bin_QT[i]*bin_QT[i]
            if ~np.isnan(QT_sum):
                all_n_bin_sum[i]+=n_bin_QT[i]
                all_bin_QT_sum[i]+=QT_sum
    all_bin_QT=all_bin_QT_sum/all_n_bin_sum
    bin_freq=100*(all_n_bin_sum/np.sum(all_n_bin_sum))
    
    ax1=plt.subplot(211)
    plt.plot(bin_centers,all_bin_QT,'-o',label=all_data_labels[d])
    if d==1:
        plt.legend(loc='upper right')
        plt.ylabel('QT length ($ms$.)')
        #plt.ylim(bottom=0);
        plt.grid(True)
        
    ax2=plt.subplot(212,sharex=ax1)
    plt.bar(bin_centers, bin_freq,width=((-1)**d*0.5*np.diff(QT_bins)),
            linewidth=0.5,align='edge',label=all_data_labels[d])


    if d==1:
        plt.legend(loc='upper right')
        plt.ylabel('Bin data ($\%$)')
        plt.xlabel('RR ($ms$.)')
        plt.suptitle('QT vs. RR plot for all data')
        plt.xticks(1000*bins_morph,[f'{1000*b:.0f}' for b in bins_morph])
        #plt.ylim(bottom=0)
        plt.grid(True)

# Adjust spacings w.r.t. figsize
fig.tight_layout()
#plt.savefig(f'{save_dir}/morph_{feat}_{i}.pdf')
plt.savefig(f'{save_dir}/morph_all_QT.png',dpi=300, bbox_inches="tight")

#Total mean error
#total_QT_err=np.sum(all_bin_QT_sum)/np.sum(all_n_bin_sum)
#print(all_bin_QT_sum/all_n_bin_sum)
#%% Overall errs

# for d in range(len(all_data)):
#     all_n_bin_sum=np.zeros(len(QT_bins)-1)
#     all_bin_QT_sum=np.zeros(len(QT_bins)-1)
#     for class_name in list(all_class_ids.keys())[:]:
#         bin_feat=bin_hrv_feat_dict[all_data_labels[d]][class_name]['HRV_'+feat]
#         n_bin_feat=bin_hrv_feat_dict[all_data_labels[d]][class_name]['HRV_n_bin']
#         for i in range(len(n_bin_feat)):
#             feat_sum=n_bin_feat[i]*bin_feat[i]
#             if ~np.isnan(QT_sum):
#                 all_n_bin_sum[i]+=n_bin_QT[i]
#                 all_bin_QT_sum[i]+=QT_sum
#     all_bin_QT=all_bin_QT_sum/all_n_bin_sum

# #%% Check some figs
# a1=real_data[0]#synth_data[0]#morph_wins
# idx_test=10
# signal_peak, morph_features = nk.ecg_delineate(a1[idx_test,:,1], 
#                         a1[idx_test,:,0], sampling_rate=Fs_ecg_new,
#                         show=True,method='peaks', show_type='peaks')
# plt.figure();plt.plot(a1[idx_test,:,0])
# plt.plot(a1[idx_test,:,1])

#%%
# bins_hrv=array([0.78904829, 1.03626535, 1.11177754, 1.16452889, 1.21694017,
#        1.26277714, 1.33428906, 1.40063765, 1.5047133 , 1.68033553,
#        3.33048631])

# bins_morph=array([0.04941, 0.109  , 0.168  , 0.227  , 0.286  , 0.345  , 0.404  ,
#        0.463  , 0.522  , 0.581  , 0.64   ])

# plt.figure()
# a#x1=plt.subplot(121)
# n, bins, patches = plt.hist(QT_real, 50, density=False, facecolor='b',
#                             alpha=0.75,histtype='step')
# #plt.subplot(122,sharex=ax1,sharey=ax1)
# n, bins, patches = plt.hist(QT_synth, bins, density=False, facecolor='g',
#                             alpha=0.75,histtype='step')
# plt.xlabel('Normalized QT length (s.)')
# plt.ylabel('No. of windows')