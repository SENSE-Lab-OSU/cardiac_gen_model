# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:52:19 2021

@author: agarwal.270a
"""

#%%
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from math import gcd
import neurokit2 as nk
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float32')

from CardioGen.lib.simulator_for_CC import Simulator
from CardioGen.HR2Rpeaks import HR2Rpeaks_Simulator
from CardioGen.Rpeaks2EcgPpg import Rpeaks2EcgPpg_Simulator
from CardioGen.lib.utils import filtr_HR
from CardioGen.lib.data import load_data_wesad as load_data
n_classes=load_data.n_classes

#sys.path.append("../data/post-training/")
#from lib.sim_for_model_4 import HR_func_generator

ver=11 #version of the model_weights to use. Refer to README for details.
#%%

class ECG_HRV_Morph_Modulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,P_ID_out,path='../data/post-training/',
                 latent_size_HRV=5,latent_size_Morph=2,Fs_HR=100,
                 Fs_pks=100,Fs_out=None):
        
        super().__init__()
        self.Fs_HR=Fs_HR
        self.Fs_pks=Fs_pks
        self.Fs_out=Fs_out
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_HRV=latent_size_HRV
        self.latent_size_Morph=latent_size_Morph
        #with open(path+"WESAD_musig_v{}".format(ver), 'r') as f:
        with open(path+"WESAD_musig_v{}.pickle".format(ver), 'rb') as handle:
            self.dict_musig = pickle.load(handle)
            self.dict_musig=self.dict_musig[self.P_ID_out]
            
        self.sim_HR2pks=HR2Rpeaks_Simulator(HR_win_len_s=8,path=path,
                        P_ID='W',Fs_HR=Fs_HR,latent_size=latent_size_HRV)
        
        #Create Simulator using train data
        self.sim_pks2sigs=Rpeaks2EcgPpg_Simulator(P_ID=P_ID_out,path=path,
                                                latent_size=latent_size_Morph)

        
    def pre_process(self,ecg,Fs_in,win_len_s=8,step_s=2):
        #Clean using neurokit2
        ecg=nk.ecg_clean(ecg.flatten(),sampling_rate=Fs_in,method="neurokit")

        Fs_out=self.Fs_pks
        if Fs_in!=Fs_out:
            divisor=gcd(Fs_in,Fs_out)
            up,down=int(Fs_out/divisor),int(Fs_in/divisor)
            ecg_resamp=load_data.resample(ecg,Fs_in,up,down)
        else:
            ecg_resamp=ecg
        
        # find arr_pks
        arr_pks, _=load_data.find_ecg_rpeaks(ecg_resamp,Fs_out)
        HR_interpol,t_stamps=load_data.Rpeak2HR(arr_pks,win_len_s,step_s,
                                                self.Fs_pks)
        arr_pks=arr_pks[t_stamps[0]:t_stamps[1]+1]
        lenth=len(HR_interpol)
        
        #Form class_signal from P_ID
        class_signal=np.zeros((lenth,load_data.n_classes))
        class_signal[:,load_data.class_ids[self.P_ID_out]]=1
        
        return [class_signal.astype(np.float32),
                HR_interpol.reshape(-1,1).astype(np.float32),
                arr_pks.reshape(-1,1).astype(np.float32)]
        
    def __call__(self,ecg,Fs,show_plots=False):
        
        #Resample to internal Fs
        class_signal,HR,_=self.pre_process(ecg,Fs)
        
        #append class_signal to input condition
        HRV_cond=np.concatenate([HR,class_signal],axis=-1)
        #Morph_cond=np.concatenate([arr_pks,class_signal],axis=-1)
        
        #Get Rpeaks from HRV module
        arr_pks,HR_curve=self.sim_HR2pks(HRV_cond,Fs_out=self.Fs_pks)
        
        #Get ECG from Morph module
        Morph_cond=np.concatenate([arr_pks.reshape(-1,1),
                                   class_signal[:len(arr_pks)]],axis=-1)
        synth_ecg,arr_pks_ecg,clip_ecg,_,_,_=self.sim_pks2sigs(Morph_cond,
                                                        sigs2return=['ECG'])
        if self.Fs_out is None:
            self.Fs_out=Fs
        
        synth_ecg_out=self.post_process(synth_ecg, Fs_in=self.Fs_pks, 
                                        Fs_out=self.Fs_out)
        
        # if show_plots:
        #     if self.Fs_out==Fs:
        #         _,synth_HR,synth_arr_pks=self.pre_process(synth_ecg_out,self.Fs_out)
        #         start,end=clip_ecg
        #         HR_list=[HR[start:end],synth_HR]
        #         arr_pks_list=[arr_pks[start:end],synth_arr_pks[:,0]]
        #         ecg_list=[ecg,synth_ecg_out]
        #         self.plot_signals(HR_list,arr_pks_list,ecg_list,clip_ecg)
        #     else:
        #         print("Currently, Fs_in == Fs_out is needed for plotting." 
        #               "Skipping...")
                
        return synth_ecg_out,[synth_ecg,arr_pks_ecg]
    
    def post_process(self,synth_ecg_out,Fs_in,Fs_out):
        #Remove normalization
        synth_ecg_out*=self.dict_musig['ecg']['sig'] #rescale
        synth_ecg_out+=self.dict_musig['ecg']['mu'] #add back mean
        
        if Fs_in!=Fs_out:
            divisor=gcd(Fs_out,Fs_in)
            up,down=int(Fs_out/divisor),int(Fs_in/divisor)
            synth_ecg_out=load_data.resample(synth_ecg_out,Fs_in,up,down)
            
        return synth_ecg_out
    
    def plot_signals(self,HR_list,arr_pks_list,ecg_list,clip_ecg,n_plots=3):
        #Visualize when using GT data
        ecg_in,synth_ecg_out=ecg_list
        factr=self.Fs_out/self.Fs_pks
        if factr%1!=0:
            print('Currently, Fs_out must be a multiple of Fs_pks for'
                  ' plotting. Skipping...')
            return
        time_vec_out=np.arange(len(synth_ecg_out))/self.Fs_out
        time_vec_pks=time_vec_out[::int(factr)]
        start,end=clip_ecg
        start,end=int(factr)*start,int(factr)*end
        plot_cntr=1
        
        # Plot HR
        HR_in,synth_HR_out=HR_list
        fig1=plt.figure()
        ax=plt.subplot(n_plots,1,plot_cntr)
        plt.plot(time_vec_pks,HR_in,time_vec_pks,synth_HR_out,'r--')
        plt.grid(True)
        plt.title('HR')
        plt.legend(['Mod_in','Mod_out'])
        #plt.xlabel('Time (s)')
        plt.ylabel('HR (BPM)')
        plot_cntr+=1
        
        # Plot arr_pks
        arr_pks_in,synth_arr_pks_out=arr_pks_list
        #fig2=plt.figure()
        plt.subplot(n_plots,1,plot_cntr,sharex=ax)
        plt.plot(time_vec_pks,arr_pks_in,time_vec_pks,synth_arr_pks_out,'r--')
        plt.legend(['Mod_in','Mod_out'])
        plt.title('Rpeak-Train')
        #plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plot_cntr+=1
        
        # Plot ecg
        plt.subplot(n_plots,1,plot_cntr,sharex=ax)
        plt.plot(time_vec_out,ecg_in[start:end],time_vec_out,synth_ecg_out,'r--')
        plt.legend(['Mod_in','Mod_out'])
        #plt.grid(True)
        plt.title('ECG')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plot_cntr+=1
        plt.tight_layout()
        return
    
    def analyse_signal(self,signal,Fs,rpeak_train=None,method='kalidas2017',
                       show_plots=True,title_hrv='',title_morph=''):
        if rpeak_train is None:
            cleaned = nk.ecg_clean(signal, sampling_rate=Fs, method=method)
            rpeak_train , rpeaks = nk.ecg_peaks(cleaned, sampling_rate=Fs, 
                                        method=method, correct_artifacts=True)
        else:
            rpeaks=np.arange(len(rpeak_train))[rpeak_train.astype(bool)]
            rpeaks={'ECG_R_Peaks':rpeaks}
            rpeak_train={'ECG_R_Peaks':rpeak_train}
            
        # #clip first 0.1 s if there is R-peak in it
        # delta=int(0.1*Fs)
        # print(rpeaks['ECG_R_Peaks'].shape,rpeak_train['ECG_R_Peaks'].shape)
        # if rpeaks['ECG_R_Peaks'][0]<=delta:
        #     rpeaks['ECG_R_Peaks']=rpeaks['ECG_R_Peaks'][1:]-delta
        #     rpeak_train['ECG_R_Peaks']=rpeak_train['ECG_R_Peaks'][delta:]
        # Visualize R-peaks in ECG signal
        #plot = nk.events_plot(rpeaks['ECG_R_Peaks'], signal)
        #plot.axes[0].grid(True)
        
        #HRV analysis
        hrv_features = nk.hrv(rpeak_train, sampling_rate=Fs, show=show_plots)
        if show_plots:
            plt.tight_layout()
            plt.grid(True)
            plt.suptitle(title_hrv)
        
        # Delineate the ECG signal and visualizing all peaks of ECG complexes
        signal_peak, morph_features = nk.ecg_delineate(signal, rpeaks, 
        sampling_rate=Fs, show=show_plots, show_type='peaks',method='peaks')
        if show_plots:
            plt.tight_layout()
            plt.title(title_morph)
        #Add r-peaks to the same dict
        morph_features['ECG_R_Peaks']=rpeaks['ECG_R_Peaks']
        
        return morph_features,hrv_features

class ECG_Morph_Modulator(ECG_HRV_Morph_Modulator):
    def __init__(self,P_ID_out,path='../data/post-training/',
                 latent_size_Morph=2,Fs_pks=100,Fs_out=None):
        #super().__init__()
        self.Fs_pks=Fs_pks
        self.P_ID_out=P_ID_out
        self.Fs_out=Fs_out
        self.path=path
        self.latent_size_Morph=latent_size_Morph
        
        with open(path+"WESAD_musig_v{}.pickle".format(ver), 'rb') as handle:
            self.dict_musig = pickle.load(handle)
            self.dict_musig=self.dict_musig[self.P_ID_out]
            
        
        #Create Simulator using train data
        self.sim_pks2sigs=Rpeaks2EcgPpg_Simulator(P_ID=P_ID_out,path=path,
                                                latent_size=latent_size_Morph)
    def __call__(self,ecg,Fs):
        
        #Resample to internal Fs
        class_signal,_,arr_pks=self.pre_process(ecg,Fs)
        
        #append class_signal to input condition
        #HRV_cond=np.concatenate([HR,class_signal],axis=-1)        
        
        #Get ECG from Morph module
        Morph_cond=np.concatenate([arr_pks,class_signal],axis=-1)
        synth_ecg,arr_pks_ecg,clip_ecg,_,_,_=self.sim_pks2sigs(Morph_cond,
                                                        sigs2return=['ECG'])
        if self.Fs_out is None:
            self.Fs_out=Fs
        
        synth_ecg_out=self.post_process(synth_ecg, Fs_in=self.Fs_pks, 
                                        Fs_out=self.Fs_out)
        
        return synth_ecg_out,[synth_ecg,arr_pks_ecg]

class ECG_HRV_Modulator(ECG_HRV_Morph_Modulator):
    def __init__(self,P_ID_in,P_ID_out,path='../data/post-training/',
                 latent_size_HRV=5,latent_size_Morph=2,Fs_HR=100,
                 Fs_pks=100,Fs_out=None):
        
        #super().__init__()
        self.Fs_HR=Fs_HR
        self.Fs_pks=Fs_pks
        self.Fs_out=Fs_out
        self.P_ID_in=P_ID_in
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_HRV=latent_size_HRV
        self.latent_size_Morph=latent_size_Morph
        with open(path+"WESAD_musig_v{}.pickle".format(ver), 'rb') as handle:
            self.dict_musig = pickle.load(handle)
            self.dict_musig=self.dict_musig[self.P_ID_out]
            
        # HR2Rpks_user2
        self.sim_HR2pks=HR2Rpeaks_Simulator(HR_win_len_s=8,path=path,
                        P_ID='W',Fs_HR=Fs_HR,latent_size=latent_size_HRV)
        
        # Rpks2Ecg_user1
        self.sim_pks2sigs=Rpeaks2EcgPpg_Simulator(P_ID=P_ID_in,path=path,
                                                latent_size=latent_size_Morph)
        return
    
    def __call__(self,ecg,Fs,show_plots=False):
        
        #Resample to internal Fs
        class_signal,HR,_=self.pre_process(ecg,Fs)
        
        #append class_signal to input condition
        HRV_cond=np.concatenate([HR,class_signal],axis=-1)
        #Morph_cond=np.concatenate([arr_pks,class_signal],axis=-1)
        
        #Get Rpeaks_user2 from HRV module
        arr_pks,HR_curve=self.sim_HR2pks(HRV_cond,Fs_out=self.Fs_pks)
        
        #Get ECGMorph_user1 from Morph module
        #Form class_signal from P_ID
        class_signal=np.zeros((len(arr_pks),load_data.n_classes))
        class_signal[:,load_data.class_ids[self.P_ID_in]]=1
        Morph_cond=np.concatenate([arr_pks.reshape(-1,1),class_signal],axis=-1)
        synth_ecg,arr_pks_ecg,clip_ecg,_,_,_=self.sim_pks2sigs(Morph_cond,
                                                        sigs2return=['ECG'])
        if self.Fs_out is None:
            self.Fs_out=Fs
        
        synth_ecg_out=self.post_process(synth_ecg, Fs_in=self.Fs_pks, 
                                        Fs_out=self.Fs_out)
        
        return synth_ecg_out,[synth_ecg,arr_pks_ecg]
#%% Sample Client
if __name__=='__main__':    
    P_ID_in, P_ID_out='S7','S15'
    
    #Get Data
    path='../data/pre-training/WESAD/'
    file_path=path+f'{P_ID_in}/{P_ID_in}.pkl'
    lenth=700000
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        ecg_in = data['signal']['chest']['ECG'][-lenth:].astype(np.float32)
        ecg_in=load_data.resample(ecg_in.reshape(-1,1),700,1,7)
    file_path=path+f'{P_ID_out}/{P_ID_out}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        ecg_out = data['signal']['chest']['ECG'][-lenth:].astype(np.float32)
        ecg_out=load_data.resample(ecg_out.reshape(-1,1),700,1,7)

    ckpt_path='../data/post-training/'
    Fs_in=100
    Fs_out=100 #Synthetic signal sampling freq
    
    # ECG HRV+Morph modulation
    hrv_morph_mod=ECG_HRV_Morph_Modulator(P_ID_out=P_ID_out,
                                    path=ckpt_path,
                                    Fs_out=Fs_out)
    # Produce synthetic from S15 to S15 itself to check performance of models
    ecg_hrv_morph_mod_check,_=hrv_morph_mod(ecg_out,Fs=Fs_in)
    # Produce synthetic from S7 to S15
    ecg_hrv_morph_mod,_=hrv_morph_mod(ecg_in,Fs=Fs_in)
    
    # Analyze HRV and Morphological properties
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_out.flatten()
                                ,Fs=Fs_in,title_hrv=P_ID_out+'_out',
                                title_morph=P_ID_out+'_out')
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_hrv_morph_mod_check
                                ,Fs=Fs_out,title_hrv=P_ID_out+'_hrv_morph_synth_check',
                                title_morph=P_ID_out+'_hrv_morph_synth_check')
    
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_in.flatten()
                                ,Fs=Fs_in,title_hrv=P_ID_in+'_in',
                                title_morph=P_ID_in+'_in')
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_hrv_morph_mod
                                ,Fs=Fs_out,title_hrv=P_ID_out+'_hrv_morph_synth',
                                title_morph=P_ID_out+'_hrv_morph_synth')
    
    # ECG Morph modulation
    morph_mod=ECG_Morph_Modulator(P_ID_out=P_ID_out,
                                  path=ckpt_path,Fs_out=Fs_out)
    ecg_morph_mod,_=morph_mod(ecg_in,Fs=Fs_in)
    
    morph_features,hrv_features= morph_mod.analyse_signal(ecg_morph_mod
                                ,Fs=Fs_out,title_hrv=P_ID_in+'_morph_synth',
                                title_morph=P_ID_out+'_morph_synth')
    
    # ECG HRV modulation
    hrv_mod=ECG_HRV_Modulator(P_ID_in=P_ID_in,P_ID_out=P_ID_out,
                                  path=ckpt_path,Fs_out=Fs_out)
    ecg_hrv_mod,_=hrv_mod(ecg_in,Fs=Fs_in)
    
    morph_features,hrv_features= hrv_mod.analyse_signal(ecg_hrv_mod
                                ,Fs=Fs_out,title_hrv=P_ID_out+'_hrv_synth',
                                title_morph=P_ID_in+'_hrv_synth')