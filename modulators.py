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
from CardioGen.Rpeaks2Sig import Rpeaks2Sig_Simulator
from CardioGen.lib.utils import filtr_HR,get_uniform_tacho, get_continous_wins
from CardioGen.lib.data import load_data_wesad as load_data

#Define global constants
proj_path='.'
n_classes=load_data.n_classes
n_stresses=load_data.n_stresses
win_len_s=load_data.win_len_s
step_s=load_data.step_s
bsize=load_data.test_bsize
Fs_ppg=load_data.Fs_ppg_new
Fs_ecg=load_data.Fs_ecg_new

path=f'{proj_path}/data/pre-training/WESAD/'
ckpt_path=f'{proj_path}/data/post-training/'
ver=12 #version of the model_weights to use. Refer to README for details.
Dsplit_filename = (f'{proj_path}/data/pre-training/'
                   f'WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{bsize}.pickle')
if os.path.isfile(Dsplit_filename):
    with open (Dsplit_filename, 'rb') as fp:
        musig_dict,Dsplit_mask_dict = pickle.load(fp)
else:
    assert False, ('Could not find existing Dsplit_mask_dict. '
                   'Run get_train_data in R2S mode first.')
#%%

class ECG_HRV_Morph_Modulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,P_ID_out,path='data/post-training/',
                 latent_size_HRV=4,latent_size_Morph=2,Fs_tacho=5,Fs_out=100):
        
        super().__init__()
        self.Fs_tacho=Fs_tacho
        self.Fs_out=Fs_out
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_HRV=latent_size_HRV
        self.latent_size_Morph=latent_size_Morph
        self.musig_dict=musig_dict[self.P_ID_out]
        self.up_factor=int(self.Fs_out/self.Fs_tacho)
        #clipping to match R2S signals with ppg_ssqi_thresholding. Not essential
        clip_times=[1,-1]
        self.clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]

        P_ID_HRV='WESAD'
        P_ID_Morph=P_ID_out
        
        self.sim_HR2pks=HR2Rpeaks_Simulator(
                    RNN_win_len_s=win_len_s+(bsize-1)*step_s,
                    step_size_s=step_s,P_ID=P_ID_HRV,path=path,Fs_HR=Fs_tacho,
                    Fs_tacho=Fs_tacho,latent_size=latent_size_HRV)
        
        #Create a subject-wise Rpeaks2Sig_Simulator object 
        self.sim_pks2sigs=Rpeaks2Sig_Simulator(Fs_in=Fs_out,Fs_out=Fs_out,
                    P_ID=P_ID_Morph,path=path,sig_id='ecg',
                    latent_size=latent_size_Morph,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s)

        
    def pre_process(self,ecg,stres,Fs_in,win_len_s=8,step_s=2):
        Fs_ecg_wesad=load_data.Fs_ecg
        #Clean using neurokit2
        ecg=nk.ecg_clean(ecg.flatten(),sampling_rate=Fs_in,method="neurokit")
        stres[stres>=5] = 0. #zero-out any meaningless label >=5

        #Resample as needed
        Fs_tacho=self.Fs_tacho
        Fs_out=self.Fs_out
        if Fs_in!=Fs_out:
            divisor=gcd(Fs_in,Fs_out)
            up,down=int(Fs_out/divisor),int(Fs_in/divisor)
            ecg_resamp=load_data.resample(ecg.reshape(-1,1),Fs_in,up,down)
        else:
            ecg_resamp=ecg
        
        # find arr_pks

        arr_pks, test=load_data.find_ecg_rpeaks(ecg_resamp,Fs_out)
        arr_pks=self.clip_sig(arr_pks,Fs_out)
        stres=self.clip_sig(stres,Fs_ecg_wesad)


        # Find uniform Tachogram at Fs_tacho
        RR_ints_NU, RR_extreme_idx=load_data.Rpeaks2RRint(arr_pks,Fs_out)
        
        # Uniformly interpolate
        t_interpol,RR_ints=get_uniform_tacho(RR_ints_NU,fs=Fs_tacho,
                                             t_bias=RR_extreme_idx[0]/Fs_out)
        tacho=load_data.tacho_filter(RR_ints, Fs_tacho,show_plots=False)
        
        avgHRV_interpol=(load_data.tacho_filter(tacho,Fs_tacho,f_cutoff=0.075,
                                       show_plots=False)).flatten()
        #arr_pks=arr_pks[((t_pks>=t_interpol[0]) & (t_pks<=t_interpol[-1]))]
        arr_pks=arr_pks[int(t_interpol[0]*Fs_out):
                        int((t_interpol[-1]+(1/Fs_tacho))*Fs_out)]
        lenth=len(avgHRV_interpol)

        stres_signal_ecg,_ = load_data.create_stress_signal(stres,
                                        Fs=Fs_ecg_wesad,t_interpol=t_interpol)
        # stres_signal_ecg=np.zeros((len(t_interpol),5))
        # stres_signal_ecg[:,1]=1
        #Form class_signal from P_ID
        class_signal=np.zeros((lenth,load_data.n_classes))
        class_signal[:,load_data.class_ids[self.P_ID_out]]=1
                
        return [stres_signal_ecg,class_signal.astype(np.float32),
                avgHRV_interpol.reshape(-1,1).astype(np.float32),
                arr_pks.reshape(-1,1).astype(np.float32)]
        
    def __call__(self,ecg,stres,Fs,Fs_final=None,show_plots=False):
        
        #Resample to internal Fs
        stres_signal_ecg,class_signal,HR,_=self.pre_process(ecg,stres,Fs)
        
        #append class_signal to input condition
        HRV_cond=np.concatenate([HR,stres_signal_ecg,class_signal],axis=-1)
        Morph_cond=np.kron(HRV_cond[:,1:], np.ones((self.up_factor,1), 
                            dtype=HRV_cond[:,1:].dtype))
        #Morph_cond=np.concatenate([arr_pks,class_signal],axis=-1)
        
        #Get Rpeaks from HRV module
        arr_pks,HRV_cond,arr_tacho=self.sim_HR2pks(HRV_cond,Fs_out=self.Fs_out)
        
        #Get ECG from Morph module
        Morph_cond=np.concatenate([arr_pks.reshape(-1,1),
                                   Morph_cond[:len(arr_pks)]],axis=-1)
        synth_ecg,Morph_cond=self.sim_pks2sigs(Morph_cond)
        if Fs_final is None:
            Fs_final=Fs*1
        
        synth_ecg_final=self.post_process(synth_ecg*1, Fs_final)
        
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
                
        return synth_ecg_final,[synth_ecg,Morph_cond[:,0]]
    
    def post_process(self,synth_ecg_out,Fs_final):
        Fs_in,Fs_out=self.Fs_out,Fs_final
        #Remove normalization
        synth_ecg_out*=self.musig_dict['ecg']['sigma'] #rescale
        synth_ecg_out+=self.musig_dict['ecg']['mu'] #add back mean
        
        if Fs_in!=Fs_out:
            divisor=gcd(Fs_out,Fs_in)
            up,down=int(Fs_out/divisor),int(Fs_in/divisor)
            synth_ecg_out=load_data.resample(synth_ecg_out.reshape(-1,1),
                                             Fs_in,up,down)
            
        return synth_ecg_out
    
    def plot_signals(self,HR_list,arr_pks_list,ecg_list,clip_ecg,n_plots=3):
        #Visualize when using GT data
        ecg_in,synth_ecg_out=ecg_list
        factr=self.Fs_out/self.Fs_out
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
                 latent_size_Morph=2,Fs_tacho=5,Fs_out=100):
        #super().__init__()
        self.P_ID_out=P_ID_out
        self.Fs_tacho=Fs_tacho
        self.Fs_out=Fs_out
        self.path=path
        self.latent_size_Morph=latent_size_Morph
        self.musig_dict=musig_dict[self.P_ID_out]
        self.up_factor=int(self.Fs_out/self.Fs_tacho)
        #clipping to match R2S signals with ppg_ssqi_thresholding. Not essential
        clip_times=[1,-1]
        self.clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]
        
        #Create a subject-wise Rpeaks2Sig_Simulator object 
        self.sim_pks2sigs=Rpeaks2Sig_Simulator(Fs_in=Fs_out,Fs_out=Fs_out,
                    P_ID=P_ID_out,path=path,sig_id='ecg',
                    latent_size=latent_size_Morph,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s)

        
    def __call__(self,ecg,stres,Fs,Fs_final=None):
        
        #Resample to internal Fs
        stres_signal_ecg,class_signal,HR,arr_pks=self.pre_process(ecg,stres,Fs)
 
        #append class_signal to input condition
        HRV_cond=np.concatenate([HR,stres_signal_ecg,class_signal],axis=-1)
        Morph_cond=np.kron(HRV_cond[:,1:], np.ones((self.up_factor,1), 
                            dtype=HRV_cond[:,1:].dtype))
                
        #Get ECG from Morph module
        Morph_cond=np.concatenate([arr_pks.reshape(-1,1),
                                   Morph_cond[:len(arr_pks)]],axis=-1)

        synth_ecg,Morph_cond=self.sim_pks2sigs(Morph_cond)

        if Fs_final is None:
            Fs_final=Fs*1
        
        synth_ecg_final=self.post_process(synth_ecg*1, Fs_final)
        
        return synth_ecg_final,[synth_ecg,arr_pks]

class ECG_HRV_Modulator(ECG_HRV_Morph_Modulator):
    def __init__(self,P_ID_in,P_ID_out,path='../data/post-training/',
                 latent_size_HRV=4,latent_size_Morph=2,Fs_tacho=5,Fs_out=100):
        
        #super().__init__()
        self.Fs_tacho=Fs_tacho
        self.Fs_out=Fs_out
        self.P_ID_in=P_ID_in
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_HRV=latent_size_HRV
        self.latent_size_Morph=latent_size_Morph
        self.musig_dict=musig_dict[self.P_ID_in]
        self.up_factor=int(self.Fs_out/self.Fs_tacho)
        #clipping to match R2S signals with ppg_ssqi_thresholding. Not essential
        clip_times=[1,-1]
        self.clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]
        
        P_ID_HRV='WESAD'
        P_ID_Morph=P_ID_in
        
        # HR2Rpks_user2
        self.sim_HR2pks=HR2Rpeaks_Simulator(
                    RNN_win_len_s=win_len_s+(bsize-1)*step_s,
                    step_size_s=step_s,P_ID=P_ID_HRV,path=path,Fs_HR=Fs_tacho,
                    Fs_tacho=Fs_tacho,latent_size=latent_size_HRV)
        # Rpks2Ecg_user1
        #Create a subject-wise Rpeaks2Sig_Simulator object 
        self.sim_pks2sigs=Rpeaks2Sig_Simulator(Fs_in=Fs_out,Fs_out=Fs_out,
                    P_ID=P_ID_Morph,path=path,sig_id='ecg',
                    latent_size=latent_size_Morph,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s)

        
        return
    
    def __call__(self,ecg,stres,Fs,Fs_final=None,show_plots=False):
        
        #Resample to internal Fs
        stres_signal_ecg,class_signal,HR,_=self.pre_process(ecg,stres,Fs)
        
        #append class_signal to input condition
        HRV_cond=np.concatenate([HR,stres_signal_ecg,class_signal],axis=-1)
        Morph_cond=np.kron(HRV_cond[:,1:], np.ones((self.up_factor,1), 
                            dtype=HRV_cond[:,1:].dtype))
        #Morph_cond=np.concatenate([arr_pks,class_signal],axis=-1)
        
        #Get Rpeaks_user2 from HRV module
        arr_pks,HRV_cond,arr_tacho=self.sim_HR2pks(HRV_cond,Fs_out=self.Fs_out)
        
        #Get ECGMorph_user1 from Morph module
        Morph_cond[:,6:]=0
        Morph_cond[:,6+load_data.class_ids[self.P_ID_in]]=1

        Morph_cond=np.concatenate([arr_pks.reshape(-1,1),
                                   Morph_cond[:len(arr_pks)]],axis=-1)
        synth_ecg,Morph_cond=self.sim_pks2sigs(Morph_cond)
        if Fs_final is None:
            Fs_final=Fs*1
        
        synth_ecg_final=self.post_process(synth_ecg*1, Fs_final)
        
        return synth_ecg_final,[synth_ecg,Morph_cond[:,0]]
#%% Sample Client
if __name__=='__main__':    
    P_ID_in, P_ID_out='S7','S15'

    #Get Data
    file_path=path+f'{P_ID_in}/{P_ID_in}.pkl'
    lenth=210000 #length must be divisible by 7 for downsampling in this case
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        ecg_in = data['signal']['chest']['ECG'][-lenth:].astype(np.float32)
        #ecg_in=load_data.resample(ecg_in.reshape(-1,1),700,1,7)
        stres_in = data['label'][-lenth:].astype(np.float32)#.reshape(-1,1)

    file_path=path+f'{P_ID_out}/{P_ID_out}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        ecg_out = data['signal']['chest']['ECG'][-lenth:].astype(np.float32)
        #ecg_out=load_data.resample(ecg_out.reshape(-1,1),700,1,7)

    Fs_in=700#100
    Fs_out=Fs_ecg #Synthetic signal sampling freq
    Fs_final=Fs_in*1
    # ECG HRV+Morph modulation
    hrv_morph_mod=ECG_HRV_Morph_Modulator(P_ID_out=P_ID_out,
                                    path=ckpt_path,Fs_tacho=5,Fs_out=Fs_out)
    # Produce synthetic from S15 to S15 itself to check performance of models
    ecg_hrv_morph_mod_check,_=hrv_morph_mod(ecg_out,stres_in,Fs=Fs_in,
                                            Fs_final=Fs_final)
    # Produce synthetic from S7 to S15
    ecg_hrv_morph_mod,_=hrv_morph_mod(ecg_in,stres_in,Fs=Fs_in,
                                      Fs_final=Fs_final)
    plt.figure();plt.plot(ecg_hrv_morph_mod)
    
    # Analyze HRV and Morphological properties
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_out.flatten()
                                ,Fs=Fs_in,title_hrv=P_ID_out+'_out',
                                title_morph=P_ID_out+'_out')
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_hrv_morph_mod_check
                                ,Fs=Fs_final,title_hrv=P_ID_out+'_hrv_morph_synth_check',
                                title_morph=P_ID_out+'_hrv_morph_synth_check')
    
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_in.flatten()
                                ,Fs=Fs_in,title_hrv=P_ID_in+'_in',
                                title_morph=P_ID_in+'_in')
    morph_features,hrv_features= hrv_morph_mod.analyse_signal(ecg_hrv_morph_mod
                                ,Fs=Fs_final,title_hrv=P_ID_out+'_hrv_morph_synth',
                                title_morph=P_ID_out+'_hrv_morph_synth')
    
    # ECG Morph modulation
    morph_mod=ECG_Morph_Modulator(P_ID_out=P_ID_out,
                                  path=ckpt_path,Fs_tacho=5,Fs_out=Fs_out)
    ecg_morph_mod,_=morph_mod(ecg_in,stres_in,Fs=Fs_in,Fs_final=Fs_final)
    
    morph_features,hrv_features= morph_mod.analyse_signal(ecg_morph_mod
                                ,Fs=Fs_final,title_hrv=P_ID_in+'_morph_synth',
                                title_morph=P_ID_out+'_morph_synth')
    plt.figure();plt.plot(ecg_morph_mod)