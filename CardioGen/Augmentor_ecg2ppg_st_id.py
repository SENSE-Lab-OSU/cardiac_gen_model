# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 01:36:15 2021

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
import copy
import time
from scipy import signal as sig

#sns.set()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float32')

#print(proj_path)



import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from lib.simulator_for_CC import Simulator
    #from HR2Rpeaks import HR2Rpeaks_Simulator
    from Rpeaks2Sig import Rpeaks2Sig_Simulator
    from lib.utils import filtr_HR, get_continous_wins
    from lib.data import load_data_wesad as load_data
    proj_path='.'

else:
    from .lib.simulator_for_CC import Simulator
    #from .HR2Rpeaks import HR2Rpeaks_Simulator
    from .Rpeaks2Sig import Rpeaks2Sig_Simulator
    from .lib.utils import filtr_HR, get_continous_wins
    from .lib.data import load_data_wesad as load_data
    proj_path=(os.path.dirname(os.path.abspath(__file__))).replace(os.sep,'/')


sys.path.append(proj_path)
#Define global constants
n_classes=load_data.n_classes
n_stresses=load_data.n_stresses
all_class_ids=copy.deepcopy(load_data.class_ids)
model_path=proj_path+'/../data/post-training/'
data_path='D:/Datasets/WESAD/'

win_len_s=load_data.win_len_s
step_s=load_data.step_s
#bsize=load_data.test_bsize
#Fs_ecg=load_data.Fs_ecg_new
Fs_ppg=load_data.Fs_ppg_new


ver=12 #version of the model_weights to use. Refer to README for details.
Dsplit_filename = (f'{proj_path}/../data/pre-training/'
                   f'WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{load_data.test_bsize}.pickle')
#sys.path.append("../data/post-training/")
#from lib.sim_for_model_4 import HR_func_generator

#%%
class ECG2PPG_Augmentor(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,P_ID_out,path='../data/post-training/',
                 latent_size_Morph=2,Fs_in=100,
                 Fs_out=None,win_len_s=8,step_s=2,
                 dict_musig={}):
        
        super().__init__()
        self.Fs_in=Fs_in
        self.Fs_out=Fs_out
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_Morph=latent_size_Morph
        self.win_len_s=win_len_s
        self.step_s=step_s
        self.dict_musig = dict_musig
        down_factor=self.Fs_in/self.Fs_out
        assert down_factor%1==0, 'Fs_in must be a multiple of Fs_out'
        self.down_factor=int(down_factor)
        #self.dict_musig=self.dict_musig[self.P_ID_out]
        
        P_ID_Morph=P_ID_out
        ppg_gen_model_config={'rnn_units':8,'disc_f_list':[8,16,16,32,64],
                              'gru_drop':0.,'z_up_factor':1}
        
        
        #Create Simulator using train data
        self.sim_pks2ppg=Rpeaks2Sig_Simulator(Fs_in=Fs_out,
            Fs_out=Fs_out,
            P_ID=P_ID_Morph,path=path,sig_id='ppg',
            latent_size=latent_size_Morph,logging=False,batch_size=32,
            RNN_win_len=win_len_s,win_step_size=step_s,
            gen_model_config=ppg_gen_model_config)
        
        
    def __call__(self,cond_Morph):
        #ecg_synth,cond_ecg=self.sim_pks2ecg(cond_ecg,step_size_s=win_len_s)
        ppg_synth,cond_ppg=self.sim_pks2ppg(cond_Morph,step_size_s=win_len_s)
        return cond_ppg,ppg_synth



#%%
# Get all [avgHRV;Y_st;Y_Wid] windows and select train+val ones of SOI (ppg/ecg)
def get_synth_data(ecg2ppg_aug_dict,class_name,seq_format_function,
                   Dsplit_mask_dict):
    #class_name,Fs_out='S5',Fs_ecg
    sample_key=list(ecg2ppg_aug_dict.keys())[0]
    #Fs_out=ecg2ppg_aug_dict[sample_key].Fs_out
    #Fs_tacho=ecg2ppg_aug_dict[sample_key].Fs_tacho
    Fs_in=ecg2ppg_aug_dict[sample_key].Fs_in
    bsize,bstep=2,1
    
    #for class_name in list(all_class_ids.keys()):
    load_data.class_ids={class_name:all_class_ids[class_name]}
    
    #Load all data
    input_dict,output_dict,musig_dict,Dsplit_mask_dict=(load_data.
                get_train_data(data_path,mode='R2S',win_len_s=win_len_s,
                step_s=step_s,Dsplit_mask_dict=Dsplit_mask_dict))
    
    
    Dsplit_mask=Dsplit_mask_dict['ecg'][class_name] #must use hrv mask
    list_cond_Morph=input_dict['ppg']
    
    train_mask=Dsplit_mask[0].astype(int)
    #val_train_mask= np.sum(Dsplit_mask[0:2],axis=0).astype(int)
    sel_mask=train_mask*1
    
    start_idxs,end_idxs=get_continous_wins(sel_mask)
    #start_idxs,n_stresses,n_classes=start_idxs[:2],3,2#TODO:For Debugging only
    print(f'Generating synthetic data using subject {class_name} with '
          f'{len(start_idxs)} sequences')
    
    in_data_synth=[[] for j in range((n_stresses-1)*n_classes)]
    out_data_synth=[[] for j in range((n_stresses-1)*n_classes)]

    # Iterate over each set and generate data
    for i in range(0,len(start_idxs)):
    #for i in range(2):
        # Defragment windows into continous signal segments.
        in_seq_wins=list_cond_Morph[0][start_idxs[i]:end_idxs[i]]
        in_seq=(load_data.sliding_window_defragmentation([in_seq_wins],
                    (win_len_s)*Fs_in,step_s*Fs_in))
        
        
        # Pick a signal and divide [avgHRV],[Y_st],[Y_Wid]. Keeping avgHRV fixed and 
        # cycle through 4 (out of 5) stress conditions and all 15 classes. So for 
        # every signal, we get 15*4=60 signals + 1 original signal. Hence, net 
        # augmentation factor of 61.
        cond_Morph_init=np.zeros(in_seq.shape)
        cond_Morph_init[:,0]=in_seq[:,0]*1
        start_time=time.time()
        #ecg2ppg_aug=ecg2ppg_aug_dict[sample_key]
        
        for s in [0,1,2,3]:
        #for s in range(n_stresses-1):
            for c in range(n_classes)[:]:
                print(f'Stress={s+1}, Class={c}')
                j=s*n_classes+c #counter
                cond_Morph=cond_Morph_init*1
                cond_Morph[:,1+s+1]=1 #extra +1 for skipping stress=0 channel
                cond_Morph[:,1+n_stresses+c]=1
                
                
                #Pick subject-specific model
                ecg2ppg_aug=ecg2ppg_aug_dict[list(all_class_ids.keys())[c]]
                #Generate Synthetic data
                cond_ppg,ppg_synth=ecg2ppg_aug(cond_Morph)
                
                min_len_fail=(len(cond_ppg)<max(((bsize-1)*step_s+win_len_s)*
                                               ecg2ppg_aug.Fs_out,747))
                if ((cond_ppg is None) or (min_len_fail)):
                    print('Not enough samples, returning empty list...\n')
                    #in_wins,out_wins=None,None
                    continue
                else:
                    in_wins,out_wins=seq_format_function(ecg2ppg_aug,
                                                         cond_ppg,ppg_synth,
                                                         bsize,bstep)
                    in_data_synth[j].append(in_wins)
                    out_data_synth[j].append(out_wins)
                        
            print(f'Time taken for sequence {i}= {time.time()-start_time}')
    in_data_list=[np.concatenate(arr_list,axis=0) 
                  for arr_list in in_data_synth if len(arr_list)>0]
    out_data_list=[np.concatenate(arr_list,axis=0) 
                   for arr_list in out_data_synth if len(arr_list)>0]
    print('\n=======================\n',len(out_data_list))
    in_data=np.concatenate(in_data_list,axis=0).astype(np.float32)
    out_data=np.concatenate(out_data_list,axis=0).astype(np.float32)
    
    # samp_idx=10
    # plt.figure();plt.plot(in_data[samp_idx,:,:])
    # plt.plot(out_data[samp_idx,:,:])
    return in_data,out_data
        


def seq_format_function_P2St(ecg2ppg_aug,cond_ppg,ppg_synth,bsize=2,bstep=1):
    #1 #TODO: we can increase bstep here
    Fs_out_sim=ecg2ppg_aug.Fs_out
    Fs_out=25
    seq_in=ppg_synth
    
    seq_out=cond_ppg
    # 747 here comes from min padlen required by ppg_filter's filtfilt
    if len(seq_out)<max(((bsize-1)*step_s+win_len_s)*Fs_out_sim,747):
        print('Not enough samples, returning empty arrays...\n')
        return np.ones(0),np.ones(0)
    
    
    #TODO: May remove these pre-pro steps for other analyses
    # Resample
    if Fs_out_sim!=Fs_out:
        #assert (Fs_out_sim/4)%1==0, 'Fs_out_sim must be a multiple of 4'
        seq_in=load_data.resample(seq_in,Fs_out_sim,int(Fs_out),
                                   int(Fs_out_sim),show_plots=False)
        seq_out=load_data.resample(seq_out,Fs_out_sim,int(Fs_out),
                                   int(Fs_out_sim),show_plots=False)
    ## TODO: filter or not? Only minor changes after filtering
    #ppg_filt=seq_in.reshape(-1,1)*1#load_data.ppg_filter(seq_in.reshape(-1,1),Fs_out,show_plots=True)
    ppg_filt=load_data.ppg_filter(seq_in.reshape(-1,1),Fs_out,show_plots=False)

    ## z-normalize
    ppg_mu,ppg_sigma=np.mean(ppg_filt.flatten()),np.std(ppg_filt.flatten())
    seq_in=((ppg_filt.flatten()-ppg_mu)/ppg_sigma)
    
    
    #print(seq_out.shape)
    # Apt block creation as per E2St but using Dsplit_mask['hrv']
    # Fragment all signals back to desired fragmentation. Could reuse train & val
    # masks for Dsplit (although potential issue with GRU memory propagation is 
    # val data may now have seen more train data in a sense)
    in_wins,out_wins=load_data.sliding_window_fragmentation([seq_in,seq_out],
                    ((bsize-1)*step_s+win_len_s)*Fs_out,
                    bstep*step_s*Fs_out)
    #select only subject ids and classes in that order
    out_wins=out_wins[:,0,1:]
    #TODO: Maybe add avgHR and arr_pk to in_wins for saving?
    return in_wins,out_wins
#%% Client

def main(seq_format_function=seq_format_function_P2St,
         save_name='WESAD_synth_cleansub_E2St',show_plots=False,
         suffix='s14_c13to17',class_ids=list(all_class_ids.keys())):
    #P_ID='WESAD'
    latent_size_Morph=2
    #Fs_tacho=5
    
    if os.path.isfile(Dsplit_filename):
        with open (Dsplit_filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                       'Run get_train_data in R2S mode first.')
        
    #Load all augmentor models in a single dict
    ecg2ppg_aug_dict={}
    
    def get_aug_dict():
        for clas_name in class_ids[:]:
            #Create Simulator Model
            ecg2ppg_aug=ECG2PPG_Augmentor(P_ID_out=clas_name,path=model_path,
                            latent_size_Morph=latent_size_Morph,Fs_in=Fs_ppg,
                            Fs_out=Fs_ppg,win_len_s=win_len_s,
                            step_s=step_s)
            
            # Put specialized models in a dict
            #ecg2ppg_aug_list.append(ecg2ppg_aug)
            ecg2ppg_aug_dict[clas_name]=ecg2ppg_aug
            del ecg2ppg_aug
        print(len(ecg2ppg_aug_dict))
        return
        
    save_dir=f'{data_path}../{save_name}'
    os.makedirs(save_dir,exist_ok=True)
    in_data_list,out_data_list=[],[]
    #class_name='S7'
    #for class_name in ['S15']:#,'S11','S17']:
        
    for class_name in class_ids[:]:
        filename = (save_dir+f'/{class_name}_{suffix}.pickle')
        
        if os.path.isfile(filename):
            with open (filename, 'rb') as fp:
                in_data,out_data = pickle.load(fp)
        else:
            if len(ecg2ppg_aug_dict)==0: get_aug_dict()
            in_data,out_data=get_synth_data(ecg2ppg_aug_dict,class_name,
                                        seq_format_function,Dsplit_mask_dict)
            # Save data
            with open(filename, 'wb') as handle:
                pickle.dump([in_data,out_data], handle)
                
        print(in_data.shape,out_data.shape)
        in_data_list.append(in_data)
        out_data_list.append(out_data)    
        
    in_data=np.concatenate(in_data_list,axis=0).astype(np.float32)
    out_data=np.concatenate(out_data_list,axis=0).astype(np.float32)
    print(in_data.shape,out_data.shape)
    return in_data,out_data
#%%
if __name__=='__main__':
    main(seq_format_function_P2St,save_name='WESAD_synth_h30_m28/P2StId',
         show_plots=False,suffix='s_c')