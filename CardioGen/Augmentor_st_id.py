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
sns.set()
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
    from HR2Rpeaks import HR2Rpeaks_Simulator
    from Rpeaks2Sig import Rpeaks2Sig_Simulator
    from lib.utils import filtr_HR, get_continous_wins
    from lib.data import load_data_wesad as load_data
    proj_path='.'

else:
    from .lib.simulator_for_CC import Simulator
    from .HR2Rpeaks import HR2Rpeaks_Simulator
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
bsize=load_data.test_bsize
Fs_ppg=load_data.Fs_ppg_new
Fs_ecg=load_data.Fs_ecg_new

ver=12 #version of the model_weights to use. Refer to README for details.
Dsplit_filename = (f'{proj_path}/../data/pre-training/'
                   f'WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{bsize}.pickle')
#sys.path.append("../data/post-training/")
#from lib.sim_for_model_4 import HR_func_generator

#%%
class avgHRV2PPG_Augmentor(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,P_ID_out,path='../data/post-training/',
                 latent_size_HRV=5,latent_size_Morph=2,Fs_tacho=5,
                 Fs_pks=100,Fs_out=None,win_len_s=8,step_s=2,bsize=13,
                 dict_musig={}):
        
        super().__init__()
        self.Fs_tacho=Fs_tacho
        self.Fs_pks=Fs_pks
        self.Fs_out=Fs_out
        self.P_ID_out=P_ID_out
        self.path=path
        self.latent_size_HRV=latent_size_HRV
        self.latent_size_Morph=latent_size_Morph
        self.win_len_s=win_len_s
        self.step_s=step_s
        self.bsize=bsize
        self.dict_musig = dict_musig
        up_factor=self.Fs_out/self.Fs_tacho
        assert up_factor%1==0, 'Fs_out must be a multiple of Fs_tacho'
        self.up_factor=int(up_factor)
        #self.dict_musig=self.dict_musig[self.P_ID_out]
        
        P_ID_HRV='WESAD'
        P_ID_Morph=P_ID_out
        ppg_gen_model_config={'rnn_units':8,'disc_f_list':[8,16,16,32,64],
                          'gru_drop':0.}
        #ecg_gen_model_config={'rnn_units':8,'disc_f_list':[8,16,16,32,64],
        #                  'gru_drop':0.}
        
        self.sim_HR2pks=HR2Rpeaks_Simulator(
                    RNN_win_len_s=win_len_s+(bsize-1)*step_s,
                    step_size_s=step_s,P_ID=P_ID_HRV,path=path,Fs_HR=Fs_tacho,
                    Fs_tacho=Fs_tacho,latent_size=latent_size_HRV)
        
        #Create Simulator using train data
        
        self.sim_pks2ppg=Rpeaks2Sig_Simulator(Fs_in=Fs_out,Fs_out=Fs_out,
                    P_ID=P_ID_Morph,path=path,sig_id='ppg',
                    latent_size=latent_size_Morph,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s,
                    gen_model_config=ppg_gen_model_config)
        
        
    def __call__(self,cond_HRV):
        
        arr_pk_synth,cond_HRV,arr_tacho_synth=self.sim_HR2pks(cond_HRV,
                    Fs_out=self.Fs_out,step_size_s=win_len_s+(bsize-1)*step_s)
        
        if len(arr_pk_synth)==0:
            return None,None,None,None
        
        cond_Morph=np.kron(cond_HRV[:,1:], np.ones((self.up_factor,1), 
                            dtype=cond_HRV[:,1:].dtype))
        cond_ppg=np.concatenate([arr_pk_synth.reshape(-1,1),cond_Morph],axis=-1)
        ppg_synth,cond_ppg=self.sim_pks2ppg(cond_ppg,step_size_s=win_len_s)
        
        return cond_HRV,arr_tacho_synth,arr_pk_synth,ppg_synth



#%%
# Get all [avgHRV;Y_st;Y_Wid] windows and select train+val ones of SOI (ppg/ecg)
def get_synth_data(avghrv2ppg_aug_dict,class_name,seq_format_function,
                   Dsplit_mask_dict):
    #class_name,Fs_out='S5',Fs_ppg
    sample_key=list(avghrv2ppg_aug_dict.keys())[0]
    Fs_out=avghrv2ppg_aug_dict[sample_key].Fs_out
    Fs_tacho=avghrv2ppg_aug_dict[sample_key].Fs_tacho
    
    #for class_name in list(all_class_ids.keys()):
    load_data.class_ids={class_name:all_class_ids[class_name]}
    
    #Load all data
    list_cond_HRV,list_HRV,Dsplit_mask_dict=(load_data.get_train_data(
                data_path,mode='HR2R',win_len_s=win_len_s,step_s=step_s,
                Fs_tacho=Fs_tacho,Dsplit_mask_dict=Dsplit_mask_dict))
    
    #ppg_in_data,ppg_out_data=input_dict['ppg'][0],output_dict['ppg'][0]
    
    Dsplit_mask=Dsplit_mask_dict['hrv'][class_name] #must use hrv mask
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
    for i in range(len(start_idxs)):
    #for i in range(2):
        
        # Defragment windows into continous signal segments.
        in_seq_wins=list_cond_HRV[0][start_idxs[i]:end_idxs[i]]
        in_seq=(load_data.sliding_window_defragmentation([in_seq_wins],
                    ((bsize-1)*step_s+win_len_s)*Fs_tacho,
                    step_s*Fs_tacho))
        
        
        # Pick a signal and divide [avgHRV],[Y_st],[Y_Wid]. Keeping avgHRV fixed and 
        # cycle through 4 (out of 5) stress conditions and all 15 classes. So for 
        # every signal, we get 15*4=60 signals + 1 original signal. Hence, net 
        # augmentation factor of 61.
        cond_HRV_init=np.zeros(in_seq.shape)
        cond_HRV_init[:,0]=in_seq[:,0]*1
        start_time=time.time()
        #avghrv2ppg_aug=avghrv2ppg_aug_dict[sample_key]
        
        for s in [0,3]:
        #for s in range(n_stresses-1):
            for c in range(n_classes)[:]:
                print(f'Stress={s+1}, Class={c}')
                j=s*n_classes+c #counter
                cond_HRV=cond_HRV_init*1
                cond_HRV[:,1+s+1]=1 #extra +1 for skipping stress=0 channel
                cond_HRV[:,1+n_stresses+c]=1
                
                #Pick subject-specific model
                avghrv2ppg_aug=avghrv2ppg_aug_dict[list(all_class_ids.keys())
                                                    [c]]
                #Generate Synthetic data
                cond_HRV,arr_tacho_synth,arr_pk_synth,ppg_synth=avghrv2ppg_aug(
                                                                cond_HRV)
                
                if cond_HRV is None:
                    #in_wins,out_wins=None,None
                    continue
                else:
                    in_wins,out_wins=seq_format_function(avghrv2ppg_aug,cond_HRV,
                                    arr_tacho_synth,arr_pk_synth,ppg_synth)
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
        


# Fragment all signals back to desired fragmentation. Could reuse train & val
# masks for Dsplit (although potential issue with GRU memory propagation is 
# val data may now have seen more train data in a sense)

# def seq_format_function_TAu(avghrv2ppg_aug,cond_HRV,arr_tacho_synth,
#                             arr_pk_synth,ppg_synth):
#     bsize,bstep=5,1 #TODO: we can increase bstep here
#     Fs_out=avghrv2ppg_aug.Fs_out
#     seq_in=ppg_synth
#     seq_out=cond_HRV[:,0]
#     #upsample seq_out to Fs_out
#     #up_factor=avghrv2ppg_aug.Fs_out/avghrv2ppg_aug.Fs_tacho
#     #assert up_factor%1==0, f'up_factor should have been an integer. but is {up_factor}'
    
#     seq_out=load_data.resample(seq_out,avghrv2ppg_aug.Fs_tacho,
#                 avghrv2ppg_aug.up_factor,1,show_plots=False).reshape(-1,1)
    
#     # Apt block creation as per TAu but using Dsplit_mask['hrv']
#     # Fragment all signals back to desired fragmentation. Could reuse train & val
#     # masks for Dsplit (although potential issue with GRU memory propagation is 
#     # val data may now have seen more train data in a sense)
#     in_wins,out_wins=load_data.sliding_window_fragmentation([seq_in,seq_out],
#                     ((bsize-1)*step_s+win_len_s)*Fs_out,
#                     bstep*step_s*Fs_out)
#     return in_wins,out_wins


def seq_format_function_TAu(avghrv2ppg_aug,cond_HRV,arr_tacho_synth,
                            arr_pk_synth,ppg_synth):
    bsize,bstep=5,2#1 #TODO: we can increase bstep here
    Fs_out=avghrv2ppg_aug.Fs_out
    seq_in=ppg_synth
    #seq_out=cond_HRV[:,0].reshape(-1,1)
    seq_out=np.stack([cond_HRV[:,0],arr_tacho_synth],axis=1)
    
    #upsample seq_out to Fs_out
    #up_factor=avghrv2ppg_aug.Fs_out/avghrv2ppg_aug.Fs_tacho
    #assert up_factor%1==0, f'up_factor should have been an integer. but is {up_factor}'
    
    seq_out=load_data.resample(seq_out,avghrv2ppg_aug.Fs_tacho,
                avghrv2ppg_aug.up_factor,1,show_plots=False)
    #print(seq_out.shape)
    # Apt block creation as per TAu but using Dsplit_mask['hrv']
    # Fragment all signals back to desired fragmentation. Could reuse train & val
    # masks for Dsplit (although potential issue with GRU memory propagation is 
    # val data may now have seen more train data in a sense)
    in_wins,out_wins=load_data.sliding_window_fragmentation([seq_in,seq_out],
                    ((bsize-1)*step_s+win_len_s)*Fs_out,
                    bstep*step_s*Fs_out)
    return in_wins,out_wins
#%% Client

def main(seq_format_function=seq_format_function_TAu,
         save_name='WESAD_synth_cleansub_TAu',show_plots=False,
         suffix='s14_c13to17'):
    #P_ID='WESAD'
    latent_size_HRV=4
    latent_size_Morph=2
    Fs_tacho=5
    
    if os.path.isfile(Dsplit_filename):
        with open (Dsplit_filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                       'Run get_train_data in R2S mode first.')
        
    #Load all augmentor models in a single dict
    avghrv2ppg_aug_dict={}
    
    def get_aug_dict():
        for clas_name in list(all_class_ids.keys())[:]:
            #Create Simulator Model
            avghrv2ppg_aug=avgHRV2PPG_Augmentor(P_ID_out=clas_name,path=model_path,
                            latent_size_HRV=latent_size_HRV,
                            latent_size_Morph=latent_size_Morph,Fs_tacho=Fs_tacho,
                            Fs_pks=100,Fs_out=Fs_ppg,win_len_s=win_len_s,
                            step_s=step_s,bsize=bsize)
            
            # Put specialized models in a dict
            #avghrv2ppg_aug_list.append(avghrv2ppg_aug)
            avghrv2ppg_aug_dict[clas_name]=avghrv2ppg_aug
            del avghrv2ppg_aug
        print(len(avghrv2ppg_aug_dict))
        return
        
    save_dir=f'{data_path}../{save_name}'
    os.makedirs(save_dir,exist_ok=True)
    in_data_list,out_data_list=[],[]
    #class_name='S7'
    #for class_name in ['S15']:#,'S11','S17']:
        
    for class_name in list(all_class_ids.keys())[:]:
        filename = (save_dir+f'/{class_name}_{suffix}.pickle')
        
        if os.path.isfile(filename):
            with open (filename, 'rb') as fp:
                in_data,out_data = pickle.load(fp)
        else:
            if len(avghrv2ppg_aug_dict)==0: get_aug_dict()
            in_data,out_data=get_synth_data(avghrv2ppg_aug_dict,class_name,
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
    
    # #test_idx=0
    # #plt.close('all')
    # for test_idx in np.linspace(0,len(in_data)-1,10).astype(int):
    #     plt.figure();plt.plot(in_data[test_idx]);plt.plot(out_data[test_idx])
    
    #========
    # list_cond_HRV,list_HRV,Dsplit_mask_dict=load_data.get_test_data(
    #                 data_path+class_name,mode='HR2R',win_len_s=win_len_s,
    #                 step_s=step_s,Fs_tacho=Fs_tacho,
    #                 Dsplit_mask_dict=Dsplit_mask_dict)
    
    # class_no,test_seq_no=0,0
    
    # cond_HRV_wins=list_cond_HRV[class_no][test_seq_no]
    # HRV_real_wins=list_HRV[class_no][test_seq_no]

    # # Add noise
    # #plt.figure(99);plt.plot(cond_ecg_wins[0,:,0])
    # #cond_ecg_wins[:,:,0]+=np.random.normal(0,0.1,cond_ecg_wins[:,:,0].shape)
    # #plt.plot(cond_ecg_wins[0,:,0],'--')
    
    # # defragment windows into continous signal
    # test_in,test_out_for_check=(load_data.sliding_window_defragmentation([
    #     cond_HRV_wins,HRV_real_wins],
    #     ((load_data.test_bsize-1)*step_s+win_len_s)*Fs_tacho,
    #     step_s*Fs_tacho))
    
    # #Generate Synthetic data
    # cond_HRV,arr_tacho_synth,arr_pk_synth,ppg_synth=avghrv2ppg_aug(test_in)
    # t_tacho=np.arange(len(arr_tacho_synth))/Fs_tacho
    # t_ppg=np.arange(len(ppg_synth))/Fs_ppg
    # #t_ecg=np.arange(len(ecg_synth))/Fs_ecg
    
    # if show_plots:
    #     plt.figure();ax1=plt.subplot(211)
    #     plt.plot(t_tacho,cond_HRV[:,0],t_tacho,arr_tacho_synth,'--')
    #     plt.legend(['In:avgHRV','Out:HRV (tacho)'])
    #     plt.grid(True)
    #     plt.subplot(212,sharex=ax1)
    #     plt.plot(t_ppg,arr_pk_synth,t_ppg,ppg_synth,'--')
    #     plt.legend(['In:R-peaks','Out:PPG'])
    #     plt.grid(True)
        
    return in_data,out_data
#%%
if __name__=='__main__':
    main(seq_format_function_TAu,save_name='WESAD_synth_TAu/ppg',
         show_plots=False,suffix='s14_c')