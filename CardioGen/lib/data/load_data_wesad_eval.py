import numpy as np
import scipy as sp
import pandas as pd
import glob
from scipy import signal as sig
import pickle
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import sys
import inspect
#add the parent module to the path for following imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils import filtr_HR, get_uniform_tacho, get_continous_wins
from .load_data_wesad import *
#from CardioGen.lib.utils import filtr_HR, get_uniform_tacho

MAX_PPG_VAL=1000 #1789
MAX_ECG_VAL=1
FIX_ECG_MEAN=0.5

Fs_acc,Fs_ppg,Fs_ecg=32,64,700
Fs_acc_new,Fs_ppg_new,Fs_ecg_new=25,25,100
Fs_ecg_E2St,Fs_ppg_P2St=256,Fs_ppg_new
assert Fs_acc_new==Fs_ppg_new, 'Fs_acc_new must be equal to Fs_ppg_new'
factr=(Fs_ecg_new/Fs_ppg_new)
assert factr%1==0, '(Fs_ecg_new/Fs_ppg_new) must be an integer'
factr=int(factr)
test_ratio,val_ratio=0.1,0.1
test_bsize,val_bsize=13,13
win_len_s=8
step_s=2

class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(4,8))+list(range(15,16)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(2,4)))}
#{'S2': 0, 'S3': 1, 'S4': 2, 'S5': 3, 'S6': 4, 'S7': 5, 'S8': 6, 'S9': 7, 'S10': 8, 'S11': 9, 'S13': 10, 'S14': 11, 'S15': 12, 'S16': 13, 'S17': 14}
#class_ids={'S17':14}
n_classes=15#len(class_ids)
n_stresses=5

#%%


def get_clean_E2St_data(files,win_len_s,step_s,
                        Dsplit_mask_dict,musig_dict,
                        bsize=2,bstep=1,mode='train'):
    '''
    Extract data from 'clean' files
    '''
    #MAX_ECG_VAL=1
    list_in,list_out=[],[]
    list_HR=[]
    #start_idx,end_idx=get_start_end_idxs(mode)
    clip_times=[1,-1]
    clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]

    Dsplit_mask_dict['E2St']={}
    Dsplit_mask_dict['Dspecs']['key_order']=[]
    #Fs_tacho=Fs_ppg_new # Important assumption
    
    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        class_name=(files[i].split('/')[-2])
        print(f'{class_name} \n')
        
        ppg = data['signal']['wrist']['BVP'].astype(np.float32)
        ecg = data['signal']['chest']['ECG'].astype(np.float32)
        stres = data['label'].astype(np.float32)#.reshape(-1,1)
        
        # CardioGen Pre-processing
        ## Clean-up signals
        ppg_filt=ppg_filter(ppg,Fs_ppg)
        ecg=nk.ecg_clean(ecg.flatten(), sampling_rate=Fs_ecg, method="neurokit")
        stres[stres>=5] = 0. #zero-out any meaningless label >=5
        
        ## Resampling
        ppg_resamp=resample(ppg_filt,Fs_ppg,25,64)
        ecg_resamp=resample(ecg,Fs_ecg,1,7)
        ppg,ecg=ppg_resamp,ecg_resamp

        ## Normalize
        ppg-=musig_dict[class_name]['ppg']['mu']
        ppg/=musig_dict[class_name]['ppg']['sigma']
        ecg-=musig_dict[class_name]['ecg']['mu']
        ecg/=musig_dict[class_name]['ecg']['sigma']
        
        
        arr_pks, rpeaks=find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        Fs_pks=Fs_ecg_new

        #data=(resample(ecg,ldw.Fs_ecg,int(freq/4),int(700/4),show_plots=False)).flatten()
        t_interpol=np.arange(len(ecg))/Fs_ecg_new
        labels,_ = create_stress_signal(stres,Fs=Fs_ecg,t_interpol=t_interpol,
                                      show_plots=False)

        ## Clip signals 1s on both sides to match with R2S data
        ppg=clip_sig(ppg,Fs_ppg_new)
        ecg=clip_sig(ecg,Fs_ecg_new)
        labels=clip_sig(labels,Fs_ecg_new)
        arr_pks=clip_sig(arr_pks,Fs_pks)
        # SSL_ECG Pre-processing
        
        #Fs_ecg_new,Fs_stres=100,100
        Fs_out_sim,Fs_out=Fs_ecg_new,Fs_ecg_E2St
        
        ## Resample
        assert (Fs_out_sim/4)%1==0, 'Fs_out_sim must be a multiple of 4'
        seq_in=resample(ecg,Fs_out_sim,int(Fs_out/4),
                                   int(Fs_out_sim/4),show_plots=False)
        seq_out=resample(labels,Fs_out_sim,int(Fs_out/4),
                                   int(Fs_out_sim/4),show_plots=False)
        ## HP filter
        ecg_filt=ecg_filter(seq_in.reshape(-1,1),Fs=Fs_ecg_E2St,
                                 f_pass=0.8,show_plots=False)
        ## z-normalize
        ecg_mu,ecg_sigma=np.mean(ecg_filt.flatten()),np.std(ecg_filt.flatten())
        seq_in=((ecg_filt.flatten()-ecg_mu)/ecg_sigma)

        # # TODO: Form class_signal from class_id
        class_id=class_ids[class_name]
        class_signal=np.zeros((*seq_out.shape[:-1],n_classes))
        class_signal[:,class_id]=1
        seq_out=np.concatenate([seq_out,class_signal],axis=-1)
        
        #Find smooth HR by average RR-intervals in windows
        t_HR=np.arange(len(seq_in))/Fs_ecg_E2St
        RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks.flatten(),Fs_pks)
        t=np.cumsum(RR_ints_NU)+(RR_extreme_idx[0]/Fs_pks)
        f_interpol = sp.interpolate.interp1d(t, RR_ints_NU,'cubic',axis=0,
                            bounds_error=False,
                            fill_value=(RR_ints_NU[0],RR_ints_NU[-1]))
        RR_ints = f_interpol(t_HR)
        # tacho=tacho_filter(RR_ints, Fs_ecg_E2St,f_cutoff=0.5,order=None,
        #            show_plots=False,margin=0.15)
        
        #TODO: Replaced HR with RR here by removing **-1
        HR_interpol=(RR_ints).flatten()
        
        seq_in=np.stack([seq_in,HR_interpol],axis=-1)
        # plt.figure()
        # plt.plot(np.arange(len(arr_pks))/Fs_pks,arr_pks,t_HR,RR_ints,
        #          t_HR,tacho,'--')
        
        #Fragment intro windows
        in_cond,out=sliding_window_fragmentation([seq_in,seq_out],
                            ((bsize-1)*step_s+win_len_s)*Fs_ecg_E2St,
                            bstep*step_s*Fs_ecg_E2St)
        
        
        # arr_pk_wins,t_HR=sliding_window_fragmentation([arr_pks.flatten(),
        #                 np.arange(len(arr_pks))/Fs_pks],
        #                 win_size=((bsize-1)*step_s+win_len_s)*Fs_pks,
        #                 step_size=bstep*step_s*Fs_pks,axes=None)
        # t_HR=np.mean(t_HR,axis=-1)
        # avgHRV=np.ones((len(arr_pk_wins),in_cond.shape[1]))
        # for j in range(len(arr_pk_wins)):
        #     RR_ints_NU, _=Rpeaks2RRint(arr_pk_wins[j],Fs_pks)
        #     avgHRV[j,:]=np.mean(RR_ints_NU)
        # #avgHRV=np.expand_dims(avgHRV,axis=-1)
        
        # assert len(in_cond)==len(avgHRV), 'Unexpected Dimensional mismatch'
        
        # #Testing iterated windowing for HRV module
        # if mode=='test_common':
        #     Dsplit_mask_clipped=np.concatenate(
        #                     [Dsplit_mask_dict['ecg'][class_name][:2],
        #                     Dsplit_mask_dict['ppg'][class_name][2:]],axis=0)
        #     Dsplit_mask_clipped=Dsplit_mask_clipped[:,idx_start:idx_stop]
        # else:
        #     #Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name]
        #     Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name][:,idx_start:idx_stop]
        
        Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name]#[:,idx_start:idx_stop]
        #plt.figure();plt.plot(Dsplit_mask_clipped.astype(int)[-1])
        assert bstep==1,'bstep MUST be 1 to avoid temporal misalignment of mask'
        
        Dsplit_mask_E2St=[get_windowed_mask(sel_mask,win_len=bsize,step=bstep)
                          for sel_mask in Dsplit_mask_clipped]
        Dsplit_mask_dict['E2St'][class_name]=np.stack(Dsplit_mask_E2St,
                                                     axis=0).astype(bool)
        Dsplit_mask_dict['Dspecs']['key_order'].append(class_name)
        
        #verify if dimensions match after processing
        assert Dsplit_mask_dict['E2St'][class_name].shape[1]==in_cond.shape[0]
        #plt.figure();plt.plot(Dsplit_mask_hrv.T);plt.legend(['train','val','test'])
        
        #in_cond=np.stack([in_cond,avgHRV],axis=-1)
        list_in.append(in_cond.astype(np.float32))# (N,2560,2)
        list_out.append(out.astype(np.float32)) #(N,2560,5)
        #list_HR.append(avgHRV) #(N,2560)

        
        if ((mode=='test') or (mode=='test_common')):
            # get selected test blocks
            test_bsel_mask=Dsplit_mask_dict['E2St'][class_name][2]
            start_idxs,end_idxs=get_continous_wins(test_bsel_mask.astype(int))
            n_segments=len(start_idxs)
            #assert np.mean(end_idxs-start_idxs)==1, 'problem with test_bsize' 
            def temp_formatter(arr_list):
                arr_list=[arr_list[0][start_idxs[j]:end_idxs[j]]
                                          for j in range(n_segments)]
                arr_list=[sliding_window_defragmentation([arr],
                            ((bsize-1)*step_s+win_len_s)*Fs_ecg_E2St,
                            bstep*step_s*Fs_ecg_E2St) for arr in arr_list]
                arr_list=[sliding_window_fragmentation([arr],
                            ((bsize-1)*step_s+win_len_s)*Fs_ecg_E2St,
                            ((bsize-1)*step_s+win_len_s)*Fs_ecg_E2St) 
                            for arr in arr_list]
                return arr_list
            list_in=temp_formatter(list_in)
            list_out=temp_formatter(list_out)
                
                
            # test_bsel_idxs=np.arange(len(test_bsel_mask)
            #                              )[test_bsel_mask.astype(bool)]
            # for arr_list in [list_in,list_out]:
            #     arr_list[-1]=[arr_list[-1][bidx:(bidx+1)] 
            #                   for bidx in test_bsel_idxs]
            
    
    return list_in,list_out,Dsplit_mask_dict

def get_clean_eval_data(files,win_len_s,step_s,musig_dict):
    '''
    Extract data from 'clean' files
    '''
    #MAX_ECG_VAL=1
    list_in,list_out=[],[]
    list_HR=[]
    #start_idx,end_idx=get_start_end_idxs(mode)
    clip_times=[1,-1]
    clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]
    
    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        class_name=(files[i].split('/')[-2])
        print(f'{class_name} \n')
        
        ppg = data['signal']['wrist']['BVP'].astype(np.float32)
        ecg = data['signal']['chest']['ECG'].astype(np.float32)
        stres = data['label'].astype(np.float32)#.reshape(-1,1)
        
        # CardioGen Pre-processing
        ## Clean-up signals
        ppg_filt=ppg_filter(ppg,Fs_ppg)
        ecg=nk.ecg_clean(ecg.flatten(), sampling_rate=Fs_ecg, method="neurokit")
        stres[stres>=5] = 0. #zero-out any meaningless label >=5
        
        ## Resampling
        ppg_resamp=resample(ppg_filt,Fs_ppg,25,64)
        ecg_resamp=resample(ecg,Fs_ecg,1,7)
        ppg,ecg=ppg_resamp,ecg_resamp

        ## Normalize
        ppg-=musig_dict[class_name]['ppg']['mu']
        ppg/=musig_dict[class_name]['ppg']['sigma']
        ecg-=musig_dict[class_name]['ecg']['mu']
        ecg/=musig_dict[class_name]['ecg']['sigma']
        
        
        arr_pks, rpeaks=find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        Fs_pks=Fs_ecg_new

        #data=(resample(ecg,ldw.Fs_ecg,int(freq/4),int(700/4),show_plots=False)).flatten()
        t_interpol=np.arange(len(ecg))/Fs_ecg_new
        labels,_ = create_stress_signal(stres,Fs=Fs_ecg,t_interpol=t_interpol,
                                      show_plots=False)

        ## Clip signals 1s on both sides to match with R2S data
        ppg=clip_sig(ppg,Fs_ppg_new)
        ecg=clip_sig(ecg,Fs_ecg_new)
        labels=clip_sig(labels,Fs_ecg_new)
        arr_pks=clip_sig(arr_pks,Fs_pks)
                
        ## Resample
        seq_in=np.stack([arr_pks[:,0],ecg.flatten()],axis=-1)
        seq_out=labels*1#np.argmax(labels,axis=-1)


        
        # #Find smooth HR by average RR-intervals in windows
        # t_HR=np.arange(len(seq_in))/Fs_ecg_new
        # RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks.flatten(),Fs_pks)
        # t=np.cumsum(RR_ints_NU)+(RR_extreme_idx[0]/Fs_pks)
        # f_interpol = sp.interpolate.interp1d(t, RR_ints_NU,'cubic',axis=0,
        #                     bounds_error=False,
        #                     fill_value=(RR_ints_NU[0],RR_ints_NU[-1]))
        # RR_ints = f_interpol(t_HR)
        # # tacho=tacho_filter(RR_ints, Fs_ecg_E2St,f_cutoff=0.5,order=None,
        # #            show_plots=False,margin=0.15)
        # HR_interpol=(RR_ints**-1)#.flatten()
        
        # seq_in=np.concatenate([seq_in,HR_interpol],axis=-1)
        # plt.figure()
        # plt.plot(np.arange(len(arr_pks))/Fs_pks,arr_pks,t_HR,RR_ints,
        #          t_HR,tacho,'--')
        
        #Fragment intro windows
        in_cond,out=sliding_window_fragmentation([seq_in,seq_out],
                             win_len_s*Fs_ecg_new,step_s*Fs_ecg_new)
        
        
        #label_wins.shape=[N,T,2],HR_wins.shape=[N,T]--> out=(st,id,HR).shape=[N,3]
    
        
        #stress
        eps=1e-8
        label=np.round(np.mean(out[:,:,0:5],axis=1)+eps).astype(int)
        check_label_mask=np.invert((np.sum(label,axis=-1)==1))
        if np.mean(check_label_mask.astype(int))!=0:
            print('\n Resolving ties in label assignment...\n ')
            label[check_label_mask,0]=0 #zero out the useless class
            check_label_mask=np.invert((np.sum(label,axis=-1)==1)) #recalculate
        assert np.mean(check_label_mask.astype(int))==0, 'ties not resolved as expected'
        
        sel_mask_label=np.invert(label[:,0].astype(bool))
        #np.argmax(labels,axis=-1)
        label=np.argmax(label,axis=-1) #remove first row
        in_cond,label=in_cond[sel_mask_label],label[sel_mask_label]
        #y_test=y_test[:,0,:]
        #ecg_test,HR_test=ecg_test[:,:,0:1],np.mean(ecg_test[:,:,1],axis=1)
    
        # # TODO: Form class_signal from class_id
        class_id=class_ids[class_name]
        class_signal=np.ones_like(label)*class_id
        out=np.stack([label,class_signal],axis=-1)
        
        #in_cond=np.stack([in_cond,avgHRV],axis=-1)
        list_in.append(in_cond.astype(np.float32))# (N,3200,2)
        list_out.append(out) #(N,2)
        #list_HR.append(avgHRV) #(N,2560)
            
    
    return list_in,list_out

def get_clean_P2St_data(files,win_len_s,step_s,
                        Dsplit_mask_dict,musig_dict,
                        bsize=2,bstep=1,mode='train'):
    '''
    Extract data from 'clean' files
    '''
    #MAX_ECG_VAL=1
    list_in,list_out=[],[]
    list_HR=[]
    #start_idx,end_idx=get_start_end_idxs(mode)
    clip_times=[1,-1]
    clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]
    Dsplit_mask_dict['P2St']={}
    Dsplit_mask_dict['Dspecs']['key_order']=[]
    #Fs_tacho=Fs_ppg_new # Important assumption
    
    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        class_name=(files[i].split('/')[-2])
        print(f'{class_name} \n')
        
        ppg = data['signal']['wrist']['BVP'].astype(np.float32)
        ecg = data['signal']['chest']['ECG'].astype(np.float32)
        stres = data['label'].astype(np.float32)#.reshape(-1,1)
        
        # CardioGen Pre-processing
        ## Clean-up signals
        ppg_filt=ppg_filter(ppg,Fs_ppg)
        ecg=nk.ecg_clean(ecg.flatten(), sampling_rate=Fs_ecg, method="neurokit")
        stres[stres>=5] = 0. #zero-out any meaningless label >=5
        
        ## Resampling
        ppg_resamp=resample(ppg_filt,Fs_ppg,25,64)
        ecg_resamp=resample(ecg,Fs_ecg,1,7)
        ppg,ecg=ppg_resamp,ecg_resamp

        ## Normalize
        ppg-=musig_dict[class_name]['ppg']['mu']
        ppg/=musig_dict[class_name]['ppg']['sigma']
        ecg-=musig_dict[class_name]['ecg']['mu']
        ecg/=musig_dict[class_name]['ecg']['sigma']
        
        
        arr_pks, rpeaks=find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        Fs_pks=Fs_ecg_new

        #data=(resample(ecg,ldw.Fs_ecg,int(freq/4),int(700/4),show_plots=False)).flatten()
        t_interpol=np.arange(len(ppg))/Fs_ppg_new
        labels,_ = create_stress_signal(stres,Fs=Fs_ecg,t_interpol=t_interpol,
                                      show_plots=False)

        ## Clip signals 1s on both sides to match with R2S data
        seq_in=clip_sig(ppg,Fs_ppg_new)
        ecg=clip_sig(ecg,Fs_ecg_new)
        seq_out=clip_sig(labels,Fs_ppg_new)
        arr_pks=clip_sig(arr_pks,Fs_pks)
        # SSL_ECG Pre-processing
        
        
        ## HP filter
        ppg_filt=ppg_filter(seq_in.reshape(-1,1),Fs_ppg_new,
                                      show_plots=False)
    
        ## z-normalize
        ppg_mu,ppg_sigma=np.mean(ppg_filt.flatten()),np.std(ppg_filt.flatten())
        seq_in=((ppg_filt-ppg_mu)/ppg_sigma)

        # TODO: Form class_signal from class_id
        class_id=class_ids[class_name]
        class_signal=np.zeros((*seq_out.shape[:-1],n_classes))
        class_signal[:,class_id]=1
        seq_out=np.concatenate([seq_out,class_signal],axis=-1)

        
        #Find smooth HR by average RR-intervals in windows
        t_HR=np.arange(len(seq_in))/Fs_ppg_P2St
        RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks.flatten(),Fs_pks)
        t=np.cumsum(RR_ints_NU)+(RR_extreme_idx[0]/Fs_pks)
        f_interpol = sp.interpolate.interp1d(t, RR_ints_NU,'cubic',axis=0,
                            bounds_error=False,
                            fill_value=(RR_ints_NU[0],RR_ints_NU[-1]))
        RR_ints = f_interpol(t_HR)
        # tacho=tacho_filter(RR_ints, Fs_ecg_E2St,f_cutoff=0.5,order=None,
        #            show_plots=False,margin=0.15)
        
        #TODO: Replaced HR with RR here by removing **-1
        HR_interpol=(RR_ints)#.flatten()
        
        seq_in=np.concatenate([seq_in,HR_interpol],axis=-1)
        # plt.figure()
        # plt.plot(np.arange(len(arr_pks))/Fs_pks,arr_pks,t_HR,RR_ints,
        #          t_HR,tacho,'--')
        
        #Fragment intro windows
        in_cond,out=sliding_window_fragmentation([seq_in,seq_out],
                            ((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St,
                            bstep*step_s*Fs_ppg_P2St)
        
        
        # arr_pk_wins,t_HR=sliding_window_fragmentation([arr_pks.flatten(),
        #                 np.arange(len(arr_pks))/Fs_pks],
        #                 win_size=((bsize-1)*step_s+win_len_s)*Fs_pks,
        #                 step_size=bstep*step_s*Fs_pks,axes=None)
        # t_HR=np.mean(t_HR,axis=-1)
        # avgHRV=np.ones((len(arr_pk_wins),in_cond.shape[1]))
        # for j in range(len(arr_pk_wins)):
        #     RR_ints_NU, _=Rpeaks2RRint(arr_pk_wins[j],Fs_pks)
        #     avgHRV[j,:]=np.mean(RR_ints_NU)
        # #avgHRV=np.expand_dims(avgHRV,axis=-1)
        
        # assert len(in_cond)==len(avgHRV), 'Unexpected Dimensional mismatch'
        
        # #Testing iterated windowing for HRV module
        # if mode=='test_common':
        #     Dsplit_mask_clipped=np.concatenate(
        #                     [Dsplit_mask_dict['ecg'][class_name][:2],
        #                     Dsplit_mask_dict['ppg'][class_name][2:]],axis=0)
        #     Dsplit_mask_clipped=Dsplit_mask_clipped[:,idx_start:idx_stop]
        # else:
        #     #Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name]
        #     Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name][:,idx_start:idx_stop]
        
        Dsplit_mask_clipped=Dsplit_mask_dict['ppg'][class_name]#[:,idx_start:idx_stop]
        #plt.figure();plt.plot(Dsplit_mask_clipped.astype(int)[-1])
        assert bstep==1,'bstep MUST be 1 to avoid temporal misalignment of mask'
        
        Dsplit_mask_P2St=[get_windowed_mask(sel_mask,win_len=bsize,step=bstep)
                          for sel_mask in Dsplit_mask_clipped]
        Dsplit_mask_dict['P2St'][class_name]=np.stack(Dsplit_mask_P2St,
                                                     axis=0).astype(bool)
        Dsplit_mask_dict['Dspecs']['key_order'].append(class_name)
        
        # #check mask windowing
        # for q in range(3):
        #         print(np.mean(Dsplit_mask_clipped[q]))
        # print('\n ======================== \n')
        # for q in range(3):
        #         print(np.mean(Dsplit_mask_dict['P2St'][class_name][q]))
                
        #verify if dimensions match after processing
        assert Dsplit_mask_dict['P2St'][class_name].shape[1]==in_cond.shape[0]
        #plt.figure();plt.plot(Dsplit_mask_hrv.T);plt.legend(['train','val','test'])
        
        #in_cond=np.stack([in_cond,avgHRV],axis=-1)
        list_in.append(in_cond.astype(np.float32))# (N,2560,2)
        list_out.append(out.astype(np.float32)) #(N,2560,5)
        #list_HR.append(avgHRV) #(N,2560)

        
        if ((mode=='test') or (mode=='test_common')):
            # get selected test blocks

            test_bsel_mask=Dsplit_mask_dict['P2St'][class_name][2]
            #test_bsel_mask=test_bsel_mask|Dsplit_mask_dict['P2St'][class_name][0]
            #test_bsel_mask=test_bsel_mask|Dsplit_mask_dict['P2St'][class_name][1]
            
            #print(f'Selected {np.mean(test_bsel_mask)} ratio of windows from {class_name}')
            start_idxs,end_idxs=get_continous_wins(test_bsel_mask.astype(int))
            n_segments=len(start_idxs)
            #assert np.mean(end_idxs-start_idxs)==1, 'problem with test_bsize' 
            def temp_formatter(arr_list):
                arr_list=[arr_list[0][start_idxs[j]:end_idxs[j]]
                                          for j in range(n_segments)]
                arr_list=[sliding_window_defragmentation([arr],
                            ((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St,
                            bstep*step_s*Fs_ppg_P2St) for arr in arr_list]
                #TODO: Create non-overlapping test windows
                arr_list=[sliding_window_fragmentation([arr],
                            ((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St,
                            ((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St) 
                            for arr in arr_list]
                # arr_list=[sliding_window_fragmentation([arr],
                #             ((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St,
                #             int(0.25*((bsize-1)*step_s+win_len_s)*Fs_ppg_P2St)) 
                #             for arr in arr_list]
                return arr_list
            list_in=temp_formatter(list_in)
            list_out=temp_formatter(list_out)
                
                
            # test_bsel_idxs=np.arange(len(test_bsel_mask)
            #                              )[test_bsel_mask.astype(bool)]
            # for arr_list in [list_in,list_out]:
            #     arr_list[-1]=[arr_list[-1][bidx:(bidx+1)] 
            #                   for bidx in test_bsel_idxs]
            
    
    return list_in,list_out,Dsplit_mask_dict

        
#%%

def get_train_data(path,win_len_s=8,step_s=2,
                   bsize=5,bstep=1,Dsplit_mask_dict=None,musig_dict=None,
                   mode='E2St'):
    '''
    Use all files in the folder 'path' except the val_files and test_files
    '''
    #files=glob.glob(path+'**/*.pkl')
    files=[path+f'{class_name}/{class_name}.pkl' for class_name in class_ids]
    files=[fil.replace(os.sep,'/') for fil in files]
    #exclude val and test files
    #s3=set(files);s4=set(val_files+test_files)
    #files_2=list(s3.difference(s4))
    assert (Dsplit_mask_dict is not None), 'Dsplit_mask_dict is required'
    assert (musig_dict is not None), 'musig_dict is required'
    if mode=='eval':
        list_in,list_out=get_clean_eval_data(files,win_len_s,step_s,musig_dict)

    elif mode=='E2St':
        list_in,list_out,Dsplit_mask_dict=get_clean_E2St_data(files,
                                                win_len_s,step_s,
                                                Dsplit_mask_dict,musig_dict,
                                                bsize=bsize,bstep=bstep)
    elif mode=='P2St':
        list_in,list_out,Dsplit_mask_dict=get_clean_P2St_data(files,
                                                win_len_s,step_s,
                                                Dsplit_mask_dict,musig_dict,
                                                bsize=bsize,bstep=bstep)
    else:
        assert False, "mode MUST be in \{'eval','E2St','P2St'\}"
    return list_in,list_out,Dsplit_mask_dict


def get_test_data(path,win_len_s=8,step_s=2,
                  bsize=5,bstep=1,Dsplit_mask_dict=None,musig_dict=None,
                  mode='E2St'):
    
    files=glob.glob(path+'/*.pkl')
    files=[fil.replace(os.sep,'/') for fil in files]
    class_name=(files[0].split('/')[-2])
    assert (Dsplit_mask_dict is not None), 'Dsplit_mask_dict is required'
    assert (musig_dict is not None), 'musig_dict is required'
    
    if mode=='eval':
        list_in,list_out=get_clean_eval_data(files,win_len_s,step_s,musig_dict)
    elif mode=='E2St':
        list_in,list_out,Dsplit_mask_dict=get_clean_E2St_data(files,
                                                win_len_s,step_s,
                                                Dsplit_mask_dict,musig_dict,
                                                bsize=bsize,bstep=bstep,
                                                mode='test_common')
    elif mode=='P2St':
        list_in,list_out,Dsplit_mask_dict=get_clean_P2St_data(files,
                                                win_len_s,step_s,
                                                Dsplit_mask_dict,musig_dict,
                                                bsize=bsize,bstep=bstep,
                                                mode='test_common')
    else:
        assert False, "mode MUST be in \{'eval','E2St','P2St'\}"
    return list_in,list_out,Dsplit_mask_dict