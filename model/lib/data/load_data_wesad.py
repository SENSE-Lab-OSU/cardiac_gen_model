
import numpy as np
import scipy as sp
import pandas as pd
import glob
from scipy import signal as sig
import pickle
import neurokit2 as nk
import matplotlib.pyplot as plt
import os

MAX_PPG_VAL=1000 #1789
MAX_ECG_VAL=1
FIX_ECG_MEAN=0.5

Fs_ppg, Fs_ppg_new=64, 25
Fs_ecg, Fs_ecg_new=700, 100
test_ratio=0.1

class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(4,8))+list(range(15,16)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(2,4)))}
#{'S2': 0, 'S3': 1, 'S4': 2, 'S5': 3, 'S6': 4, 'S7': 5, 'S8': 6, 'S9': 7, 'S10': 8, 'S11': 9, 'S13': 10, 'S14': 11, 'S15': 12, 'S16': 13, 'S17': 14}
#class_ids={'S17':14}
n_classes=15#len(class_ids)

#%%
def Rpeak2HR(test_pks,win_len_s=8,step_s=2,Fs_pks=100):
    win_len=int(win_len_s*Fs_pks)+1 #added +1 =>have odd-order =>simpler group delay
    step=int(step_s*Fs_pks)
    HR_curve=[np.sum(test_pks[step*i:step*i+win_len])/(win_len/Fs_pks) for i in 
                range(int((len(test_pks)-win_len+1)/step))]
    HR_curve=(np.array(HR_curve)*60)
    #t_interpol=np.arange(len(test_pks))
    t=step*np.arange(int((len(test_pks)-win_len+1)/step))+int((win_len-1)/2)
    #t=t_interpol[int((win_len-1)/2):-int((win_len-1)/2):step]
    t_interpol=np.arange(t[0],t[-1]+1) #Only interpolate, no extrapolation
    #print(t.shape,HR_curve.shape)
    f_interpol = sp.interpolate.interp1d(t, HR_curve,'cubic',axis=0)
    HR_curve_interpol = f_interpol(t_interpol).astype(np.float32)
    
    #plt.figure();plt.plot(t,HR_curve,t_interpol,HR_curve_interpol,'r--')
    #Band-pass it till cutoff freq of 1/step_s
    #HR_curve_interpol=filtr_HR(HR_curve_interpol,cutoff=1/step_s)
    #plt.plot(t_interpol,HR_curve_interpol,'g--');plt.grid(True)
    return HR_curve_interpol,(t_interpol[0],t_interpol[-1])

def Rpeaks2RRint(arr_pks, Fs_pks=100):
    r_pk_locs_origin=np.arange(len(arr_pks))[arr_pks.flatten().astype(bool)]
    RR_ints=np.diff(r_pk_locs_origin/Fs_pks).reshape((-1,1))
    RR_extreme_idx=(r_pk_locs_origin[0],r_pk_locs_origin[-1])
    return RR_ints, RR_extreme_idx

#Filter ppg
def ppg_filter(X0,Fs_ppg,show_plot=False):
    '''
    Band-pass filter multi-channel PPG signal X0
    '''
    nyq=Fs_ppg/2
    n=10*Fs_ppg-1 # filter order
    b = sig.firls(n,np.array([0,0.3,0.5,4.5,5,nyq]),
                  np.array([0,0,1,1,0,0]),np.array([5,1,1]),nyq=nyq)
    X=np.zeros(X0.shape)
    for i in range(X0.shape[1]):
        #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
        X[:,i] = sig.filtfilt(b,[1],X0[:,i])
    
    if show_plot:
        w, h = sig.freqz(b)
        plt.figure()#;plt.subplot(211)
        plt.plot(w*(nyq/np.pi), 20 * np.log10(abs(h)), 'b')
        plt.grid(True)
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        plt.title('Digital filter frequency response')
    return X

def resample(y,Fs,up=1,down=1,show_plot=False):
    t_dur=len(y)/Fs
    num=len(y)*(up/down)
    assert num%1==0, f'no. of re-samples MUST be an integer but is {num} for this {t_dur} s. long signal.'
    num=int(num)
    if up==1: #only downsampling needed
        y_resampled = sig.decimate(y, down, ftype='fir',axis=0)
    else:
        y_resampled = sig.resample(y, num, axis=0)
        #y_resampled = sig.resample_poly(y, up, down, axis=0)
    
    if show_plot:
        x = np.linspace(0,t_dur,len(y),endpoint=False)
        x_resampled = np.linspace(0,t_dur,num,endpoint=False)
        plt.figure()
        plt.plot(x, y, 'b.-', x_resampled, y_resampled, 'r.--')
        #plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
        plt.legend(['original','resampled'], loc='best')
        plt.grid(True)
    return y_resampled.astype(np.float32)

def find_ecg_rpeaks(ecg,Fs,method='kalidas2017', show_plot=False):
    cleaned = nk.ecg_clean(ecg, sampling_rate=Fs, method=method)
    rpeak_train , rpeaks = nk.ecg_peaks(cleaned, 
                sampling_rate=Fs, method=method, correct_artifacts=True)
    rpeak_train=rpeak_train['ECG_R_Peaks'].values
    rpeaks=rpeaks['ECG_R_Peaks']
    #print(len(rpeaks))
    # detectors = Detectors(Fs)
    # rpeaks = detectors.christov_detector(ecg)
    # rpeak_train=np.zeros_like(ecg)
    # rpeak_train[rpeaks]=1
    if show_plot:
        plt.figure();plt.plot(ecg)
        plt.plot(np.arange(len(ecg))[rpeaks],ecg[rpeaks],'ro')
        plt.legend(['ECG','Detected_Rpeaks'])
        plt.title(f'Detected Rpeaks by {method} method')
        
    return rpeak_train.reshape(-1,1), rpeaks

def get_start_end_idxs(mode):
    if mode=='train':
        start_idx=lambda l: 0
        end_idx=lambda l: int((1-test_ratio)*l)
    elif mode=='test':
        start_idx=lambda l: int((1-test_ratio)*l)
        #start_idx=lambda l: 0
        #end_idx=lambda l: int((1-test_ratio)*l)
        end_idx=lambda l: l
    else:
        assert False, 'mode MUST be in \{test, train\}'
        
    return start_idx,end_idx
    
def get_clean_HR2R_data(files,win_len_s,step_s,Fs_pks,mode='train'):
    '''
    Extract data from 'clean' files
    '''
    #MAX_ECG_VAL=1
    list_in,list_arr_pks=[],[]
    start_idx,end_idx=get_start_end_idxs(mode)

    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            ecg = data['signal']['chest']['ECG'].astype(np.float32)
            #max_ecg=np.max(np.abs(ecg.flatten()))
            #if MAX_ECG_VAL<max_ecg: MAX_ECG_VAL=max_ecg*1
            ecg_resamp=resample(ecg,Fs_ecg,1,7)
            ecg_resamp/=MAX_ECG_VAL
            
            # find arr_pks
            arr_pks, _=find_ecg_rpeaks(ecg_resamp,Fs_ecg_new)
            HR_interpol,t_stamps=Rpeak2HR(arr_pks,win_len_s,step_s,Fs_pks)
            arr_pks=arr_pks[t_stamps[0]:t_stamps[1]+1]
            lenth=len(HR_interpol)
            
            #Form class_signal from class_id
            class_id=class_ids[(files[i].split('/')[-2])]
            class_signal=np.zeros((lenth,n_classes))
            class_signal[:,class_id]=1
            
            #append class_signal to input condition
            in_cond=np.concatenate([HR_interpol.reshape(-1,1),class_signal], 
                                   axis=-1)
            list_in.append(in_cond[start_idx(lenth):end_idx(lenth)].astype(np.float32))
            list_arr_pks.append(arr_pks[start_idx(lenth):end_idx(lenth)].astype(np.float32))
    return list_in,list_arr_pks

def get_clean_R2EP_data(files,mode='train',show_plot=False):
    '''
    Extract data from 'clean' files
    '''
    list_clean_ppg,list_arr_pks,dict_musig=[],[],{}
    start_idx,end_idx=get_start_end_idxs(mode)

    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            class_name=(files[i].split('/')[-2])
            data = pickle.load(file, encoding='latin1')
            ppg = data['signal']['wrist']['BVP'].astype(np.float32)
            ecg = data['signal']['chest']['ECG'].astype(np.float32)
            
            #Pre_process
            ecg=nk.ecg_clean(ecg.flatten(), sampling_rate=Fs_ecg, method="neurokit")
            ppg_filt=ppg_filter(ppg,Fs_ppg)
            ppg_resamp=resample(ppg_filt,Fs_ppg,25,64)
            ecg_resamp=resample(ecg.reshape(-1,1),Fs_ecg,1,7)
            assert len(ecg_resamp)==4*len(ppg_resamp), 'Check ppg and ecg resampling'
            lenth=min(int(len(ecg_resamp)/4),len(ppg_resamp))
            ppg_resamp,ecg_resamp=ppg_resamp[:lenth],ecg_resamp[:4*lenth]
            # find arr_pks
            arr_pks, _=find_ecg_rpeaks(ecg_resamp,Fs_ecg_new)
            
            #Normalize
            dict_musig[class_name]={'ppg':{'mu':np.mean(ppg_resamp.flatten()),
                                           'sig':np.std(ppg_resamp.flatten())},
                                    'ecg':{'mu':np.mean(ecg_resamp.flatten()),
                                           'sig':np.std(ecg_resamp.flatten())}}
            # ppg_resamp/=MAX_PPG_VAL
            # #list_means+=[np.mean(ecg_resamp)]
            # list_means+=[FIX_ECG_MEAN]
            # ecg_resamp-=list_means[-1]
            # ecg_resamp/=MAX_ECG_VAL
            ppg_resamp-=dict_musig[class_name]['ppg']['mu']
            ppg_resamp/=dict_musig[class_name]['ppg']['sig']
            ecg_resamp-=dict_musig[class_name]['ecg']['mu']
            ecg_resamp/=dict_musig[class_name]['ecg']['sig']

            # plot
            if show_plot: 
                plt.figure();plt.plot(np.arange(len(ecg_resamp))/Fs_ecg_new,ecg_resamp)
                plt.plot(np.arange(len(ppg_resamp))/Fs_ppg_new,ppg_resamp)
                #plt.plot(np.arange(len(ppg))/Fs_ppg,ppg_filt/MAX_PPG_VAL)
            # find arr_pks
            #arr_pks, _=find_ecg_rpeaks(ecg_resamp,Fs_ecg_new)
            
            #Form class_signal from class_id
            class_id=class_ids[class_name]
            class_signal=np.zeros((len(arr_pks),n_classes))
            class_signal[:,class_id]=1
            
            clean_ppg=np.concatenate([ppg_resamp,ecg_resamp.reshape((-1,4))], 
                                     axis=-1)
            arr_pks=np.concatenate([arr_pks,class_signal], axis=-1)
            list_clean_ppg+=[clean_ppg[start_idx(lenth):end_idx(lenth)].astype(np.float32)]
            list_arr_pks+=[arr_pks[4*start_idx(lenth):4*end_idx(lenth)].astype(np.float32)]

    return list_arr_pks,list_clean_ppg,dict_musig

#%%
def get_train_data(path,mode='HR2R',win_len_s=8,step_s=2,Fs_pks=100):
    '''
    Use all files in the folder 'path' except the val_files and test_files
    '''
    #files=glob.glob(path+'**/*.pkl')
    files=[path+f'{class_name}/{class_name}.pkl' for class_name in class_ids]
    files=[fil.replace(os.sep,'/') for fil in files]
    #exclude val and test files
    #s3=set(files);s4=set(val_files+test_files)
    #files_2=list(s3.difference(s4))
    if mode=='HR2R':
        list_HR,list_arr_pks=get_clean_HR2R_data(files,win_len_s,step_s,Fs_pks)
        return list_HR,list_arr_pks
    elif mode=='R2EP':
        list_arr_pks,list_clean_ppg,dict_musig=get_clean_R2EP_data(files)
        return list_arr_pks,list_clean_ppg,dict_musig
    else:
        assert False, "mode MUST be in \{'HR2R','R2EP'\}"


def get_test_data(path,mode='HR2R',win_len_s=8,step_s=2,Fs_pks=100):
    
    files=glob.glob(path+'/*.pkl')
    files=[fil.replace(os.sep,'/') for fil in files]
    class_name=(files[0].split('/')[-2])
    if mode=='HR2R':
        list_HR,list_arr_pks=get_clean_HR2R_data(files,win_len_s,step_s,
                                                 Fs_pks,mode='test')
        return list_HR[0],list_arr_pks[0]
    elif mode=='R2EP':
        list_arr_pks,list_clean_ppg,dict_musig=get_clean_R2EP_data(files,
                                                                   mode='test')
        return list_arr_pks[0],list_clean_ppg[0],dict_musig[class_name]
    else:
        assert False, "mode MUST be in \{'HR2R','R2EP'\}"
