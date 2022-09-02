
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
from .utils import filtr_HR, get_uniform_tacho, get_continous_wins

rng = np.random.default_rng(seed=1)
MAX_PPG_VAL=1000 #1789
MAX_ECG_VAL=1
FIX_ECG_MEAN=0.5

Fs_acc,Fs_ppg,Fs_ecg=32,64,700
Fs_acc_new,Fs_ppg_new,Fs_ecg_new=25,25,100
assert Fs_acc_new==Fs_ppg_new, 'Fs_acc_new must be equal to Fs_ppg_new'
factr=(Fs_ecg_new/Fs_ppg_new)
assert factr%1==0, '(Fs_ecg_new/Fs_ppg_new) must be an integer'
factr=int(factr)
win_len_s=8
step_s=2
bsize=13
bstep=1
test_ratio,val_ratio=0.1,0.1
test_bsize,val_bsize=bsize,bsize

class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(4,8))+list(range(15,16)))}
#class_ids={f'S{k}':v for v,k in enumerate(list(range(2,4)))}
#{'S2': 0, 'S3': 1, 'S4': 2, 'S5': 3, 'S6': 4, 'S7': 5, 'S8': 6, 'S9': 7, 'S10': 8, 'S11': 9, 'S13': 10, 'S14': 11, 'S15': 12, 'S16': 13, 'S17': 14}
#class_ids={'S17':14}
n_classes=15#len(class_ids)
n_stresses=5

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
    HR_curve_interpol=filtr_HR(HR_curve_interpol,cutoff=1/step_s)
    #plt.plot(t_interpol,HR_curve_interpol,'g--');plt.grid(True)
    #return HR_curve_interpol,(t_interpol[0],t_interpol[-1])
    return HR_curve_interpol,t_interpol/Fs_pks


def Rpeaks2RRint(arr_pks, Fs_pks=100):
    r_pk_locs_origin=np.arange(len(arr_pks))[arr_pks.flatten().astype(bool)]
    RR_ints=np.diff(r_pk_locs_origin/Fs_pks).reshape((-1,1))
    RR_extreme_idx=(r_pk_locs_origin[0],r_pk_locs_origin[-1])
    return RR_ints, RR_extreme_idx

#Filter ppg
def ppg_filter(X0,Fs_ppg,show_plots=False):
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
    
    if show_plots:
        w, h = sig.freqz(b)
        plt.figure()#;plt.subplot(211)
        plt.plot(w*(nyq/np.pi), 20 * np.log10(abs(h)), 'b')
        plt.grid(True)
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        plt.title('Digital filter frequency response')
    return X

def acc_filter(X0,Fs_acc,f_cutoff=10,show_plots=False):
    '''
    Band-pass filter multi-channel PPG signal X0
    '''
    nyq=Fs_acc/2
    assert f_cutoff<nyq,'f_cutoff should be < nyquist frequency for Fs'
    n=10*Fs_acc-1 # filter order
    b = sig.firls(n,np.array([0,f_cutoff-1,f_cutoff,nyq]),
                  np.array([1,1,0,0]),np.array([1,1]),nyq=nyq)
    X=np.zeros(X0.shape)
    for i in range(X0.shape[1]):
        #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
        X[:,i] = sig.filtfilt(b,[1],X0[:,i])
    
    if show_plots:
        w, h = sig.freqz(b)
        plt.figure()#;plt.subplot(211)
        plt.plot(w*(nyq/np.pi), 20 * np.log10(abs(h)), 'b')
        plt.grid(True)
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        plt.title('Digital filter frequency response')
    return X

def ecg_filter(X0,Fs=256.,f_stop=0.4,f_pass=1.,atte_stop=60,
                    ripp_pass=1,show_plots=False):
    '''
    Equivalent python filter for following MATLAB cheby2 filter
    highpass_filter = designfilt('highpassiir', 'StopbandFrequency', 0.4, 
                                 'PassbandFrequency', 0.8, ...
                                 'StopbandAttenuation', 60, 
                                 'PassbandRipple', 1, 'SampleRate', 256, 
                                 'DesignMethod', 'cheby2');
    '''
    #Fs=float(Fs)
    # High-pass stop-band frequency
    ws = f_stop/(Fs/2)
    # High-pass pass-band frequency
    wp = f_pass/(Fs/2)
    #design
    b,a = sig.iirdesign(wp, ws, ripp_pass, atte_stop, analog=False, 
                        ftype='cheby2',output='ba', fs=None)
    # orderFilter = sig.cheb2ord(wp, ws, gpass, gstop, analog=False, fs=256)
    
    #filter
    X=np.zeros(X0.shape)
    for i in range(X0.shape[1]):
        #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
        X[:,i] = sig.filtfilt(b,a,X0[:,i],axis=0)
        #X[:,i] = sig.lfilter(b, a, X0[:,i], axis=0, zi=None)
    if show_plots:
        w, h = sig.freqz(b,a)
        plt.figure()#;plt.subplot(211)
        plt.plot(w*(0.5*Fs/np.pi), 20 * np.log10(abs(h)), 'b')
        plt.grid(True)
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        plt.title('Digital filter frequency response')
        
        for j in range(X0.shape[1]):
            plt.figure()
            plt.plot(X0[:,j]);plt.plot(X[:,j],'--')
            plt.legend(['Original','Filtered'])
            plt.grid(True)
    return X

def tacho_filter(X0,Fs,f_cutoff=0.5,order=None,show_plots=False,margin=0.075):
    '''
    Band-pass filter multi-channel PPG signal X0
    '''
    nyq=Fs/2
    assert f_cutoff+margin<nyq,'f_cutoff should be < nyquist frequency for Fs'
    if order is None: order=50*Fs-1 # filter order
    b = sig.firls(order,np.array([0,f_cutoff,f_cutoff+margin,nyq]),
                  np.array([1,1,0,0]),np.array([1,1]),nyq=nyq)
    X=np.zeros(X0.shape)
    for i in range(X0.shape[1]):
        #X[:,i] = sig.convolve(X0[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
        X[:,i] = sig.filtfilt(b,[1],X0[:,i])
    
    if show_plots:
        w, h = sig.freqz(b)
        plt.figure()#;plt.subplot(211)
        plt.plot(w*(nyq/np.pi), 20 * np.log10(abs(h)), 'b')
        plt.grid(True)
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        plt.title('Digital filter frequency response')
    return X

def resample(y,Fs,up=1,down=1,show_plots=False):
    t_dur=len(y)/Fs
    num=len(y)*(up/down)
    assert num%1==0, f'no. of re-samples MUST be an integer but is {num} for this {t_dur} s. long signal.'
    num=int(num)
    if up==1: #only downsampling needed
        y_resampled = sig.decimate(y, down, ftype='fir',axis=0)
    else:
        y_resampled = sig.resample(y, num, axis=0)
        #y_resampled = sig.resample_poly(y, up, down, axis=0)
    
    if show_plots:
        x = np.linspace(0,t_dur,len(y),endpoint=False)
        x_resampled = np.linspace(0,t_dur,num,endpoint=False)
        plt.figure()
        plt.plot(x, y, 'b.-', x_resampled, y_resampled, 'r.--')
        #plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
        plt.legend(['original','resampled'], loc='best')
        plt.grid(True)
    return y_resampled.astype(np.float32)

def find_ecg_rpeaks(ecg,Fs,method='kalidas2017', show_plots=False):
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
    if show_plots:
        plt.figure();plt.plot(ecg)
        plt.plot(np.arange(len(ecg))[rpeaks],ecg[rpeaks],'ro')
        plt.legend(['ECG','Detected_Rpeaks'])
        plt.title(f'Detected Rpeaks by {method} method')
        
    return rpeak_train.reshape(-1,1), rpeaks

def sliding_window_fragmentation(tseries,win_size,step_size,axes=None):
    '''
    sliding_window_fragmentation along the axis dimension
    for eg.:
    tseries=list of (numpy array of shape [n_tsteps,vector_dim_at_every_tstep])
    '''
    assert type(tseries)==type([]), 'Input time-series should be a list of numpy arrays'
    if axes is None:
        axes=[0 for _ in range(len(tseries))]
    tot_len=tseries[0].shape[axes[0]]
    indices=np.stack([np.arange(i,i+(tot_len-win_size+1),step_size)
                        for i in range(win_size)],axis=1)
    #slid_arr=np.stack([tseries[i:i+(tot_len-win_size+1):step_size8]
    #                    for i in range(win_size)],axis=1)
    frag_tseries=[np.take(tseries[j],indices,axis=axes[j]) 
                    for j in range(len(tseries))]
    
    if len(frag_tseries)==1:
        return frag_tseries[0]
    else:
        return frag_tseries

def sliding_window_defragmentation(tseries,win_size,step_size): 
    '''
    sliding_window_defragmentation along the axis dimension
    for eg.:
    tseries=list of (numpy array of shape [n_wins,n_tsteps,vector_dim_at_every_tstep])
    '''
    assert type(tseries)==type([]), 'Input time-series should be a list of numpy arrays'
    assert win_size==tseries[0].shape[1], 'Incorrect window size of input tseries'
    defrag_tseries=[]
    for j in range(len(tseries)):
        if len(tseries[j].shape)==2: tseries[j]=np.expand_dim(tseries[j],-1)
        defrag_tseries.append(np.concatenate([tseries[j][0,...],
        tseries[j][1:,-step_size:,...].reshape([-1,*tseries[j].shape[2:]])],
        axis=0))
    
    if len(defrag_tseries)==1:
        return defrag_tseries[0]
    else:
        return defrag_tseries
    
def get_windows_at_peaks(pk_locs,y,w_pk=25,w_l=10,show_plots=False,n_wins=30,
                         ylims=[-25,25]):
    '''
    find and store peaks from signal y, centered at r_pk_locs
    pk_locs: R-peak locations
    y:signal to window
    w_pk= window length
    w_l= no.of samples in the window, to the left of peak.
    
    ?????????????????????????????????
    Should I take pk shapes looking from ECG perspective or correct for PTT?
    Right now I take from ECG persepective so that after PCA, the components
    will have implicit PTT.
    '''
    w_r=w_pk-w_l-1
    #remove terminal pk_locs
    #pk_locs=pk_locs[(pk_locs>=w_l) & (pk_locs<(len(y)-w_r))]
    
    # add peaks
    windowsPPG1=[y[pk_locs[i]-w_l:pk_locs[i]+w_r+1] \
                 for i in range(len(pk_locs))]
    windowsPPG1=np.array(windowsPPG1)
    print(windowsPPG1.shape)
    if show_plots:
        j,k=0,0#8
        #len(windowsPPG1)
        x_arr=np.stack(n_wins*[np.arange(w_pk)-w_l],axis=0).T
        
        # plt.figure()
        # plt.plot(x_arr,windowsPPG1[:n_wins].T,'r',alpha=0.1)
        # plt.grid(True)
        
        #plt.close('all')

        while (j+k)<(len(windowsPPG1)/n_wins):
            plt.figure();k=1
            while (k<=8):
                plt.subplot(4,2,k)
                cntr=j+k-1;
                #plt.plot(x_arr,windowsPPG1[cntr*n_wins:(cntr+1)*n_wins],'r',alpha=0.1)
                plt.plot(x_arr,windowsPPG1[cntr*n_wins:(cntr+1)*n_wins].T,'r',alpha=0.1)
                plt.grid(True)
                plt.ylim(ylims)
                plt.title(f'Peaks {cntr*n_wins}-{(cntr+1)*n_wins-1}')
                k=k+1
            j=j+8;
            
    assert len(windowsPPG1.shape)==2
    return windowsPPG1

def create_stress_signal(stres,Fs,dsampling_factor=None,t_interpol=None,
                         show_plots=False):
    time_ax=np.arange(len(stres))/Fs
    if t_interpol is not None:
        f_interpol = sp.interpolate.interp1d(time_ax, stres.flatten(),'linear',axis=0)
        stres_resamp=f_interpol(t_interpol)
    elif dsampling_factor is not None:
        stres_resamp,t_interpol=sliding_window_fragmentation([stres,time_ax],
                    win_size=dsampling_factor,step_size=dsampling_factor)
        #stres_resamp,_=stats.mode(stres_resamp,axis=1)
        stres_resamp=np.mean(stres_resamp,axis=1)
        t_interpol=np.mean(t_interpol,axis=-1)
    else:
        assert False, 'Either t_interpol or dsampling_factor must be given.'

    
    stres_resamp=np.round(stres_resamp).astype(int).flatten()
    stres_signal=np.zeros((len(stres_resamp),5))
    stres_signal[np.arange(len(stres_resamp)),stres_resamp]=1
    stres_mask=np.invert(stres_signal[:,0].astype(bool))
    #stres_signal=stres_signal[:,1:]
    if show_plots:
        plt.figure()
        plt.plot(time_ax,stres)
        plt.plot(t_interpol,stres_resamp,'r--')
        plt.legend(['Original','Resampled'])
        plt.xlabel('Time (S.)')
        plt.grid(True)
    return stres_signal, stres_mask


def ppg_ssqi_thresholding(ppg,Fs,show_plots=False,thres=None):
    # win_size=3s. step_size=1s. decides clip_times, Fs(ssqi) respectively
    ppg_wins=sliding_window_fragmentation([ppg],win_size=3*Fs,step_size=1*Fs)
    #mu,sig=np.mean(ppg),np.std(ppg)
    ssqi=[np.mean(((ppg_win-np.mean(ppg_win))/np.std(ppg_win))**3) 
          for ppg_win in ppg_wins]
    ssqi=np.array(ssqi)
    clip_times=[1,-1] #to center sliding window of 3s. correctly (in s.)
    
    #Upsample ssqi by repeating
    if thres is not None:
        ssqi_up=np.repeat(ssqi,Fs)
        thres_mask=((ssqi_up>=thres[0]) & (ssqi_up<=thres[1]))
    #ssqi_up=np.zeros_like(ppg)+np.nan
    #ssqi_up[::Fs]=ssqi
    #ssqi_up=interp1d()
    if show_plots:
        ppg=1*ppg[clip_times[0]*Fs:clip_times[1]*Fs]
        t_ppg=np.arange(len(ppg))/Fs
        t_ssqi=np.arange(len(ssqi))/1
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(t_ppg,ppg)
        if thres is not None:
            plt.plot(t_ppg[thres_mask],ppg[thres_mask],'.')
            plt.legend(['PPG','SSQI_thres_PPG'],loc='lower right')
        plt.title('PPG');plt.grid(True)
        plt.subplot(212,sharex=ax)
        plt.plot(t_ssqi,ssqi)
        plt.title('S_{SQI}');plt.grid(True)
        
    return thres_mask,clip_times

def acc_thresholding(acc,thres):
    '''
    acc and ppg should be at same Fs
    '''
    acc_mag=(((acc[:,0]**2+acc[:,1]**2+acc[:,2]**2))**0.5)/64
    thres_mask=((acc_mag>=thres[0]) & (acc_mag<=thres[1]))
    return thres_mask

def get_start_end_idxs(mode):
    #TODO: Remember to change this
    if mode=='train':
        start_idx=lambda l: 0
        end_idx=lambda l: int((1-test_ratio)*l)
    elif mode=='test':
        #start_idx=lambda l: int((1-0.4)*l)
        start_idx=lambda l: 0
        #end_idx=lambda l: int((1-test_ratio)*l)
        end_idx=lambda l: l
    else:
        assert False, 'mode MUST be in \{test, train\}'
        
    return start_idx,end_idx

def plot_STFT_tacho(tacho,tacho_synth,inv_HR_interpol,Fs_tacho):
    nfft=40*Fs_tacho
    f, t, Zxx=sp.signal.stft(tacho, fs=Fs_tacho, window='rectangular', nperseg=40*Fs_tacho, 
                   noverlap=35*Fs_tacho, nfft=nfft, detrend='constant', 
                   return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    vmax=np.nanpercentile(np.abs(Zxx).flatten(),99)
    mask=f<=0.6
    plt.figure()
    ax1=plt.subplot(311)
    plt.pcolormesh(t, f[mask], np.abs(Zxx[mask]), vmin=0,vmax=vmax,cmap='jet',shading='gouraud')
    plt.title('Tacho STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.grid(True)
    plt.colorbar()
    #plt.xlabel('Time [sec]')
    
    f, t, Zxx=sp.signal.stft(tacho_synth, fs=Fs_tacho, window='rectangular', 
                             nperseg=40*Fs_tacho, 
                   noverlap=35*Fs_tacho, nfft=nfft, detrend='constant', 
                   return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    plt.subplot(312,sharex=ax1,sharey=ax1)
    plt.pcolormesh(t, f[mask], np.abs(Zxx[mask]), vmin=0,vmax=vmax,cmap='jet', shading='gouraud')
    plt.title('Tacho_synth STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.grid(True)
    plt.colorbar()
    
    f, t, Zxx=sp.signal.stft(inv_HR_interpol, fs=Fs_tacho, window='rectangular', 
                             nperseg=40*Fs_tacho, 
                   noverlap=35*Fs_tacho, nfft=nfft, detrend='constant', 
                   return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    plt.subplot(313,sharex=ax1,sharey=ax1)
    plt.pcolormesh(t, f[mask], np.abs(Zxx[mask]), vmin=0,vmax=vmax,cmap='jet', shading='gouraud')
    plt.title('inv_HR STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.grid(True)
    plt.colorbar()
    return

def plot_periodograms(tacho,tacho_synth,cond,Fs_tacho=5):
    inv_HR_interpol=cond[:,0]
    win_size=32*Fs_tacho
    step_size=2*Fs_tacho
    noverlap=win_size-step_size
    nfft=win_size*1#8*Fs_tacho

    
    stres_signal=np.argmax(cond[:,1:6],axis=-1)
    # stres_wins,t_wins=sliding_window_fragmentation([stres_signal.flatten(),
    #         np.arange(len(stres_signal))/Fs_tacho],win_size=win_size,
    #         step_size=step_size,axes=None)
    # stres_resamp=np.round(np.mean(stres_wins,axis=-1)).astype(int)
    # t_stres=np.mean(t_wins,axis=-1)
    
    # # plot outputs directly
    # t_tacho=np.arange(len(tacho))/Fs_tacho
    # plt.figure()
    # plt.plot(t_tacho,inv_HR_interpol,t_tacho,tacho,'--',t_tacho,tacho_synth,':')
    # plt.legend(['avgHRV','HRV_{true}','HRV_{synth}'])
    # plt.grid(True)
    
    #plt.subplot(212,sharex=ax1,sharey=ax1)
    #plt.plot(t_tacho,tacho,t_tacho,tacho_synth,'--')
    #ll_list,ul_list=freq_bands[:-1],freq_bands[1:]
    #freq_masks=[((freqs>ll) & (freqs<=ul)) for ll,ul in zip(ll_list,ul_list)]
    signals=[tacho,tacho_synth,inv_HR_interpol]
    signal_names=['tacho_true','tacho_synth','tacho_GAN_input']
    marker_list=['b-','r--','g:']
    n_subplots=5
    stft_list=[]
    for j in range(len(signals)):
        f, t, Zxx=sp.signal.stft(signals[j], fs=Fs_tacho, window='rectangular', 
                nperseg=win_size,noverlap=noverlap,nfft=nfft,detrend='constant', 
                return_onesided=True, boundary='zeros', padded=True, axis=-1)
        stft_list.append(Zxx)
    t_interpol=t-0.2 #TODO: Why?
    t_interpol[t_interpol<0]=0
    stres_cat,_=create_stress_signal(stres_signal,Fs=Fs_tacho,
                                   t_interpol=t_interpol)
    
    fig=plot_categorical_peridogram(stft_list,f,stres_cat,signal_names,
                                marker_list)
    return fig

def plot_categorical_peridogram(stft_list,freqs,stres_cat,signal_names,
                                marker_list):
    '''
    Parameters
    ----------
    stft_list : [[freqs idx,time idx],... n_sig items]
        DESCRIPTION.
    freqs : freqs
        DESCRIPTION.
    stres_cat : [time idx, n_cat]
        DESCRIPTION.
    signal_names : [... n_sig items]
        DESCRIPTION.
    marker_list : [... n_sig items]
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    freq_mask=((freqs>0) & (freqs<=0.6))
    n_cat=stres_cat.shape[1]
    fig=plt.figure()
    for k in range(n_cat): 
        #stress_mask=(stres_resamp==k)
        stress_mask=(stres_cat[:,k].astype(bool))
        plt.subplot(n_cat,1,k+1)
        for j in range(len(stft_list)):
            stft_masked=stft_list[j][:,stress_mask]
            plt.plot(freqs[freq_mask],
            np.mean(np.abs(stft_masked[freq_mask])**2,axis=1),
            marker_list[j])
        #if (j==(len(signals)-1)):
        plt.title(f'Stress condition {k}')
        plt.grid(True)
        if k==0: plt.legend(signal_names)
        if k==int(n_cat/2): plt.ylabel('Periodogram Magnitude')
        if k==(n_cat-1): plt.xlabel('Frequency (Hz.)')
    return fig

def compare_spectrums(sig1,sig2,Fs,freq_range=[0,12.5]):
    err_msg=(f'sig1 shape {sig1.shape} is not equal to'
                 f' sig2 shape {sig2.shape}')
    assert sig1.shape==sig2.shape, err_msg
    rmse=lambda y,y_hat:np.sqrt(np.mean((y.reshape(-1)-y_hat.reshape(-1))**2))
    plt.figure();
    plt.subplot(211)
    plt.plot(sig1);plt.plot(sig2,'--')
    plt.legend(['sig1','sig2']);plt.grid(True)
    plt.title('RMSE = {:.4f}'.format(rmse(sig1,sig2)))
    plt.subplot(212)
    freqs=np.fft.rfftfreq(len(sig1),d=1/Fs)
    fmask=((freqs>freq_range[0]) & (freqs<=freq_range[1]))
    plt.plot(freqs[fmask],np.abs(np.fft.rfft(sig1))[fmask])
    plt.plot(freqs[fmask],np.abs(np.fft.rfft(sig2))[fmask],'--')
    plt.grid(True)
    plt.ylabel('Frequency (Hz.)')
    plt.xlabel('FFT Magnitude')
    return

def get_Dsplit_mask(sel_mask,stres_mask,win_len_s,step_size_s,Fs_out,
                    test_ratio,test_block_size,
                    val_ratio,val_block_size,sel_thres=0.85,stres_thres=0.99,
                    test_avail_sel_mask=None,test_sub_sel_mask=None):
    win_len,step_size=win_len_s*Fs_out,step_size_s*Fs_out
    
    # Window data
    sel_mask_wins,stres_mask_wins=sliding_window_fragmentation([sel_mask,
                                            stres_mask],win_len,step_size)
            
    # sqi based window selection
    #sel_mask_wins=arr_y[:,:,C_sel_mask] #(N,T)
    #arr_y=arr_y[:,:,C_sel_mask+1:] #Remove selection mask channel
    sel_mask_thres=(np.mean(sel_mask_wins,axis=1)>sel_thres)
    stres_mask_thres=(np.mean(stres_mask_wins,axis=1)>stres_thres)

    
    # Keep atleast ratio no. of windows
    n_olap_wins=int(np.ceil(win_len/step_size)-1)
    n_all_wins=np.sum(sel_mask_thres)
    n_test_blocks=int(np.ceil(((n_all_wins*test_ratio)/test_block_size)))
    n_val_blocks=int(np.ceil(((n_all_wins*val_ratio)/val_block_size)))
    
    if test_sub_sel_mask is None:
        sub_bsel_idxs_init=None
        n_test_init=0
    else:
        # Reduce already selected test blocks
        test_sub_bsel_mask=get_windowed_mask(test_sub_sel_mask,
                                win_len=test_block_size,step=test_block_size)
        n_test_init=np.sum(test_sub_bsel_mask)
        n_test_blocks-=n_test_init
        sub_bsel_idxs_init=np.arange(len(test_sub_bsel_mask)
                                     )[test_sub_bsel_mask.astype(bool)]
    
    # Remove stress=0 samples
    print(np.sum(sel_mask_thres))
    sel_mask_sample=(sel_mask_thres & stres_mask_thres)
    if test_avail_sel_mask is not None:
        test_avail_sel_mask=(test_avail_sel_mask & stres_mask_thres)
        
    print(np.sum(sel_mask_sample))
    test_sel_mask, sel_mask_sample=sample_continous_blocks(sel_mask_sample,
                                test_block_size,n_test_blocks,n_olap_wins,
                                test_avail_sel_mask,sub_bsel_idxs_init)
    print(np.sum(sel_mask_sample))
    bsel_wins=test_sel_mask+(sel_mask_sample[:,0].astype(int))
    assert (np.max(bsel_wins)<=1), 'block selection based split is overlapping'
    val_sel_mask, sel_mask_sample=sample_continous_blocks(sel_mask_sample,
                                val_block_size,n_val_blocks,0)
    print(np.sum(sel_mask_sample),np.sum((sel_mask_thres & 
                                        np.invert(stres_mask_thres))))
    #Add valid stress=0 samples back to training data
    sel_mask_sample=(sel_mask_sample | (sel_mask_thres & 
                                        np.invert(stres_mask_thres)))
    train_sel_mask=sel_mask_sample.flatten().astype(int)
    


    # verify block selection based split 
    bsel_wins=test_sel_mask+val_sel_mask+train_sel_mask
    assert (np.max(bsel_wins)<=2), 'block selection based split is overlapping'
    bsel_olap_wins=(bsel_wins==2).astype(int)
    n_bsel_wins,n_bsel_olap_wins=np.sum(bsel_wins),np.sum(bsel_olap_wins)
    check_up=(n_all_wins-n_bsel_wins)<=(n_test_blocks+n_test_init)*(2*n_olap_wins)
    check_down=(n_bsel_olap_wins)<=n_val_blocks*(2*n_olap_wins)
    assert (check_up and check_down), 'unexpected no. of blocks selection windows'
    #arr_y,out=arr_y[sel_mask_thres],out[sel_mask_thres]
    Dsplit_mask=np.stack([train_sel_mask,val_sel_mask,test_sel_mask],axis=0)
    plt.figure();plt.plot(Dsplit_mask.T);plt.legend(['train','val','test'])
    return Dsplit_mask

def get_windowed_mask(sel_mask,win_len=12,step=1):
    sel_mask_wins=sliding_window_fragmentation([sel_mask],win_len,step)
    sel_mask_windowed=(np.mean(sel_mask_wins,axis=1)==1).astype(int)
    return sel_mask_windowed
    
def sample_continous_blocks(sel_mask_thres,block_size,n_blocks,n_olap_wins=0,
                            avail_sel_mask=None,sub_bsel_idxs_init=None):
    
    if avail_sel_mask is None: avail_sel_mask=(sel_mask_thres*1).astype(bool)
    
    step_size,sel_thres,lenth=block_size*1,1,len(sel_mask_thres)
    #form continous block selection mask
    bsel_mask_wins=sliding_window_fragmentation([avail_sel_mask],
                                          block_size,step_size)
    bsel_mask_thres=(np.mean(bsel_mask_wins,axis=1)==sel_thres).flatten()
    #bsel_idxs=np.arange(lenth-block_size+1)[bsel_mask_thres]
    bsel_idxs=np.arange(len(bsel_mask_thres))[bsel_mask_thres]
    assert len(bsel_idxs)>0, 'No continous blocks :('
        
    bsel_idxs=np.sort(bsel_idxs)
    if n_blocks>len(bsel_idxs):
        print(f'Not enough continous blocks. Will select all available and '
              f'try to proceed with {n_blocks-len(bsel_idxs)} fewer blocks.')
        sub_bsel_idxs=bsel_idxs*1
    else:
        print('Enough continous blocks. Will select uniformly.')
        # Choose uniform indexing style
        # max_step=int(len(bsel_idxs)/n_blocks)
        # sub_bsel_idxs=bsel_idxs[:max_step*n_blocks:max_step]
        # sub_bsel_idxs=bsel_idxs[np.round(np.linspace(0,len(bsel_idxs)-1,
        #                                              n_blocks)).astype(int)]
        sub_bsel_idxs=rng.permutation(bsel_idxs)[:n_blocks]
        assert len(sub_bsel_idxs)==n_blocks, 'uniform election incorrect'

    
    # append existing sub_sel_mask's idx to bidx
    if sub_bsel_idxs_init is not None: 
        sub_bsel_idxs=np.concatenate([sub_bsel_idxs,sub_bsel_idxs_init])
    sub_sel_mask=np.zeros(lenth)
    
    # Add sub_sel_idxs blocks to sub-dataset
    for bidx in sub_bsel_idxs:
        idx=(bidx*step_size)
        sub_sel_mask[idx:idx+block_size]=1
        #TODO:Check fails when sub_bsel_idxs_init is not None. Removed for now
        #assert np.mean(avail_sel_mask[idx:idx+block_size].flatten())==1
    
    # Remove sub_sel_idxs blocks from all available
    for bidx in sub_bsel_idxs:
        idx=(bidx*step_size)
        if idx-n_olap_wins<0: #start edge case
            sel_mask_thres[:idx+block_size+n_olap_wins]=0
        elif idx+block_size+n_olap_wins>lenth: #end edge case
            sel_mask_thres[idx-n_olap_wins:]=0
        else:
            #print(np.mean(sel_mask_thres[idx:idx+block_size].flatten()),bidx)
            sel_mask_thres[idx-n_olap_wins:idx+block_size+n_olap_wins]=0
    
    return sub_sel_mask.astype(int), sel_mask_thres
#%% For CG_HRV

def get_clean_HR2R_data(files,win_len_s_avg,step_s_avg,Fs_tacho,Dsplit_mask_dict,
                        mode='train'):
    '''
    Extract data from 'clean' files
    '''
    #MAX_ECG_VAL=1
    list_in,list_out=[],[]
    #start_idx,end_idx=get_start_end_idxs(mode)
    clip_times=[1,-1]
    clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]

    Dsplit_mask_dict['hrv']={}
    Dsplit_mask_dict['hrv_clipped']={}
    Dsplit_mask_dict['Dspecs']['key_order']=[]
    
    for i in range(len(files)):
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        class_name=(files[i].split('/')[-2])
        print(f'{class_name} \n')
        

        ecg = data['signal']['chest']['ECG'].astype(np.float32)
        stres = data['label'].astype(np.float32)#.reshape(-1,1)
        
        # Clean-up signals
        ecg=nk.ecg_clean(ecg.flatten(), sampling_rate=Fs_ecg, method="neurokit")
        stres[stres>=5] = 0. #zero-out any meaningless label >=5
        
        ecg=resample(ecg,Fs_ecg,1,7)
        arr_pks, rpeaks=find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        Fs_pks=Fs_ecg_new
        
        # CLip signals 1s on both sides to match with R2S data
        arr_pks=clip_sig(arr_pks,Fs_pks)
        stres=clip_sig(stres,Fs_ecg)
        
        # Find uniform Tachogram at Fs_tacho
        RR_ints_NU, RR_extreme_idx=Rpeaks2RRint(arr_pks,Fs_pks)
        
        # Uniformly interpolate
        #t_interpol,RR_ints=get_leading_tacho(RR_ints_NU,fs=Fs_tacho)
        t_interpol,RR_ints=get_uniform_tacho(RR_ints_NU,fs=Fs_tacho,
                                             t_bias=RR_extreme_idx[0]/Fs_pks)
        tacho=tacho_filter(RR_ints, Fs_tacho,f_cutoff=0.5,order=100-1,
                           show_plots=False,margin=0.15)
        avgHRV_interpol=(tacho_filter(RR_ints,Fs_tacho,f_cutoff=0.125,
                        order=200-1,show_plots=False,margin=0.075)).flatten()
        #adjust t_interpol for RR_extreme_idx. Now included in get_tacho()
        #t_interpol=t_interpol+(RR_extreme_idx[0]/Fs_pks)
        
        #TODO: May also need to return arr_pks for comparison while testing
        
        # #Find smooth tacho by average RR-intervals in windows
        # arr_pk_wins,t_HR=sliding_window_fragmentation([arr_pks.flatten(),
        #     np.arange(len(arr_pks))/Fs_pks],win_size=Fs_pks*win_len_s_avg,
        #     step_size=Fs_pks*step_s_avg,axes=None)
        # t_HR=np.mean(t_HR,axis=-1)
        # avgHRV=np.ones(len(arr_pk_wins))
        # for j in range(len(arr_pk_wins)):
        #     RR_ints_NU, _=Rpeaks2RRint(arr_pk_wins[j],Fs_pks)
        #     avgHRV[j]=np.mean(RR_ints_NU)

        # # find nearest 2s divisible extreme times on t_interpol grid.
        # # clip signals and masks accordingly
        # # take block average of masks and window signals
        # # verify dimensional match
        # f_interpol = sp.interpolate.interp1d(t_HR, avgHRV,'cubic',axis=0)
        
        t_HR=t_interpol
        # find nearest 2s divisible extreme times on t_interpol grid.
        t_start=np.ceil(t_HR[0]).astype(int)
        t_start+=(step_s_avg-(t_start%step_s_avg))
        t_stop=np.floor(t_HR[-1]).astype(int)
        t_stop-=(t_stop%step_s_avg)
        idx_start=int(t_start/step_s_avg)
        idx_stop=int((t_stop-win_len_s_avg)/step_s_avg)+1
        
        clip_tacho_mask=((t_interpol>=t_start) & (t_interpol<=t_stop))
        t_interpol,tacho=t_interpol[clip_tacho_mask],tacho[clip_tacho_mask]
        avgHRV_interpol=avgHRV_interpol[clip_tacho_mask]
        #avgHRV_interpol = f_interpol(t_interpol)
        # avgHRV_interpol=(tacho_filter(tacho,Fs_tacho,f_cutoff=0.075,
        #                                show_plots=False)).flatten()

        #compare_spectrums(avgHRV_interpol,avgHRV_interpol_2,Fs_tacho,[0,0.2])
        #compare_spectrums(tacho.flatten(),avgHRV_interpol_2,Fs_tacho,[0,0.6])

        #tacho=tacho.flatten()
        stres_signal_ecg,_ = create_stress_signal(stres,Fs=Fs_ecg,
                                                  t_interpol=t_interpol)
        lenth=len(t_interpol)
        #Form class_signal from class_id
        class_id=class_ids[class_name]
        class_signal=np.zeros((lenth,n_classes))
        class_signal[:,class_id]=1
        
        #append class_signal to input condition
        in_cond=np.concatenate([avgHRV_interpol.reshape(-1,1),stres_signal_ecg,
                                class_signal],axis=-1)


        #TODO: Important win_lenth and step_size assumptions here
        in_cond,tacho=sliding_window_fragmentation([in_cond,tacho],
                            ((test_bsize-1)*step_s_avg+win_len_s_avg)*Fs_tacho,
                            step_s_avg*Fs_tacho)
        #TODO: Testing iterated windowing for HRV module
        if mode=='test_common':
            Dsplit_mask_clipped=np.concatenate(
                            [Dsplit_mask_dict['ecg'][class_name][:2],
                            Dsplit_mask_dict['ppg'][class_name][2:]],axis=0)
            Dsplit_mask_clipped=Dsplit_mask_clipped[:,idx_start:idx_stop]
        else:
            #Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name]
            Dsplit_mask_clipped=Dsplit_mask_dict['ecg'][class_name][:,idx_start:idx_stop]
        
        #Dsplit_mask_dict['hrv_clipped'][class_name]=Dsplit_mask_clipped.astype(bool)
        
        Dsplit_mask_hrv=[get_windowed_mask(sel_mask,win_len=test_bsize,step=1)
                          for sel_mask in Dsplit_mask_clipped]
        Dsplit_mask_dict['hrv'][class_name]=np.stack(Dsplit_mask_hrv,
                                                     axis=0).astype(bool)
        Dsplit_mask_dict['Dspecs']['key_order'].append(class_name)
        
        # #TODO: For debugging
        # def check_mask(mask,mask_hrv,show_plots=False):
        #     if show_plots:
        #         plt.figure();plt.plot(mask.astype(int))
        #         plt.plot(mask_hrv.astype(int))
        #     print(np.mean(mask),np.mean(mask_hrv))
        #     return
        # check_mask(Dsplit_mask_clipped[0],Dsplit_mask_hrv[0])
        # check_mask(Dsplit_mask_clipped[1],Dsplit_mask_hrv[1])
        
        #verify if dimensions match after processing
        assert Dsplit_mask_dict['hrv'][class_name].shape[1]==tacho.shape[0]
        #plt.figure();plt.plot(Dsplit_mask_hrv.T);plt.legend(['train','val','test'])
        
        list_in.append(in_cond.astype(np.float32))
        list_out.append(tacho.astype(np.float32))

        
        if ((mode=='test') or (mode=='test_common')):
            # get selected test blocks
            test_bsel_mask=Dsplit_mask_dict['hrv'][class_name][2]
            start_idxs,end_idxs=get_continous_wins(test_bsel_mask.astype(int))
            n_segments=len(start_idxs)
            for arr_list in [list_in,list_out]:
                arr_list[-1]=[arr_list[-1][start_idxs[j]:end_idxs[j]] 
                              for j in range(n_segments)]
            
            test_bsel_idxs=np.arange(len(test_bsel_mask)
                                          )[test_bsel_mask.astype(bool)]
            for arr_list in [list_in,list_out]:
                arr_list[-1]=[arr_list[-1][bidx:(bidx+1)] 
                              for bidx in test_bsel_idxs]
    
    return list_in,list_out,Dsplit_mask_dict
    
#%% For CG_Morph
def get_clean_R2S_data(files,win_len_s=8,step_size_s=2, Dsplit_mask_dict=None,
                       mode='train',show_plots=False):
    '''
    Extract data from 'clean' files
    '''
    input_dict={'ppg':[],'ecg':[]}
    output_dict={'ppg':[],'ecg':[]}
    musig_dict={}
    
    if Dsplit_mask_dict is None:
        Dsplit_mask_dict={'ppg':{},'ecg':{},'Dspecs':{'key_order':[],
                                'win_len':win_len_s,'step_size':step_size_s}}
        form_Dsplit_mask_dict=True
    else:
        form_Dsplit_mask_dict=False
        Dsplit_mask_dict['Dspecs']['key_order']=[] #reform key_order
        

    for i in range(len(files)):
        class_name=(files[i].split('/')[-2])
        print(f'Processing subject {class_name}')
        #Load signals
        with open(files[i], 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        acc = 1+data['signal']['wrist']['ACC'].astype(np.float32)#+1 for [0,63]->[1,64]
        ppg = data['signal']['wrist']['BVP'].astype(np.float32)
        ecg = data['signal']['chest']['ECG'].astype(np.float32)
        stres = data['label'].astype(np.float32)#.reshape(-1,1)
        
        #Clean/filter signals
        acc_filt=acc_filter(acc,Fs_acc)
        ppg_filt=ppg_filter(ppg,Fs_ppg)
        ecg_filt=nk.ecg_clean(ecg.flatten(),sampling_rate=Fs_ecg,method="neurokit")
        #np.isfinite(ecg_filt).all()
        ecg_filt=ecg_filter(ecg_filt.reshape(-1,1),Fs=Fs_ecg,show_plots=False)
        stres[stres>=5] = 0. #zero-out any meaningless label >=5
        
        #Resample at desired frequencies
        acc_resamp=resample(acc_filt,Fs_acc,25,32)
        ppg_resamp=resample(ppg_filt,Fs_ppg,25,64)
        ecg_resamp=resample(ecg_filt.reshape(-1,1),Fs_ecg,1,7)
        assert len(ecg_resamp)==factr*len(ppg_resamp), 'Check ppg and ecg resampling'
        
        acc,ppg,ecg=acc_resamp,ppg_resamp,ecg_resamp
        acc_mag=(((acc[:,0]**2+acc[:,1]**2+acc[:,2]**2))**0.5)/64
        arr_pks_ecg, rpeaks=find_ecg_rpeaks(ecg,Fs_ecg_new,show_plots=False)
        
        #Normalize
        musig_dict[class_name]={'ppg':{'mu':np.mean(ppg.flatten()),
                                       'sigma':np.std(ppg.flatten())},
                                'ecg':{'mu':np.mean(ecg.flatten()),
                                       'sigma':np.std(ecg.flatten())}}
        ppg-=musig_dict[class_name]['ppg']['mu']
        ppg/=musig_dict[class_name]['ppg']['sigma']
        ecg-=musig_dict[class_name]['ecg']['mu']
        ecg/=musig_dict[class_name]['ecg']['sigma']
        
        if show_plots:
            t_ppg=np.arange(len(ppg))/Fs_ppg_new
            t_ecg=np.arange(len(ecg))/Fs_ecg_new
            t_acc=np.arange(len(acc))/Fs_acc_new
            
            plt.figure()
            ax1=plt.subplot(411)
            plt.plot(t_acc,acc[:,0],'b-',t_acc,acc[:,1],'r--',t_acc,acc[:,2],'g:')
            plt.legend(['X','Y','Z'],loc='lower right');plt.title('Acc');plt.grid(True)
            plt.subplot(412,sharex=ax1)
            plt.plot(t_acc,acc_mag)
            plt.title('Acc Magnitude in g');plt.grid(True)
            #plt.ylim([-0.5,1.5])
            ax3=plt.subplot(413,sharex=ax1)
            plt.plot(t_ppg,ppg)
            plt.title('PPG');plt.grid(True)
            plt.subplot(414,sharex=ax1)
            plt.plot(t_ecg,ecg)
            plt.title('ECG');plt.grid(True)
            plt.suptitle(class_name)
            
            
        #Create a signal from windows to check morph profiles
        stres_signal_ecg, stres_mask_ecg = create_stress_signal(stres,
                                                Fs=Fs_ecg,dsampling_factor=7)
        stres_signal_ppg, stres_mask_ppg = create_stress_signal(stres,
                                                Fs=Fs_ecg,dsampling_factor=7*4)
        
        #Form class_signal from class_id
        class_id=class_ids[class_name]
        class_signal=np.zeros((len(arr_pks_ecg),n_classes))
        class_signal[:,class_id]=1
        
        #Get pks at Fs_ppg_new

        r_pk_locs=np.floor(rpeaks/factr).astype(int)
        arr_pks_ppg=np.zeros((int(len(arr_pks_ecg)/factr),1)).astype(np.float32)
        arr_pks_ppg[r_pk_locs]=1
        
        #get sqi mask (signal quality index). Will need 1s. clipping from start and end each.
        thres_mask_ssqi,clip_times=ppg_ssqi_thresholding(ppg,Fs=Fs_ppg_new,
                                                    show_plots=False,thres=[-1,0])
        clip_sig=lambda sig,Fs:sig[clip_times[0]*Fs:clip_times[1]*Fs]
        acc=clip_sig(acc,Fs_acc_new)
        thres_mask_acc=acc_thresholding(acc,thres=[1-0.05,1+0.05])
        sqi_mask_ppg=(thres_mask_acc & thres_mask_ssqi)#.astype(np.float32)#.reshape(-1,1)

        
        ppg=clip_sig(ppg,Fs_ppg_new)
        arr_pks_ppg=clip_sig(arr_pks_ppg,Fs_ppg_new)
        stres_signal_ppg=clip_sig(stres_signal_ppg,Fs_ppg_new)
        stres_mask_ppg=clip_sig(stres_mask_ppg,Fs_ppg_new)
        
        ecg=clip_sig(ecg,Fs_ecg_new)
        arr_pks_ecg=clip_sig(arr_pks_ecg,Fs_ecg_new)
        stres_mask_ecg=clip_sig(stres_mask_ecg,Fs_ecg_new)
        stres_signal_ecg=clip_sig(stres_signal_ecg,Fs_ecg_new)
        class_signal=clip_sig(class_signal,Fs_ecg_new)
        
        #thres_mask=(thres_mask_ssqi).astype(np.float32)#.reshape(-1,1)
        
        #TODO: changed here to have more useful data
        #sel_mask_ecg=stres_mask_ecg.astype(np.float32).reshape(-1,1)
        #sel_mask_ppg=(sqi_mask_ppg & stres_mask_ppg).astype(np.float32).reshape(-1,1)
        sel_mask_ecg=np.ones_like(stres_mask_ecg).astype(np.float32).reshape(-1,1)
        sel_mask_ppg=sqi_mask_ppg.astype(np.float32).reshape(-1,1)
        stres_mask_ecg=stres_mask_ecg.astype(np.float32).reshape(-1,1)
        stres_mask_ppg=stres_mask_ppg.astype(np.float32).reshape(-1,1)
        
        
        cond_ecg=np.concatenate([arr_pks_ecg,stres_signal_ecg,class_signal],axis=-1)
        cond_ppg=np.concatenate([arr_pks_ppg,stres_signal_ppg,
                                 class_signal[::factr]],axis=-1)

        
        
        if show_plots:
            t_ppg=clip_sig(t_ppg,Fs_ppg_new)
            #Post-processing ppg for verification
            sel_mask_wins=sliding_window_fragmentation([sel_mask_ppg.flatten()]
                    ,win_size=Fs_ppg_new*8,step_size=Fs_ppg_new*8,axes=None)
            sel_mask_thres=(np.mean(sel_mask_wins,axis=1)>0.85).astype(int)
            print(np.sum(sel_mask_thres)/len(sel_mask_thres))
            sel_mask_thres_up=np.repeat(sel_mask_thres,Fs_ppg_new*8,axis=0)
            
            #Clip to integral multiples of window length
            t_ppg=t_ppg[:len(t_ppg)-len(t_ppg)%(Fs_ppg_new*8)]
            ppg_clipped=ppg[:len(ppg)-len(ppg)%(Fs_ppg_new*8)]
            arr_pks_ppg=arr_pks_ppg[:len(ppg)-len(ppg)%(Fs_ppg_new*8)]
            ax3.plot(t_ppg[sel_mask_thres_up.astype(bool)],
                     ppg_clipped[sel_mask_thres_up.astype(bool)],'r.')
            
            arr_pks_ppg[np.invert(sel_mask_thres_up.astype(bool))]=0 #Zero out unselected rpeaks
            ax3.plot(t_ppg,arr_pks_ppg,'g')
            ax3.legend(['PPG','SQI_thres_PPG','R-peaks'],loc='lower right')
        
        if form_Dsplit_mask_dict:
            #get Data split masks
            Dsplit_mask_ppg=get_Dsplit_mask(sel_mask_ppg,stres_mask_ppg,win_len_s,step_size_s,
                            Fs_ppg_new,test_ratio=test_ratio,test_block_size=test_bsize,
                            val_ratio=val_ratio,val_block_size=val_bsize,sel_thres=0.85)
            Dsplit_mask_dict['ppg'][class_name]=Dsplit_mask_ppg.astype(bool)
            print(f'Selected '
                  f'{np.sum(Dsplit_mask_ppg.flatten())/Dsplit_mask_ppg.shape[-1]} ' 
                  f' ratio of windows from class {class_name}')
            
            # Do test_ecg selection based on test_ppg selection
            test_avail_sel_mask=np.invert(np.sum(Dsplit_mask_ppg,axis=0)
                                          .astype(bool).reshape(-1,1))
            test_sub_sel_mask=Dsplit_mask_ppg[2]
            
            Dsplit_mask_ecg=get_Dsplit_mask(sel_mask_ecg,stres_mask_ecg,win_len_s,step_size_s,
                            Fs_ecg_new,test_ratio=test_ratio,test_block_size=test_bsize,
                            val_ratio=val_ratio,val_block_size=val_bsize,sel_thres=0.85,
                            test_avail_sel_mask=test_avail_sel_mask,
                            test_sub_sel_mask=test_sub_sel_mask)
            Dsplit_mask_dict['ecg'][class_name]=Dsplit_mask_ecg.astype(bool)
            print(f'Selected '
                  f'{np.sum(Dsplit_mask_ecg.flatten())/Dsplit_mask_ecg.shape[-1]} ' 
                  f' ratio of windows from class {class_name}')
        
        Dsplit_mask_dict['Dspecs']['key_order'].append(class_name)
        
        #TODO: Chnaged here to remove stress condition for morph module
        #remove stress column. Needed for Dsplit mask but not for morph model
        
        # cond_ecg=np.concatenate([cond_ecg[:,0:1],cond_ecg[:,6:]],axis=-1)
        # cond_ppg=np.concatenate([cond_ppg[:,0:1],cond_ppg[:,6:]],axis=-1)

        cond_ppg,ppg=sliding_window_fragmentation([cond_ppg,ppg],
                                win_len_s*Fs_ppg_new,step_size_s*Fs_ppg_new)
        cond_ecg,ecg=sliding_window_fragmentation([cond_ecg,ecg],
                                win_len_s*Fs_ecg_new,step_size_s*Fs_ecg_new)
        
        input_dict['ppg']+=[cond_ppg.astype(np.float32)]
        output_dict['ppg']+=[ppg.astype(np.float32)]
        input_dict['ecg']+=[cond_ecg.astype(np.float32)]
        output_dict['ecg']+=[ecg.astype(np.float32)]
        
        if mode=='test':
            for k in ['ppg','ecg']:
                # get selected test blocks
                test_bsel_mask=get_windowed_mask(Dsplit_mask_dict[k]
                            [class_name][2],win_len=test_bsize,step=test_bsize)
                test_bsel_idxs=np.arange(len(test_bsel_mask)
                                              )[test_bsel_mask.astype(bool)]
                for arr_dict in [input_dict,output_dict]:
                    arr_dict[k][-1]=[arr_dict[k][-1]
                                          [bidx*test_bsize:(bidx+1)*test_bsize] 
                                          for bidx in test_bsel_idxs]
        elif mode=='test_common':
            # get selected test blocks
            test_bsel_mask=get_windowed_mask(Dsplit_mask_dict['ppg']
                        [class_name][2],win_len=test_bsize,step=test_bsize)
            test_bsel_idxs=np.arange(len(test_bsel_mask)
                                          )[test_bsel_mask.astype(bool)]
            for k in ['ppg','ecg']:
                for arr_dict in [input_dict,output_dict]:
                    arr_dict[k][-1]=[arr_dict[k][-1]
                                          [bidx*test_bsize:(bidx+1)*test_bsize] 
                                          for bidx in test_bsel_idxs]
    
    return input_dict,output_dict,musig_dict,Dsplit_mask_dict

        
#%%
def get_train_data(path,mode='HR2R',win_len_s=8,step_s=2,Fs_tacho=5,
                   Dsplit_mask_dict=None):
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
        assert (Dsplit_mask_dict is not None), 'Dsplit_mask_dict is required'
        list_avgHRV,list_HRV,Dsplit_mask_dict=get_clean_HR2R_data(files,
                                                win_len_s,step_s,
                                                Fs_tacho,Dsplit_mask_dict)
        return list_avgHRV,list_HRV,Dsplit_mask_dict
    elif mode=='R2S':
        input_dict,output_dict,musig_dict,Dsplit_mask_dict=get_clean_R2S_data(
            files,win_len_s,step_s,Dsplit_mask_dict)
        return input_dict,output_dict,musig_dict,Dsplit_mask_dict
    else:
        assert False, "mode MUST be in \{'HR2R','R2S'\}"


def get_test_data(path,mode='HR2R',win_len_s=8,step_s=2,Fs_tacho=5,
                  Dsplit_mask_dict=None):
    
    files=glob.glob(path+'/*.pkl')
    files=[fil.replace(os.sep,'/') for fil in files]
    class_name=(files[0].split('/')[-2])
    assert (Dsplit_mask_dict is not None), 'Dsplit_mask_dict is required'

    
    if mode=='HR2R':
        list_avgHRV,list_HRV,Dsplit_mask_dict=get_clean_HR2R_data(files,
                                        win_len_s,step_s,Fs_tacho,
                                        Dsplit_mask_dict,mode='train')
        return list_avgHRV,list_HRV,Dsplit_mask_dict
    elif mode=='R2S':
        input_dict,output_dict,musig_dict,Dsplit_mask_dict=get_clean_R2S_data(
                files,win_len_s,step_s,Dsplit_mask_dict,mode='train')
        
        # input_dict['ppg']=input_dict['ppg'][0]
        # input_dict['ecg']=input_dict['ecg'][0]
        # output_dict['ppg']=output_dict['ppg'][0]
        # output_dict['ecg']=output_dict['ecg'][0]
        return input_dict,output_dict,musig_dict,Dsplit_mask_dict
    else:
        assert False, "mode MUST be in \{'HR2R','R2S'\}"