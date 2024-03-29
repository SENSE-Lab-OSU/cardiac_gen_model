# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:53:22 2018

@author: agarwal.270a
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil, errno
import glob
from tensorflow.keras import Model, layers
AUTOTUNE = tf.data.experimental.AUTOTUNE
import scipy.signal as sig
import scipy as sp
from sklearn.metrics import confusion_matrix
import pandas as pd

eps=1e-7

def flip(x,y):
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    return x,y

def make_plot(z_hat,z,y,x_hat,x):
    avg_y=np.mean(y)
    freq=np.fft.fftfreq(x.shape[0])*25
    spect=np.abs(z)
    plt.figure()
    plt.subplot(211);plt.plot(np.array(2*[avg_y]),
                              np.array([np.min(spect),np.max(spect)]),'k')
    plt.plot(freq,spect,'b');plt.plot(freq,np.abs(z_hat),'r--')
    plt.legend(['True avg freq.','input FFT','Predicted Sparse FFT'])
    plt.title('Signal Spectrum');plt.grid(True)
    
    plt.subplot(212);plt.plot(np.real(x),'b');plt.plot(np.real(x_hat),'r--')
    plt.legend(['True Signal','Reconstructed Signal'])
    plt.title('Time domain Signal');plt.grid(True)

def set_plot_font(SMALL_SIZE = 6,MEDIUM_SIZE = 9,BIGGER_SIZE = 10):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return
    

def make_x2xeval_plots(data,model,log_prefix,expid_list,label_list,
                       confusion_keys,title,save_fig=True,
                       marker_list=['-o','--o',':o']):
    assert len(label_list)==len(marker_list),'len(label_list) must be = len(marker_list)'
    #some settings of matplotlib backend
    set_plot_font(8,9,10)
    plt.rcParams['text.usetex'] = True

    sig_test,y_test,HR_test=data
    total_err_list,handle_list=[],[]
    weights_path_prefix=log_prefix
    main_exp_tag=expid_list[0].split("_")[0]
    fig_dir=f'{weights_path_prefix}/figures'
    table_dir=f'{weights_path_prefix}/tables/{main_exp_tag}'
    os.makedirs(fig_dir,exist_ok=True)
    os.makedirs(table_dir,exist_ok=True)

    fig=plt.figure()
    for k in range(len(expid_list)):
        exp_sub_id=expid_list[k]
        
        #% Load HR model
        #weights_path_prefix=glob.glob('../experiments/' + expid + '*')[0] #for latest experiment weights
        weights_dir=weights_path_prefix+f'/{exp_sub_id}/checkpoints'
        #weights_filepath_HR=weights_dir_HR+'/cp.ckpt'
        latest_ckpt = tf.train.latest_checkpoint(weights_dir)
        print('Loading model from ckpt {}'.format(latest_ckpt))
        #infer_model_HR.load_weights(latest_ckpt)
        model.load_weights(latest_ckpt)
        #model_sl.evaluate(val_ds)
        y_hat=model.predict(sig_test)
        cm=confusion_matrix(np.argmax(y_test,axis=-1),
                         np.argmax(y_hat,axis=-1),normalize=None)

        df=pd.DataFrame(data=cm, index=confusion_keys, 
                        columns=confusion_keys, dtype=None)
        df.to_csv(f'{table_dir}/{exp_sub_id}.csv',float_format='%.3f')
        check_test=(np.argmax(y_test,axis=-1)==np.argmax(y_hat,axis=-1))
        acc_test=np.mean(check_test.astype(int))
        print(acc_test*100)
        
        # Do binning and check plots
        err=np.invert(check_test).astype(int)
        n_bins = 10
        
        #TODO: Bins will be same as long as all_tacho, n_bins same
        #_,eqbins=pd.cut(1000*HR_test,n_bins,labels=False,retbins=True)
        #eqbins=np.round(eqbins,-1).astype(int)
        cuts,bins=pd.qcut(1000*HR_test,n_bins,labels=False,retbins=True)

        bin_err=np.zeros(n_bins)
        for i in range(n_bins):
            print(len(err[cuts==i]))
            bin_err[i]=np.mean(err[cuts==i])*100
    
    
        #width = 0.9 * (bins[1] - bins[0])
        #print(bin_err)
        center = (bins[:-1] + bins[1:]) / 2
        han,=plt.plot(center, bin_err,marker_list[k],label=label_list[k])
        
        total_err_list.append(np.mean(err))
        handle_list.append(han)
    plt.xlabel('Average RR ($ms$.)')
    plt.ylabel('Classification Error ($\%$)')
    plt.suptitle(title)
    #plt.gca().set_xscale('log',basex=10)
    #plt.yscale('log',base=10)
    plt.legend()#loc='upper right')
    plt.grid(True)
    #plt.ylim(bottom=0)
    #plt.xticks(eqbins,[f'{b:.0f}' for b in eqbins])
    
    print(total_err_list)
    print(((total_err_list[0]-total_err_list[-1])/
          total_err_list[0])*100)
    
    fig.tight_layout()
    if save_fig:
        # Best for professional typesetting, e.g. LaTeX
        #plt.savefig(f'{fig_dir}/{expid_list[-1]}_eqbins_notrain.pdf')
        plt.savefig(f'{fig_dir}/{main_exp_tag}.png',dpi=300)
    
    # #Plot training data size in each bin
    # # Load train_data if it doesn't exist
    # try:
    #     train_HR=60*HR_train
    # except NameError:
    #     d_list=load_data_helper(Dsplit_mask_dict,musig_dict,data_path,win_len_s,
    #                             step_s,bsize,bstep,ecg_win_len,Fs_new,augmentor,
    #                             seq_format_function_E2St,dsampling_factor_aug=1)
    #     val_data,train_data,_,_,misc_list=d_list
    #     Dsplit_mask_dict,HR_train=misc_list
    #     train_HR=60*HR_train
            
    # # available training data in every bin
    # train_HR=train_HR.flatten()
    
    # train_cuts,train_bins=pd.cut(train_HR,bins=bins,labels=False,retbins=True)
    # assert train_bins.all()==bins.all()
    
    # bin_n_train=np.zeros(n_bins)
    # for i in range(n_bins):
    #     bin_n_train[i]=100*(np.mean((train_cuts==i).astype(int)))
        
    # plt.grid(True) #first axis grid
    # ax = plt.gca()    # Get current axis
    # ax2 = ax.twinx()  # make twin axis based on x
    # #plot
    # tdata,=ax2.plot(center, bin_n_train,'r--x',label='train_data %')
    # ax2.set_ylabel("Training Data %")
    # handle_list.append(tdata)
    # plt.grid(True) #second axis grid
    
    # #TODO: legend on ax vs ax2
    # ax2.legend(handles=handle_list,loc='best')
    # ax2.yaxis.label.set_color(tdata.get_color())
    # # Adjust spacings w.r.t. figsize
    # fig.tight_layout()
    
    # # Best for professional typesetting, e.g. LaTeX
    # plt.savefig(f'{fig_dir}/{expid_list[-1]}_eqbins.pdf')
    # plt.savefig(f'{fig_dir}/{expid_list[-1]}_eqbins.png',dpi=300)
    
    return fig

def make_data_pipe(data,batch_size=64,shuffle=True,drop_remainder=True):
    #dataset = tf.data.Dataset.from_tensor_slices((data[0],data[1],data[2]))
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
    if shuffle:
        dataset=dataset.shuffle(buffer_size=np.max([6*batch_size,1000]).astype(int))
    dataset=dataset.batch(batch_size,drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    return dataset


def check_graph(model,shape_list=[(256,),(10,)],
                plot_model=True,file_path='Decoder.png'):
    '''
    Assumes batch_size of 1 while drawing graph
    '''
    #self.build((None, *shape))
    #dummy_data_list=[np.zeros((1,*shape),dtype=np.float32) for shape in shape_list]
    input_list= [layers.Input(shape=(shape))  for shape in shape_list]
    if len(input_list)==1:
        #model.call(dummy_data_list[0])
        temp_model=Model(inputs=input_list[0], outputs=model.call(input_list[0]))
    else:
        #model.call(dummy_data_list)
        temp_model=Model(inputs=input_list, outputs=model.call(input_list))
        
    print(temp_model.summary())
    if plot_model:
        tf.keras.utils.plot_model(
        temp_model,                      # here is the trick (for now)
        to_file=file_path, dpi=200,              # saving  
        show_shapes=True, show_layer_names=True,  # show shapes and layer name
        expand_nested=False                       # will show nested block
        )
    tf.keras.backend.clear_session() #clear out this temp model
    return

def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def smallest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i: # mean if NOT divisible
            i+=1
        else:
            return i
    return n
    #raise AssertionError("Something seems wrong. Couldn't find a prime factor.")
    #return

def find_batch_size(N,thres=1500,mode='train'):
    N_old=N*1
    if mode=='val':
        L=largest_prime_factor(N)
        #print(L)
        if L>thres:
            print('Largest factor is pretty high at {}. Be careful.'.format(L))
            return L,int(N_old/L)
    else:
        L=1
    N=N//L
    while N>=2:
        #print(N)
        l=smallest_prime_factor(N)
        #print(l)
        if L*l>thres:
            break
        else:
            L=L*l
            N=N//l
    return L,int(N_old/L)

def start_logging(log_file_name):
    log_file= open(log_file_name,'a')
    print('Printing stdout to the log_file '+log_file_name)
    origin_stdout=sys.stdout #keep this for reverting later
    sys.stdout = log_file
    return origin_stdout,log_file

def stop_logging(origin_stdout,log_file):
    #revert back to original stdout
    sys.stdout = origin_stdout
    log_file.close()
    return

def copy_any(src, dst):
    '''
    Source: tzot's answer at https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python
    '''
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
    return

def copy_R2Smodel_weights(proj_path,exp_no,sig_id_list=['ecg','ppg'],
        P_ID_list=[f'S{i}' for i in list(range(2,12))+list(range(13,18))]):
    ver=exp_no[:2]
    for P_ID in P_ID_list:
        for sig_id in sig_id_list:
            src=(proj_path+
                 f'/experiments/condWGAN/{exp_no}/{P_ID}_{sig_id}_R2S/checkpoints/')
            dst=(proj_path+
                 f'/data/post-training/model_weights_v{ver}/{P_ID}_{sig_id}_Morph_model')
            
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(dst)
            
            # Copy user desired exp_no and ckpt_no files to destination
            with open(src+'checkpoint','r') as file:
                ckpt_no = file.readline().rstrip().split(' ')[1][1:-1]
            files2copy=[src+'checkpoint']+list(glob.glob(src+ckpt_no+'*'))
            for file in files2copy: shutil.copy(file,dst)
    #copy_R2Smodel_weights('..','12_27_20220217-014313')
    return

def filtr_HR(X0,Fs=100,filt=True,cutoff=0.5):
    nyq=Fs/2
    assert cutoff+0.5<nyq,'cutoff+0.5 should be less than nyquist'
    if len(X0.shape)==1:
        X0=X0.reshape(-1,1)
    X1 = np.copy(X0)#sig.detrend(X0,type='constant',axis=0); # Subtract mean
    if filt:
        # filter design used from Ju's code with slight changes for python syntax
        b = sig.firls(219,np.array([0,cutoff,cutoff+0.5,nyq]),np.array([1,1,0,0]),np.array([1,1]),nyq=nyq);
        X=np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the centered signal without any delay
            X[:,i] = sig.filtfilt(b,[1],X1[:,i])
    else:
        X=X1
    #X=sig.detrend(X,type='constant',axis=0); # subtracted mean again to center around x=0 just in case things changed during filtering
    return X

def get_uniform_tacho(nn, fs=4, t_bias=0):
    '''
    Resampling (with 4Hz) and interpolate because RRi are unevenly space
    nn in [s]
    ''' 
    t=np.cumsum(nn)+t_bias
    #t-=t[0] #TODO: This line seemed wrong based on tacho deh if time matters
    f_interpol = sp.interpolate.interp1d(t, nn,'cubic',axis=0)
    
    t_start,t_end=np.ceil(t[0]),np.floor(t[-1]) #helps get integers on uniform grid
    t_interpol = np.arange(t_start,t_end, 1/fs)
    #Had to add explicit clipping due to floating point errors here.
    #t_interpol[-1]=min(t[-1],t_interpol[-1])
    #t_interpol[0]=max(t[0],t_interpol[0])
    nn_interpol = f_interpol(t_interpol)
    return t_interpol,nn_interpol

def get_leading_tacho(nn, fs=4):
    '''
    Resampling (with 4Hz) and interpolate because RRi are unevenly space
    nn in [s]
    '''
    
    t=np.cumsum(nn)#.astype(float)
    t=np.roll(t, shift=1, axis=0)
    t[0]=0
    
    f_interpol = sp.interpolate.interp1d(t, nn,'cubic',axis=0)
    t_interpol = np.arange(t[0], t[-1], 1./fs)
    nn_interpol = f_interpol(t_interpol)
    return t_interpol,nn_interpol


def get_continous_wins(sel_mask):
    # Get all sets of consecutive windows
    diff_arr=np.concatenate([[0],np.diff(sel_mask)])
    start_idxs=np.arange(len(sel_mask))[diff_arr==1]
    end_idxs=np.arange(len(sel_mask))[diff_arr==-1]
    #correct for terminal idxs
    if sel_mask[0]==1: start_idxs=np.concatenate([[0],start_idxs])
    if sel_mask[-1]==1: end_idxs=np.concatenate([end_idxs,[len(sel_mask)]])
    assert len(start_idxs)==len(end_idxs), "no. of start and end indices aren't equal"
    # # Check figure for idx detection
    # plt.figure();plt.plot(sel_mask)
    # plt.plot(start_idxs,sel_mask[start_idxs],'go')
    # plt.plot(end_idxs-1,sel_mask[end_idxs-1],'r+')
    return start_idxs,end_idxs
    