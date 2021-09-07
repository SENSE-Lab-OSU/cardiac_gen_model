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
from tensorflow.keras import Model, layers
AUTOTUNE = tf.data.experimental.AUTOTUNE
import scipy.signal as sig
import scipy as sp

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


    
def make_data_pipe(data,batch_size=64,shuffle=True):
    #dataset = tf.data.Dataset.from_tensor_slices((data[0],data[1],data[2]))
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
    if shuffle:
        dataset=dataset.shuffle(buffer_size=np.max([6*batch_size,1000]).astype(int))
    dataset=dataset.batch(batch_size,drop_remainder=True).prefetch(AUTOTUNE)
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

def get_uniform_tacho(nn, fs=4):
    '''
    Resampling (with 4Hz) and interpolate because RRi are unevenly space
    nn in [s]
    ''' 
    t=np.cumsum(nn)
    #t-=t[0] #TODO: This line seemed wrong based on tacho deh if time matters
    f_interpol = sp.interpolate.interp1d(t, nn,'cubic',axis=0)
    t_interpol = np.arange(t[0], t[-1], 1./fs)
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
    