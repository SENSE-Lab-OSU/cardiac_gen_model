# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:46:09 2020

@author: agarwal.270a
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
#from scipy import io
#from pathlib import Path
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import layers
import glob
import pandas as pd
import scipy as sp
#import mpld3
#from lib.data import load_data_sense as load_data
from CardioGen.lib.data import load_data_wesad as load_data
from CardioGen.lib.simulator_for_CC import Simulator
#from lib.model_CondGAN import Net_CondGAN, Model_CondGAN
from CardioGen.lib.model_CondWGAN import Model_CondWGAN, downsample, upsample
from CardioGen.lib.utils import copy_any, start_logging, stop_logging, check_graph
from CardioGen.lib.utils import get_leading_tacho, get_uniform_tacho

import datetime
import os
import pickle
import time
import neurokit2 as nk
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
tf.keras.backend.set_floatx('float32')
proj_path='.'#(os.path.dirname(os.path.abspath(__file__))).replace(os.sep,'/')
def show_plot():
    #fig=plt.gcf()
    #mpld3.display(fig)
    return
#tf.config.set_visible_devices([], 'GPU')
ver=11 #version of the model_weights to use. Refer to README for details.
#%%
class HR2Rpeaks_Simulator(Simulator):
    '''
    Produces peak train from given smooth HR data
    '''
    def __init__(self,input_list=[],output_list=[],HR_win_len_s=8,P_ID='',
                 path='./',Fs_HR=25,Fs_tacho=5,latent_size=2,exp_id=f'{ver}_9'):
        '''
        in/output format: list of numpy arrays
        '''
        super(HR2Rpeaks_Simulator,self).__init__(input_list,output_list)
        self.P_ID=P_ID
        self.path=path
        self.Fs_HR=Fs_HR
        self.Fs_tacho=Fs_tacho
        self.HR_win_len_s=HR_win_len_s
        self.latent_size=latent_size
        self.exp_id=exp_id
        
        # gen_model is a 2nd order p(RR_prev,RR_next| HR cluster) distribution
        self.gen_model_path=path+"model_weights_v{}/{}_HRV_gen_model".format(ver,P_ID)
        
        fname=glob.glob(self.gen_model_path+'/checkpoint')
        if len(fname)==0:
            print('\n Learning HRV Gen Model... \n')
            self.gen_model=self.learn_gen_model(save_flag=True,EPOCHS = 2000, 
                                                latent_size=self.latent_size)
            del self.gen_model
        
        #Load and Test
        self.RNN_win_len=1# Reduce RNN win_len to reduce test time wastage
        self.win_step_size=1*self.RNN_win_len
        print('HRV Gen Model Exists. Loading ...')
        self.gen_model=self.load_gen_model(self.gen_model_path,
                            self.Fs_HR)
        print('Done!')
        #self.make_gen_plots(self.ecg_gen_model,int(self.w_pk*self.Fs_ecg))
        return
    
    def load_gen_model(self,path,Fs_out,stateful=True):
        '''
        Load a model from the disk.
        '''
        RNN_win_len=int(self.RNN_win_len)
        if Fs_out==100:
            model=self.create_gen_model(shape_in=[(None,RNN_win_len,1)],
                                    shape_out=[(None,RNN_win_len,1)],stateful=stateful,
                                    batch_size=1,model_path=path, latent_size=self.latent_size)
        elif Fs_out==25:
            model=self.create_gen_model(shape_in=[(None,RNN_win_len,1)],
                                shape_out=[(None,RNN_win_len,1)],stateful=stateful,
                                batch_size=1,model_path=path,latent_size=self.latent_size)
        else:
            raise AssertionError('Fs_out can only be 25 or 100 at this time.')  
            
        #model.load_weights(path)
        model.ckpt.restore(model.manager.latest_checkpoint)
        print("Restored from {}".format(model.manager.latest_checkpoint))
        return model
    

    def _hrv_freq_loss(self,y_true,y_hat): 
        def get_PSD(tacho):
            tacho = tf.squeeze(tacho,axis=[-1])
            fft_U=tf.signal.rfft(tacho,fft_length=[nfft])
            PSD_U=tf.math.divide(tf.square(tf.abs(fft_U)),norm_factr)
            #Sub-Select the freq band of interest
            PSD=PSD_U[:,1:len_fft_out] #skipping DC value
            #PSD_U=tf.reshape(PSD_U,shape=[-1,1])
            return PSD
        # Get some common parameters
        Fs_fft=1
        nfft,last_dim=y_true.get_shape().as_list()[1:]
        assert last_dim==1, 'last dimension is not 1. Please check the tacho tensor'
        dsamp_factor=int(self.Fs_tacho/Fs_fft)
        assert (self.Fs_tacho/Fs_fft)%1==0, 'Fs_tacho must be a natural no.'
        len_fft_out=int(nfft/(2*dsamp_factor))#+1
        #tacho_fft=tacho[:,::dsamp_factor] #dsample tacho
        #Find periodogram by simple welch method
        norm_factr=tf.constant(int(nfft/2)+1,dtype=tf.float32)
        #get PSDs
        PSD_true=get_PSD(y_true)
        PSD_hat=get_PSD(y_hat)
        loss=tf.reduce_mean(tf.square(PSD_true-PSD_hat))
        #return  PSD_true,PSD_hat,loss #TODO: for debugging
        return loss
        
    def create_gen_model(self,shape_in=None,shape_out=None,latent_size=4,
                     model_path='',stateful=False,batch_size=1,
                     optimizers = None,save_flag=True):
        '''
        model_path must be a dir
        '''
        # Create an instance of the model
        
        gen_layer_list=[downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                upsample(filters=16, kernel_size=(1,5),strides=(1,5),
                         apply_dropout=True),
                upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                upsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                upsample(filters=1, kernel_size=(1,3),strides=(1,2)),
                layers.GRU(8,return_sequences=True,stateful=stateful),
                layers.Conv1D(1,1,padding='same')]
        
        model = Model_CondWGAN(latent_size=latent_size,model_path=model_path,
                       gen_layers=gen_layer_list,use_x=True, mode='GAN',
                       optimizers=optimizers,save_flag=save_flag,
                       aux_losses=[self._hrv_freq_loss],
                       aux_losses_weights=[5e2])
        #model.summary()
        return model
    
    def learn_gen_model(self,latent_size=4,save_flag=True,make_plots=True,
                        EPOCHS = 100, logging=True):
        input_list,output_list=self.input,self.output
        #Pre-process data
        model_cond_in,model_RR_out=[],[]
        RNN_win_len,step_size=int(self.Fs_tacho*40),int(self.Fs_tacho*5)
        list_cond,list_arr_pks=input_list,output_list
        C_HR=0
        for j in range(len(list_cond)):
            
            #Get RR_ints in seconds
            RR_ints_NU, RR_extreme_idx=load_data.Rpeaks2RRint(list_arr_pks[j])
            cond=list_cond[j][RR_extreme_idx[0]:RR_extreme_idx[1]+1]
            
            # Uniformly interpolate at >=4 Hz
            t_interpol,RR_ints=get_leading_tacho(RR_ints_NU,fs=self.Fs_tacho)
            #get apt HR indices at t_interpol
            cond=cond[(t_interpol*self.Fs_HR).astype(int)]
            #HR=HR.reshape(-1,1)/60 #convert to BPS
            cond[:,C_HR:C_HR+1]=cond[:,C_HR:C_HR+1]/60 #convert HR to BPS

            #RR_ints_prev=np.concatenate([RR_ints[0:1],RR_ints[:-1]],axis=0)
            #HR=HR[:-1]/60#remove last HR as unusable and convert to BPS
            #print(len(RR_ints)-len(HR))
            cond,RR_ints=self.sliding_window_fragmentation(
                                    [cond,RR_ints.astype(np.float32)],RNN_win_len,step_size)
            model_cond_in.append(cond)
            #model_RR_in.append(RR_ints_prev)
            model_RR_out.append(RR_ints)
        model_cond_in=np.concatenate(model_cond_in,axis=0)
        #model_RR_in=np.concatenate(model_RR_in,axis=0)
        model_RR_out=np.concatenate(model_RR_out,axis=0)
        #model_in=np.concatenate([model_HR_in,model_RR_in],axis=-1)
        model_in = model_cond_in
        model_out= model_RR_out
        print(model_in.shape,model_out.shape)
        self.model_in=model_in
        self.model_out=model_out

        #partition
        val_ratio=0.1
        val_idx=int(val_ratio*len(model_in))
        val_data=[model_in[0:val_idx],model_out[0:val_idx]]
        train_data=[model_in[val_idx:],model_out[val_idx:]]
        
        #shuffle AFTER partition as time series based
        perm=np.random.permutation(len(train_data[1]))
        train_data=[train_data[0][perm],train_data[1][perm]]
        #perm=np.random.permutation(len(val_data[1]))
        #val_data=[val_data[0][perm],val_data[1][perm]]
        
        #tensorboard stuff
        #define log paths
    
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_prefix='../experiments/condWGAN/{}_{}/HR2R'.format(self.exp_id,current_time)
        train_log_dir = log_prefix + '/train'
        test_log_dir = log_prefix + '/test'
        stdout_log_file = log_prefix + '/stdout.log'
        checkpoint_dir = log_prefix + "/checkpoints"
        #figs_dir = log_prefix + "/figures"
        os.makedirs(log_prefix,exist_ok=True)
        
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        model=self.create_gen_model(latent_size=latent_size,save_flag=save_flag,
                                    model_path=checkpoint_dir)
        
        if logging:
            origin_stdout,log_file=start_logging(stdout_log_file)
        #TODO: Uncomment these to check the graphs
        print(model_in.dtype,model_out.dtype)
        print([(*model_in.shape[:2],latent_size),tuple(model_in.shape[:])])
        check_graph(model.gen,shape_list=[(model_in.shape[1],latent_size),tuple(model_in.shape[1:])],
                                              file_path=log_prefix+'/Generator.png')
        check_graph(model.disc,shape_list=[tuple(model_out.shape[1:]),tuple(model_in.shape[1:])],
                                              file_path=log_prefix+'/Discriminator.png')
        check_graph(model,shape_list=[tuple(model_in.shape[1:])],
                                              file_path=log_prefix+'/condWGAN.png')
        

        #Training in GAN mode
        model.fit(data=[train_data,val_data],
                  summaries=[train_summary_writer,test_summary_writer],
                  epochs = EPOCHS,batch_size=16*2)
        if logging:
            stop_logging(origin_stdout,log_file)
        
        # copy latest checkpoint_dir to gen_model_path
        copy_any(checkpoint_dir, self.gen_model_path)
        
        #if make_plots:
         #   self.make_stitchGAN_plots(model,w_pk)
            
        return model
    
    def __call__(self,cond_curve,Fs_out=100):
        
        #Get complete leading tacho
        C_HR=0
        RNN_win_len,step_size=int(self.Fs_tacho*40),int(self.Fs_tacho*40)
        assert (self.Fs_HR%self.Fs_tacho)==0, 'Fs_HR MUST be a multiple of Fs_tacho'
        factr_tacho=int(self.Fs_HR/self.Fs_tacho)
        cond_tacho=cond_curve*1
        cond_tacho[:,C_HR:C_HR+1]=cond_tacho[:,C_HR:C_HR+1]/60 #convert HR to BPS
        cond_tacho=cond_tacho[::factr_tacho] #dsample aptly
        cond_tacho=cond_tacho[:-int(len(cond_tacho)%RNN_win_len)]#.reshape(-1) #clip end aptly
        cond_tacho_windows=self.sliding_window_fragmentation([cond_tacho],
                                                   RNN_win_len,step_size)
        
        if len(cond_tacho_windows.shape)==2:
            cond_tacho_windows=np.expand_dims(cond_tacho_windows,axis=-1)
        #Stitch them together
        arr_tacho=np.zeros([*cond_tacho_windows.shape[:-1],1],dtype=np.float32)
        rnn_state=None
        for i in range(cond_tacho_windows.shape[0]):
            arr_tacho[i:i+1]=self.gen_model.predict([cond_tacho_windows[i:i+1]],rnn_state=rnn_state)
            rnn_state=self.gen_model.gen.rnn_state_out #update state
        arr_tacho=arr_tacho.reshape(-1)
        
        # Create a HR to leading tacho mapping
        cond_curve=cond_curve[:(len(cond_tacho)-1)*factr_tacho+1]
        t_steps=np.arange(len(cond_curve))
        lead_tacho = sp.interpolate.interp1d(t_steps[::factr_tacho], 
                                                 arr_tacho,'cubic',axis=0)
        
        #plt.figure()
        #plt.plot(t_steps[::factr_tacho],arr_tacho,'o-',t_steps,lead_tacho(t_steps),'r--')
        #plt.legend(['True','Upsampled'])
        
        #Use the mapping to form the R-peak train
        factr=(Fs_out/self.Fs_HR)
        arr_pk=np.zeros(int(factr*len(cond_curve)))
        #TODO: bunch of changes here
        #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
        #plt.figure();plt.plot(gauss)
        #HR_t=HR_curve[i,0]#np.mean(HR_curve[0:i])
        while i < (len(cond_curve)-1):
            #HR_t=HR_curve[i,0]
            idx=int(factr*i)
            arr_pk[idx]=1
            RR_next=lead_tacho(i)

            #update for next
            i+=int(RR_next*self.Fs_HR)
        return arr_pk,cond_curve

#%% Sample Client
if __name__=='__main__':
    len_in_s=20.48 #s
    arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists

    #Get Train Data for simulator
    plt.close('all')
    path='D:/Datasets/WESAD/'
    win_len_s=8;step_s=2;Fs_pks=100
    input_list,output_list=[],[]
    # (kalidas,7306,8.6s), (nabian,7270,9.2s), (neurokit,7253,7.6), 
    # (pantompkins1985, 7346, 14.4), (hamilton2002, 7355, 11), (elgendi2010, 7332, 19.35)
# =============================================================================
#     start_time=time.time()
#     test_in,test_out_for_check=load_data.get_test_data(path+'S2',mode='HR2R',win_len_s=win_len_s
#                                              ,step_s=step_s,Fs_pks=100)
#     print(f'{time.time()-start_time} s.')
# =============================================================================
    
    train_data_filename='processed_HRC2R_Kalidas_train_data.pkl'
    if not os.path.isfile(path+train_data_filename):
        input_list,output_list=load_data.get_train_data(path,'HR2R',win_len_s,step_s,Fs_pks)
        with open(path+train_data_filename, 'wb') as fp:
            pickle.dump([input_list,output_list], fp)
            
    with open (path+train_data_filename, 'rb') as fp:
        itemlist = pickle.load(fp)
    input_list,output_list=itemlist
    
    #define constants
    #path_prefix= os.path.dirname(os.path.abspath(__file__)).replace(os.sep,'/') #'C:/Users/agarwal.270/Box/'
    ckpt_path=proj_path+'/../data/post-training/'
    P_ID='W'
    Fs=100 #Hz
    #Train
    sim_HR2pks=HR2Rpeaks_Simulator(input_list,output_list,
                            HR_win_len_s=win_len_s,path=ckpt_path,P_ID=P_ID,
                            Fs_HR=Fs,latent_size=5,exp_id='{}_1'.format(ver))
    
    ##%%
    #Test Simulator
    #Use simulator to produce synthetic output given input
    all_class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
    #load_data.class_ids:
    for class_name in ['S15']:
    #for class_name in all_class_ids:
        # load_data.class_ids={'S7':all_class_ids['S7']}
        # test_in,test_out_for_check=load_data.get_test_data(path+'S7',mode='HR2R',win_len_s=win_len_s
        #                                      ,step_s=step_s,Fs_pks=100)
        # HR_S7=test_in[:,0]
        
        load_data.class_ids={class_name:all_class_ids[class_name]}
        test_in,test_out_for_check=load_data.get_test_data(path+class_name,mode='HR2R',win_len_s=win_len_s
                                             ,step_s=step_s,Fs_pks=100)
        
        # lenth=min(len(HR_S7),len(test_in))
        # test_in=test_in[:lenth]
        # test_in[:,0]=HR_S7[:lenth]
    
        
        #Fs_ppg=25;Fs_ecg=100
        arr_pk,test_in=sim_HR2pks(test_in)#.reshape(-1,1))
        #plt.figure(100);plt.plot(test_in[:,0])
        
        # #Plot some stuff
        # fig=plt.figure()  
        # plt.plot(test_out_for_check[19:])
        # plt.plot(arr_pk)
        # plt.legend(['True','Synthetic'])
        # #mpld3.display(fig)
        # show_plot()
        #HRV analysis
        hrv_features = nk.hrv(test_out_for_check[19:], sampling_rate=Fs, show=True);show_plot()
        plt.suptitle(class_name)
        hrv_features_synth = nk.hrv(arr_pk, sampling_rate=Fs, show=True);show_plot()
        plt.suptitle(class_name)
    
    
    #plt.figure(100);plt.legend(['S3', 'S7', 'S10', 'S11', 'S15'])
    
    hrv_lomb = nk.hrv_frequency(test_out_for_check[19:], sampling_rate=100, show=True, psd_method="lomb");show_plot()
    hrv_lomb_synth = nk.hrv_frequency(arr_pk, sampling_rate=100, show=True, psd_method="lomb");show_plot()

#%%    
    #manual calculation
    def find_tacho(arr_pk,Fs_pks=100):
        Fs_pks=100
        pk_locs=np.arange(len(arr_pk))[arr_pk.astype(bool)]
        time_locs=pk_locs/Fs_pks
        RR_ints=np.diff(time_locs).reshape((-1,1))
        return time_locs[1:],RR_ints
    
    time_locs,RR_ints=find_tacho(test_out_for_check[19:])
    time_locs_synth,RR_ints_synth=find_tacho(arr_pk)
    
    # Get lomb periodogram
    
    #import scipy.signal as signal
    from astropy.timeseries import LombScargle
    frequency, power = LombScargle(time_locs, RR_ints.reshape(-1)).autopower()
    frequency_synth, power_synth = LombScargle(time_locs_synth, RR_ints_synth.reshape(-1)).autopower()
    
    #from lib.simple_NFFT import ndft#,nfft2, nfft3
    #power_synth = ndft(time_locs, RR_ints.reshape(-1),int(len(power)))

    #f = np.linspace(0.01, 12, 1000)
    #pgram = signal.lombscargle(time_locs, RR_ints.reshape(-1), f, normalize=False)
    #pgram_synth = signal.lombscargle(time_locs_synth, RR_ints_synth.reshape(-1), f, normalize=False)

    plt.figure()
    plt.plot(time_locs,RR_ints,time_locs_synth,RR_ints_synth,'r--')
    plt.title('Tachogram')
    plt.xlabel('Time (s)');plt.ylabel('RR_interval (s)')
    plt.legend(['True','Synthetic'])
    plt.grid(True)
    
    plt.figure()
    #plt.plot(f,pgram,f,pgram_synth,'r--')
    plt.plot(frequency, power, frequency_synth, power_synth,'r--')
    plt.title('PSD of Tachogram')
    plt.xlabel('Frequency (Hz)');plt.ylabel('PSD')
    plt.legend(['True','Synthetic'])
    plt.grid(True)
    
    #plt.figure();plt.plot(power)
    #plt.figure();plt.plot(np.abs(power_synth)**2)

# =============================================================================
#     #generate HR
#     len_in_s=20.48 #s
#     len_out=4
#     len_in=Fs*len_in_s
#     arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists
#     t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
#     t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
#     HR_curve_f,D_HR=HR_func_generator(t1)
#     
#     #Test Simulator
#     sim_HR2pks=HR2Rpeaks_Simulator([],[],path=path,P_ID=P_ID,Fs_HR=Fs)
#     arr_pk=sim_HR2pks(HR_curve_f)
#     
#     #Check if upsampling works
#     arr_pk_upsampled=sim_HR2pks(HR_curve_f,Fs_out=100)
#     check=arr_pk_upsampled.reshape(-1,4)
#     plt.figure();plt.plot(arr_pk);plt.plot(arr_pk_upsampled[::4])
#     #ppg1,HR1=gen_ppg_from_HR(t1,HR_curve_f,D_HR,peak_id,make_plots=make_plots)
# =============================================================================
