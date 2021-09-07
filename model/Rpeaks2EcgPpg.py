# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:47:57 2020

@author: agarwal.270a
"""

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
#from scipy import io
from scipy import signal as sig

#from scipy.signal import detrend
#from pathlib import Path
from lib.simulator_for_CC import Simulator
#from lib.model_stitchGAN import Model_stitchGAN, Net_stitchGAN
import tensorflow as tf
from tensorflow.keras import layers
import datetime
#from Rpeaks2EcgPpg_gen_model_v2 import NLL_loss, mse_metric
#from scipy.stats import truncnorm
#import pickle
#from lib.model_CondGAN import Net_CondGAN, Model_CondGAN
from lib.model_CondWGAN import Model_CondWGAN, downsample, upsample
from lib.utils import copy_any, start_logging, stop_logging, check_graph
#from lib.data import load_data_sense as load_data
from lib.data import load_data_wesad as load_data
n_classes=load_data.n_classes

import os
import sys
import pickle
import neurokit2 as nk

tf.keras.backend.set_floatx('float32')
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

proj_path='.'#(os.path.dirname(os.path.abspath(__file__))).replace(os.sep,'/')
def show_plot():
    #fig=plt.gcf()
    #mpld3.display(fig)
    return
#tf.config.set_visible_devices([], 'GPU')
ver=11 #version of the model_weights to use. Refer to README for details.

#%%
#del Rpeaks2EcgPpg_Simulator
class Rpeaks2EcgPpg_Simulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,input_list=[],output_list=[],P_ID='',path='./',
                 ppg_id='ppg_green',latent_size=6,logging=False,current_time=None,
                 exp_id='{}_3'.format(ver)):
        '''
        in/output format: list of numpy arrays. 1st channel of output is 
        w_pk and w_l defined in seconds
        '''
        super(Rpeaks2EcgPpg_Simulator,self).__init__(input_list,output_list)
        # TODO: Didn't change w_pk and w_l sizes for ECG as ECG needs smaller 
        # windows than ppg. But might need to change depending upon changing Fs
        self.P_ID=P_ID
        self.path=path
        self.ppg_id=ppg_id
        self.ecg_id='ecg'
        self.Cout_ppg=(0,1)
        self.Cout_ecg=(1,5)
        self.Fs_ppg=25
        self.Fs_ecg=100
        self.Fs_pks=100
        #self.latent_size=latent_size
        self.latent_size=latent_size

        self.RNN_win_len=8 #in secs
        self.win_step_size=0.25*self.RNN_win_len #in secs
        self.n_stitch_channels=4
                
        self.logging=logging
        self.exp_id= exp_id
        
        if current_time is None:
            self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.current_time = current_time
        
            
        # self.ppg_stitchGAN_path=path+"model_weights_v{}/{}_{}_stitch_model".format(
        #                                             ver,self.P_ID,self.ppg_id)
        # os.makedirs(self.ppg_stitchGAN_path,exist_ok=True)

        # fname=glob.glob(self.ppg_stitchGAN_path+'/*.index')
        # if len(fname)==0:
        #     print('\n Learning PPG stitchGAN... \n')
        #     self.log_prefix='../experiments/condWGAN/{}_{}/{}_{}_R2S_stitch'.format(self.exp_id,self.current_time,self.P_ID,self.ppg_id)

        #     self.learn_ppg_stitchGAN(save_flag=True,
        #                              latent_size=self.latent_size)
        #     del self.ppg_stitchGAN #delete after training
            
        self.ecg_stitchGAN_path=path+"model_weights_v{}/{}_{}_stitch_model".format(
                                                    ver,self.P_ID,self.ecg_id)
        os.makedirs(self.ecg_stitchGAN_path,exist_ok=True)
        fname=glob.glob(self.ecg_stitchGAN_path+'/*.index')
        if len(fname)==0:
            if len(input_list)==0:
                assert False,('ECG Morph model does-not-exist. Supply Training'
                            'data as input_list and output_list arguments')
            print('\n Learning ECG stitchGAN... \n')
            self.log_prefix='../experiments/condWGAN/{}_{}/{}_{}_R2S_stitch'.format(self.exp_id,self.current_time,self.P_ID,self.ecg_id)

            self.learn_ecg_stitchGAN(save_flag=True,
                                     latent_size=self.latent_size)
            del self.ecg_stitchGAN
        
        #Load and Test
        #self.RNN_win_len=1# Reduce RNN win_len to reduce test time wastage
        self.win_step_size=1*self.RNN_win_len
            
        # print('PPG stitchGAN Exists. Loading ...')
        # self.ppg_stitchGAN=self.load_model_stitchGAN(self.ppg_stitchGAN_path,self.w_pk,
        #                                        self.Fs_ppg)
        # print('Done!')
        # #self.make_gen_plots(self.ppg_gen_model,int(self.w_pk*self.Fs_ppg))
        
        print('ECG Morph model Exists. Loading ...')
        self.ecg_stitchGAN=self.load_model_stitchGAN(self.ecg_stitchGAN_path,
                            self.Fs_ecg)
        print('Done!')
        #self.make_gen_plots(self.ecg_gen_model,int(self.w_pk*self.Fs_ecg))
     
    def load_model_stitchGAN(self,path,Fs_out,stateful=False):
        '''
        Load a model from the disk.
        '''
        RNN_win_len=int(Fs_out*self.RNN_win_len)
        model=self.create_stitchGAN_model(shape_in=[(None,RNN_win_len,2*self.n_stitch_channels+1)],
                                shape_out=[(None,RNN_win_len,1)],stateful=stateful,
                                batch_size=1,latent_size=self.latent_size,
                                model_path=path)
        #model.load_weights(path)
        model.ckpt.restore(model.manager.latest_checkpoint)
        print("Restored from {}".format(model.manager.latest_checkpoint))
        return model

    def ppg_filter(self,X0,filt=True):
        '''
        Band-pass filter multi-channel PPG signal X0
        '''
        nyq=self.Fs_ppg/2
        X1 = sig.detrend(X0,type='constant',axis=0); # Subtract mean
        if filt:
            b = sig.firls(219,np.array([0,0.3,0.5,4.5,5,nyq]),
                          np.array([0,0,1,1,0,0]),np.array([10,1,1]),nyq=nyq)
            X=np.zeros(X1.shape)
            for i in range(X1.shape[1]):
                #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
                X[:,i] = sig.filtfilt(b,[1],X1[:,i])
    
        else:
            X=X1
        return X
    
    def GAN_stitcher(self,X0,filt=True):
        '''
        Use GAN sticher (The Generator) to put things together
        '''
        return self.stitchGAN.predict(X0)
    

    def learn_ppg_stitchGAN(self,save_flag=False,latent_size=6):
        '''
        Regenerate the ppg_basis
        '''
        
        self.ppg_stitchGAN=self.learn_stitchGAN_model(self.input,self.output,
                                        self.ppg_stitchGAN_path,
                                        self.Cout_ppg,Fs_in=self.Fs_pks,
                                        Fs_out=self.Fs_ppg,
                                        save_flag=save_flag,EPOCHS = 2,
                                        latent_size=latent_size)
        return
    
    
    def learn_ecg_stitchGAN(self,save_flag=False,latent_size=6):
        '''
        Regenerate the ecg_basis
        '''
        
        self.ecg_stitchGAN=self.learn_stitchGAN_model(self.input[::4],
                                        self.output[::4],
                                        self.ecg_stitchGAN_path,
                                        self.Cout_ecg,Fs_in=self.Fs_pks,
                                        Fs_out=self.Fs_ecg,
                                        save_flag=save_flag,EPOCHS = 500,
                                        latent_size=latent_size)
        return
    

    def create_stitchGAN_model(self,shape_in=None,shape_out=None,latent_size=6,
                     model_path='',stateful=False,batch_size=1,
                     optimizers = None,save_flag=True,aux_losses_weights=None):
        # Create an instance of the model
        gen_layer_list=[downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                        downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                        downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=8, kernel_size=(1,3),strides=(1,2),
                                 apply_dropout=True),
                        upsample(filters=4,kernel_size=(1,3),strides=(1,2)),
                        layers.GRU(8,return_sequences=True,stateful=stateful),
                        layers.Conv1D(1,1,padding='same')]
        
        model = Model_CondWGAN(latent_size=latent_size,model_path=model_path,
                       gen_layers=gen_layer_list,use_x=True, mode='GAN',
                       optimizers=optimizers,save_flag=save_flag,
                       aux_losses=[tf.keras.losses.MeanSquaredError()],
                       aux_losses_weights=aux_losses_weights,
                       n_classes=n_classes)
        #model.summary()
        return model
    
    def learn_stitchGAN_model(self,input_list,output_list,path4model,
                              Cout=(0,1),Fs_in=100,Fs_out=25,
                              save_flag=True,make_plots=True,
                              EPOCHS = 100,latent_size= 6):
        '''
        Learn a generative model
        sim_pks2sigs
        '''
        #logging stuff
        #log_prefix='../experiments/condWGAN/{}_{}/R2S_stitch'.format(self.exp_id,self.current_time)
        log_prefix=self.log_prefix
        train_log_dir = log_prefix + '/train'
        test_log_dir = log_prefix + '/test'
        stdout_log_file = log_prefix + '/stdout.log'
        checkpoint_dir = log_prefix + "/checkpoints"
        #figs_dir = log_prefix + "/figures"
        os.makedirs(log_prefix,exist_ok=True)
        if self.logging:
            origin_stdout,log_file=start_logging(stdout_log_file)
            
            
        #TODO: Think about this
        if Fs_out==self.Fs_ppg:
            fig_no=200
            fixed_epochs=0#200
            batch_size_init=32
            alw=[2]
        elif Fs_out==self.Fs_ecg:
            fig_no=100
            fixed_epochs=0#80
            batch_size_init=16
            alw=[5*2]
        else:
            assert False, 'Fs_out should be 25 or 100. No other value is supported currently. \n'
        
        factr=int(Fs_in/Fs_out)
        RNN_win_len=int(self.RNN_win_len*Fs_out)
        step_size=int(self.win_step_size*Fs_out)
        C_pks=0
        def get_data():
            #RNN_win_len=300;step_size=100 #TODO: Override for debugging ECG
            outpt=[output_list[i][:,Cout[0]:Cout[1]].reshape(-1) 
                for i in range(len(output_list))]
            GAN_input_list,GAN_output_list=[],[]
            
            for i in range(len(input_list)):
                arr_pks = input_list[i][:,C_pks]
                arr_y,clip_indices=self.get_intermediate_signal(arr_pks,factr)
                out = outpt[i][clip_indices[0]:clip_indices[1]] 
                if i==0: plt.figure();plt.plot(arr_y);plt.plot(out)
                
                #Append the class_ids:
                arr_y=np.concatenate([arr_y.reshape(-1,1),
                        input_list[i][:len(arr_y),-n_classes:]],axis=-1)
                arr_y,out=self.sliding_window_fragmentation([arr_y,out],
                                    RNN_win_len,step_size)
                #arr_y=np.concatenate([arr_y,
                #        np.ones(list(arr_y.shape[:-1])+[1],dtype=np.float32)],axis=-1)
                GAN_input_list.append(arr_y)
                GAN_output_list.append(np.expand_dims(out,axis=-1))
            
            GAN_input=np.concatenate(GAN_input_list,axis=0)
            GAN_output=np.concatenate(GAN_output_list,axis=0)
            print(GAN_input.shape,GAN_output.shape)
            #partition
            val_perc=0.14
            val_idx=int(val_perc*len(GAN_input))
            val_data=[GAN_input[0:val_idx],GAN_output[0:val_idx]]
            train_data=[GAN_input[val_idx:],GAN_output[val_idx:]]
            
            #shuffle after partition as time series based
            #perm=np.random.permutation(len(train_data[1]))
            #train_data=[train_data[0][perm],train_data[1][perm]]
            #GAN_input,GAN_output=GAN_input[perm],GAN_output[perm]
            #perm=np.random.permutation(len(val_data[1]))
            #val_data=[val_data[0][perm],val_data[1][perm]]
            return train_data,val_data
        
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        
        model=self.create_stitchGAN_model(latent_size=latent_size,save_flag=save_flag,
                                    model_path=checkpoint_dir,aux_losses_weights=alw)
        #Restore pre-trained weights
        model.ckpt.restore(model.manager.latest_checkpoint)
        print("Restored from {}".format(model.manager.latest_checkpoint))
        train_data,val_data=get_data()
        plt.figure(fig_no);plt.plot(train_data[0][5,:,0])
        plt.figure(fig_no+1);plt.plot(train_data[1][5,:,0])
        
        print('Printing Gen in_shape',[(train_data[0].shape[1],latent_size),
                    (train_data[0].shape[1],train_data[0].shape[2])])
        #TODO: Uncomment these to check the graphs
        check_graph(model.gen,shape_list=[(int(train_data[0].shape[1]/8),latent_size),
                    (train_data[0].shape[1],train_data[0].shape[2])],
                    file_path=log_prefix+'/Generator.png')
        check_graph(model.disc,shape_list=[tuple(train_data[1].shape[1:]),
                    (train_data[0].shape[1],train_data[0].shape[2])],
                    file_path=log_prefix+'/Discriminator.png')
        check_graph(model,shape_list=[(train_data[0].shape[1],
                                       train_data[0].shape[2])],
                    file_path=log_prefix+'/condWGAN.png')
        
        #train_data_fixed_in=train_data[0]*1 #TODO: store for future need
        model.mode='GAN_recon'
        if fixed_epochs:
            model.fit(data=[train_data,val_data],
                      summaries=[train_summary_writer,test_summary_writer],
                      epochs = fixed_epochs, batch_size=int(batch_size_init))
        
        
        
        #Switching to GAN mode
        model.mode='GAN'
        epochs2regen=1
        #batch_size_list=6*[batch_size_init]+2*[int(batch_size_init/2)]
        batch_size_list=int(EPOCHS/epochs2regen)*[batch_size_init]#+2*[int(batch_size_init/2)]

        for j in range(int(EPOCHS/epochs2regen)):
            
            train_data_aug=[np.copy(train_data[0]),np.copy(train_data[1])]
            #Shuffle
            perm=np.random.permutation(len(train_data_aug[1]))
            train_data_aug=[train_data_aug[0][perm],train_data_aug[1][perm]]
            
            model.fit(data=[train_data_aug,val_data],
                      summaries=[train_summary_writer,test_summary_writer],
                      epochs = epochs2regen, batch_size=batch_size_list[j])
        

        #self.GAN_input=GAN_input
        #self.GAN_output=GAN_output
        
        if save_flag:
            # copy latest checkpoint_dir to gen_model_path
            copy_any(checkpoint_dir, path4model)
            
        if self.logging:
            stop_logging(origin_stdout,log_file)
        return model
    
    
    def get_intermediate_signal(self,arr_pks,factr):
        '''
        sim_pks2sigs
        '''
        r_pk_locs_origin=np.arange(len(arr_pks))[arr_pks.astype(bool)]
        #clip terminal r_pks
        #TODO: Made changes here for better utilization of data
        #r_pk_locs_origin=r_pk_locs_origin
        #get nearest dsampled idx
        r_pk_locs=np.floor(r_pk_locs_origin[1:-1]/factr).astype(int)
        #RR_int=np.diff(np.floor(r_pk_locs_origin[:-1]/factr).astype(int))
        # find RR-intervals in seconds
        #RR_ints=np.diff(r_pk_locs_origin[:-1]/self.Fs_pks).reshape((-1,1))
        
        clip_indices=[r_pk_locs[0],r_pk_locs[-1]+1]
        #construct dsampled arr_pks
        arr_pks_dsampled=np.zeros(int(len(arr_pks)/factr)).astype(np.float32)
        arr_pks_dsampled[r_pk_locs]=1

        arr_pks_dsampled=arr_pks_dsampled[clip_indices[0]:clip_indices[1]]
        return arr_pks_dsampled,clip_indices
    
    def __call__(self,cond_sig,sigs2return=['PPG','ECG']):
        '''
        arr_pks: Location of R-peaks determined from ECG
        k: No. of prominent basis to keep
        sim_pks2sigs
        '''
        factr_ppg=int(self.Fs_pks/self.Fs_ppg)
        factr_ecg=int(self.Fs_pks/self.Fs_ecg)
        
        RNN_win_len_ppg=int(self.RNN_win_len*self.Fs_ppg)
        step_size_ppg=RNN_win_len_ppg*1#int(self.win_step_size*self.Fs_ppg)
        RNN_win_len_ecg=int(self.RNN_win_len*self.Fs_ecg)
        step_size_ecg=RNN_win_len_ecg*1#int(self.win_step_size*self.Fs_ecg)
        C_pks=0
        arr_pks=cond_sig[:,C_pks]
        
        returned_sigs=6*[None]
        if 'PPG' in sigs2return:
            #w_pk_ppg=int(self.w_pk*self.Fs_ppg)
            #w_l_ppg=int(self.w_l*self.Fs_ppg)
            #Get Morphologies
            arr_pks_ppg,clip_indices_ppg=(self.get_intermediate_signal(arr_pks,
                                        factr_ppg))

            
            arr_pks_ppg=arr_pks_ppg[:-int(len(arr_pks_ppg)%RNN_win_len_ppg)]
            arr_pks_ppg=np.concatenate([arr_pks_ppg.reshape(-1,1),
                        cond_sig[:len(arr_pks_ppg),-n_classes:]],axis=-1)
            #arr_interm_ppg=np.expand_dims(arr_interm_ppg,axis=1)
            arr_pks_ppg_windows=self.sliding_window_fragmentation([arr_pks_ppg],
                                            RNN_win_len_ppg,step_size_ppg)
            #arr_pks_ppg_windows=np.expand_dims(arr_pks_ppg_windows,axis=-1)
            
            #Stitch them together
            arr_ppg=np.zeros([*arr_pks_ppg_windows.shape[:-1],1])
            rnn_state=None
            for i in range(arr_pks_ppg_windows.shape[0]):
                arr_ppg[i:i+1]=self.ppg_stitchGAN.predict([arr_pks_ppg_windows[i:i+1]],rnn_state=rnn_state)
                rnn_state=self.ppg_stitchGAN.gen.rnn_state_out #update state
            #arr_ppg=np.random.normal(loc=arr_ppg[:,:,0],scale=arr_ppg[:,:,1])
            #arr_ppg=truncnorm.rvs(-1,1,loc=arr_ppg[:,:,0],scale=0.2*arr_ppg[:,:,1])

            #arr_ppg_stitched=np.concatenate([arr_ppg[0,:,0],
            #                arr_ppg[1:,-step_size_ppg:,0].reshape(-1)])
            arr_ppg=arr_ppg.reshape(-1)
            #modify end clip idx to account for wastage at test time due to RNN_win_len
            clip_indices_ppg[1]=clip_indices_ppg[0]+len(arr_ppg)
            ppg_out_list=[arr_ppg, arr_pks_ppg, clip_indices_ppg]
            returned_sigs[3:]=ppg_out_list
            
        if 'ECG' in sigs2return:
            #Get Morphologies
            arr_pks_ecg,clip_indices_ecg=(
                                        self.get_intermediate_signal(arr_pks,
                                        factr_ecg))
            arr_pks_ecg=arr_pks_ecg[:-int(len(arr_pks_ecg)%RNN_win_len_ecg)]
            arr_pks_ecg=np.concatenate([arr_pks_ecg.reshape(-1,1),
                        cond_sig[:len(arr_pks_ecg),-n_classes:]],axis=-1)
            #arr_interm_ecg=np.expand_dims(arr_interm_ecg,axis=1)
            arr_pks_ecg_windows=self.sliding_window_fragmentation([arr_pks_ecg],
                                            RNN_win_len_ecg,step_size_ecg)
            #arr_pks_ecg_windows=np.expand_dims(arr_pks_ecg_windows,axis=-1)
            #Stitch them together
            arr_ecg=np.zeros([*arr_pks_ecg_windows.shape[:-1],1])
            rnn_state=None
            for i in range(arr_pks_ecg_windows.shape[0]):
                arr_ecg[i:i+1]=self.ecg_stitchGAN.predict([arr_pks_ecg_windows[i:i+1]],rnn_state=rnn_state)
                rnn_state=self.ecg_stitchGAN.gen.rnn_state_out #update state
            #arr_ecg=np.random.normal(loc=arr_ecg[:,:,0],scale=arr_ecg[:,:,1])
            #arr_ecg=truncnorm.rvs(-1,1,loc=arr_ecg[:,:,0],scale=0.2*arr_ecg[:,:,1])
            #arr_ecg_stitched=np.concatenate([arr_ecg[0,:,0],
            #                arr_ecg[1:,-step_size_ecg:,0].reshape(-1)])
            arr_ecg=arr_ecg.reshape(-1)
            #modify end clip idx to account for wastage at test time due to RNN_win_len
            clip_indices_ecg[1]=clip_indices_ecg[0]+len(arr_ecg)
            
            #modify clip indices of ecg to match ppg ones, if ppg exists
            if returned_sigs[-1] is not None:
                err_message=("ECG length should be = 4 times PPG length." 
                             "It's not. ECG len= {}, PPG len = {}. "
                             "Debug code or contact the author."
                             ).format(len(arr_ecg),len(arr_ppg))
                assert len(arr_ecg)==4*len(arr_ppg), err_message
                
            ecg_out_list=[arr_ecg,arr_pks_ecg, clip_indices_ecg]
            returned_sigs[:3]=ecg_out_list
        return returned_sigs
    
#%% Client
if __name__=='__main__':
    # Data Helpers
    import seaborn as sns
    sns.set()
    path='D:/Datasets/WESAD/'
    ckpt_path=proj_path+'/../data/post-training/'

    #max_ppg_val=load_data.MAX_PPG_VAL
    #max_ecg_val=load_data.MAX_ECG_VAL

    for v,k in enumerate(list(range(11,12))+list(range(13,17))):
        class_name=f'S{k}'
        load_data.class_ids={class_name:v}

        #Get Train Data for simulator
        plt.close('all')
        #path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #        
        # train_data_filename='processed_RC2EP_Kalidas_train_data.pkl'
        # if not os.path.isfile(path+train_data_filename):
        #     input_list,output_list,dict_musig=load_data.get_train_data(path,mode='R2EP')
        #     with open(path+train_data_filename, 'wb') as fp:
        #         pickle.dump([input_list,output_list,dict_musig], fp)
                
        # with open (path+train_data_filename, 'rb') as fp:
        #     itemlist = pickle.load(fp)
        # input_list,output_list,dict_musig=itemlist
        
        input_list,output_list,dict_musig=load_data.get_train_data(path,mode='R2EP')
        
        # filename = path+"WESAD_musig_v{}.pickle".format(ver)
        # with open(filename, 'wb') as handle:
        #     pickle.dump(dict_musig, handle)
        #See ECG
        #aa=output_list[0][:,0:1].reshape(-1)
        #plt.figure();plt.plot(aa)
        
        #Create Simulator using train data
        ckpt_path=proj_path+'/../data/post-training/'
        sim_pks2sigs=Rpeaks2EcgPpg_Simulator(input_list,output_list,P_ID=class_name,
                path=ckpt_path,latent_size=2,logging=True,current_time='20210904-152320',
                exp_id='{}_6{}'.format(ver,class_name))
        del sim_pks2sigs
#%%
    #Use simulator to produce synthetic output given input
    all_class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
    #load_data.class_ids:
    for class_name in ['S15']:
        load_data.class_ids={class_name:all_class_ids[class_name]}
        #Create Simulator using train data
        sim_pks2sigs=Rpeaks2EcgPpg_Simulator(input_list=[],output_list=[],P_ID=class_name,
                path=ckpt_path,latent_size=2)
        
        test_in,test_out_for_check,dict_musig=load_data.get_test_data(path+class_name,mode='R2EP')
        synth_ecg_out,test_in_ecg,clip_ecg,synth_ppg_out,test_in_ppg,clip_ppg=sim_pks2sigs(test_in,sigs2return=['ECG'])
        
        start,end=clip_ecg
        fig2=plt.figure()
        plt.plot((test_out_for_check[:,1:5].reshape(-1))[start:end])
        plt.plot(synth_ecg_out)
        plt.plot(test_in_ecg[:,0])
        plt.legend(['True','Synthetic','R-peaks'],loc='lower right')
        plt.title(class_name);plt.grid(True)
        
        Fs_ecg_new=100
        delta=5
        rpeaks=np.arange(len(test_in_ecg))[test_in_ecg[:,0].astype(bool)]
        #rpeaks={'ECG_R_Peaks':rpeaks}
        signal_peak, morph_features = nk.ecg_delineate(
            (test_out_for_check[:,1:5].reshape(-1))[start+delta:end], 
            rpeaks[1:]-delta, sampling_rate=Fs_ecg_new, show=True,
            method='peaks', show_type='peaks')
        plt.title(class_name);
        signal_peak, morph_features = nk.ecg_delineate(synth_ecg_out[delta:], 
            rpeaks[1:]-delta, sampling_rate=Fs_ecg_new, show=True,method='peaks', show_type='peaks')
    
        del sim_pks2sigs

    #synth_ecg_out,test_in_ecg,clip_ecg,synth_ppg_out,test_in_ppg,clip_ppg,ppg_pks,ecg_pks=sim_pks2sigs(test_in)

    # synth_ppg_out*=dict_musig['ppg']['sig'] #rescale
    # synth_ppg_out+=dict_musig['ppg']['mu'] #add back mean
    # synth_ecg_out*=dict_musig['ecg']['sig'] #rescale
    # synth_ecg_out+=dict_musig['ecg']['mu'] #add back mean
    
    #with open('./figures/fig_REPv3.pickle', 'wb') as file:
    #    pickle.dump([fig1,fig2],file