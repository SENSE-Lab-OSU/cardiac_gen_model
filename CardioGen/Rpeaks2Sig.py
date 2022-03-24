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
import copy

#from scipy.signal import detrend
#from pathlib import Path
#from lib.model_stitchGAN import Model_stitchGAN, Net_stitchGAN
import tensorflow as tf
from tensorflow.keras import layers
import datetime
#from scipy.stats import truncnorm
#import pickle

import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    #from lib.model_CondGAN import Net_CondGAN, Model_CondGAN
    from lib.simulator_for_CC import Simulator
    from lib.model_CondWGAN import Model_CondWGAN#, downsample, upsample
    from lib.networks_GAN import downsample, upsample
    from lib.utils import copy_any, start_logging, stop_logging, check_graph
    #from lib.data import load_data_sense as load_data
    from lib.data import load_data_wesad as load_data
else:
    #from lib.model_CondGAN import Net_CondGAN, Model_CondGAN
    from .lib.simulator_for_CC import Simulator
    from .lib.model_CondWGAN import Model_CondWGAN#, downsample, upsample
    from .lib.networks_GAN import downsample, upsample
    from .lib.utils import copy_any, start_logging, stop_logging, check_graph
    #from lib.data import load_data_sense as load_data
    from .lib.data import load_data_wesad as load_data

n_classes=load_data.n_classes

import os
import sys
import pickle
import neurokit2 as nk

tf.keras.backend.set_floatx('float32')
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

proj_path='.'#(os.path.dirname(os.path.abspath(__file__))).replace(os.sep,'/')
def show_plot():
    #fig=plt.gcf()
    #mpld3.display(fig)
    return
#tf.config.set_visible_devices([], 'GPU')
ver=12 #version of the model_weights to use. Refer to README for details.

#%%
class Rpeaks2Sig_Simulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,Fs_in,Fs_out,input_list=[],output_list=[],
                 Dsplit_mask_list=[],P_ID='',
                 path='./',sig_id='ppg_green',latent_size=6,logging=False,
                 epochs=2,batch_size=16,aux_loss_weights=[1],RNN_win_len=8,
                 win_step_size=2,current_time=None,exp_id=f'{ver}_1',
                 gen_model_config={'rnn_units':8,'disc_f_list':[8,16,32,32,64]}
                 ):
        '''
        in/output format: list of numpy arrays. 1st channel of output is 
        w_pk and w_l defined in seconds
        '''
        super().__init__(input_list,output_list)
        # TODO: Didn't change w_pk and w_l sizes for ECG as ECG needs smaller 
        # windows than ppg. But might need to change depending upon changing Fs
        self.Dsplit_mask_list=Dsplit_mask_list
        self.P_ID=P_ID
        self.path=path
        self.sig_id=sig_id
        self.Fs_in=Fs_in
        self.Fs_out=Fs_out
        self.latent_size=latent_size
        self.RNN_win_len=RNN_win_len #in secs
        self.win_step_size=win_step_size #in secs
        self.n_olap_wins=int(self.RNN_win_len/self.win_step_size)
        
        self.epochs=epochs
        self.batch_size=batch_size
        
        self.logging=logging
        self.exp_id= exp_id
        self.gen_model_config=gen_model_config

        #Add a time_stamp to experiment
        if current_time is None:
            self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.current_time = current_time
        
        self.gen_model_path=path+"model_weights_v{}/{}_{}_Morph_model".format(
                                                    ver,self.P_ID,self.sig_id)
        os.makedirs(self.gen_model_path,exist_ok=True)

        fname=glob.glob(self.gen_model_path+'/*.index')
        if len(fname)==0:
            print(f'\n Learning {self.sig_id} Morph gen_model... \n')
            self.log_prefix='../experiments/condWGAN/{}_{}/{}_{}_R2S'.format(
                        self.exp_id,self.current_time,self.P_ID,self.sig_id)
            
            self.gen_model=self.learn_gen_model(input_list,output_list,
                                Dsplit_mask_list,
                                self.gen_model_path,Fs_in=self.Fs_in,
                                Fs_out=self.Fs_out,save_flag=True,
                                show_plots=False,EPOCHS=self.epochs,
                                latent_size=self.latent_size,
                                batch_size=self.batch_size,
                                alw=aux_loss_weights,recon_epochs=0)
            del self.gen_model #delete after training
            
        
        #Load and Test
        #self.RNN_win_len=1# Reduce RNN win_len to reduce test time wastage
        self.win_step_size=1*self.RNN_win_len
            
        print(f'{self.sig_id} Morph gen_model exists. Loading ...')
        self.gen_model=self.load_gen_model(self.gen_model_path,self.Fs_out)
        print('Done!')
        #self.make_gen_plots(self.gen_model,int(self.w_pk*self.Fs_out))
        return 

    def create_gen_model(self,latent_size=6,
                     model_path='',stateful=False,batch_size=1,
                     optimizers = None,save_flag=True,aux_losses_weights=None):
        #Define gen layers
        gen_layer_list=[downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                        downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                        downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        downsample(filters=32, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                        upsample(filters=8, kernel_size=(1,3),strides=(1,2),
                                 ),
                        upsample(filters=8,kernel_size=(1,3),strides=(1,2)),
                        layers.GRU(self.gen_model_config['rnn_units'],
                                   return_sequences=True,stateful=stateful, 
                                   dropout=self.gen_model_config['gru_drop']),
                        layers.Conv1D(1,1,padding='same')]
        
        #Define Disc Layers
        f_list=self.gen_model_config['disc_f_list']#[8,16,32,32,64]
        #f_list=[8,16,16,32,32]
        k_list=[3,3,3,3,3]
        s_list=[2,2,2,2,2]
        act_list=len(f_list)*['relu']
        def_disc_layers=[]
        for i in range(len(f_list)):
            def_disc_layers.append(layers.Conv1D(f_list[i],k_list[i],
                            name='conv1d_disc_{}'.format(i),padding="same"))
            def_disc_layers.append(layers.MaxPooling1D(pool_size=s_list[i], 
                                    strides=s_list[i], padding="same",
                                    name='maxpool_disc_{}'.format(i)))
            def_disc_layers.append(layers.Activation(act_list[i],
                                    name='act_disc_{}'.format(i)))
            
        def_disc_layers.append(layers.Flatten(name='flat_disc_1'))
        def_disc_layers.append(layers.Dense(1,name='fc_disc_1'))
        
        # Create an instance of the model
        model = Model_CondWGAN(latent_size=latent_size,model_path=model_path,
                       gen_layers=gen_layer_list,disc_layers=def_disc_layers,
                       use_x=True, mode='GAN',
                       optimizers=optimizers,save_flag=save_flag,
                       aux_losses=[tf.keras.losses.MeanSquaredError()],
                       aux_losses_weights=aux_losses_weights,
                       n_classes=n_classes,Unet_reps=2)
        #model.summary()
        return model
    
    def learn_gen_model(self,input_list,output_list,Dsplit_mask_list,path4model,
                        Fs_in=100,Fs_out=25,save_flag=True,show_plots=True,
                        EPOCHS = 100,latent_size= 6,fig_no=200,
                        batch_size=32,alw=[2],recon_epochs=0):
        '''
        Learn a generative model
        sim_pks2sigs
        '''
        #logging stuff
        log_prefix=self.log_prefix
        train_log_dir = log_prefix + '/train'
        test_log_dir = log_prefix + '/test'
        stdout_log_file = log_prefix + '/stdout.log'
        checkpoint_dir = log_prefix + "/checkpoints"
        #figs_dir = log_prefix + "/figures"
        os.makedirs(log_prefix,exist_ok=True)
        if self.logging:
            origin_stdout,log_file=start_logging(stdout_log_file)
        
        def get_data():
            #RNN_win_len=300;step_size=100 #TODO: Override for debugging ECG
            train_in_list,train_out_list=[],[]
            val_in_list,val_out_list=[],[]
            for i in range(len(input_list)):
                #data split masks
                sel_mask_train,sel_mask_val,_=Dsplit_mask_list[i]
                print(f'Selected {np.sum(sel_mask_train)/len(sel_mask_train)} ' 
                      f' ratio of samples from class {i} for training')
                
                val_in_list.append(input_list[i][sel_mask_val])
                val_out_list.append(output_list[i][sel_mask_val])
                train_in_list.append(input_list[i][sel_mask_train])
                train_out_list.append(output_list[i][sel_mask_train])
            
            val_data=[np.concatenate(val_in_list,axis=0),
                      np.concatenate(val_out_list,axis=0)]
            train_data=[np.concatenate(train_in_list,axis=0),
                      np.concatenate(train_out_list,axis=0)]

            print(val_data[0].shape,val_data[1].shape,
                  train_data[0].shape,train_data[1].shape)
            
            return train_data,val_data
        
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        
        model=self.create_gen_model(latent_size=latent_size,save_flag=save_flag,
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
        check_graph(model.gen,shape_list=[(int(train_data[0].shape[1]/model.z_up_factor),
                                           latent_size),
                    (train_data[0].shape[1],train_data[0].shape[2])],
                    file_path=log_prefix+'/Generator.png')
        check_graph(model.disc,shape_list=[tuple(train_data[1].shape[1:]),
                    (train_data[0].shape[1],train_data[0].shape[2])],
                    file_path=log_prefix+'/Discriminator.png')
        check_graph(model,shape_list=[(train_data[0].shape[1],
                                       train_data[0].shape[2])],
                    file_path=log_prefix+'/condWGAN.png')
        
        #train_data_fixed_in=train_data[0]*1 #TODO: store for future need
        if recon_epochs:
            model.mode='GAN_recon'
            model.fit(data=[train_data,val_data],
                      summaries=[train_summary_writer,test_summary_writer],
                      epochs = recon_epochs, batch_size=int(batch_size))
        
        #Switching to GAN mode
        model.mode='GAN'
        model.fit(data=[train_data,val_data],
                  summaries=[train_summary_writer,test_summary_writer],
                  epochs = EPOCHS, batch_size=batch_size)
        
        #self.GAN_input=GAN_input
        #self.GAN_output=GAN_output
        
        if save_flag:
            # copy latest checkpoint_dir to gen_model_path
            copy_any(checkpoint_dir, path4model)
            
        if self.logging:
            stop_logging(origin_stdout,log_file)
        return model
    
    def load_gen_model(self,path,Fs_out,stateful=False):
        '''
        Load a model from the disk.
        '''
        model=self.create_gen_model(latent_size=self.latent_size,
                                    stateful=stateful,batch_size=1,
                                    model_path=path)
        #model.load_weights(path)
        model.ckpt.restore(model.manager.latest_checkpoint)
        print("Restored from {}".format(model.manager.latest_checkpoint))
        return model 
    
    def __call__(self,cond_sig,step_size_s=None,show_plots=False):
        '''
        arr_pks: Location of R-peaks determined from ECG
        k: No. of prominent basis to keep
        sim_pks2sigs
        '''
        #if win_step_size is None: win_step_size=self.win_step_size
        #step_size=int(win_step_size*self.Fs_out)        
        #cond_sig_windows=cond_sig
        RNN_win_len=int(self.RNN_win_len*self.Fs_out)

        if step_size_s is None:
            step_size=RNN_win_len#int(self.win_step_size*self.Fs_out)
        else:
            step_size=int(self.Fs_out*step_size_s)
            assert step_size<=RNN_win_len, f'step_size_s must be <= {int(RNN_win_len/self.Fs_tacho)}'


        # #Fragment signal into windows
        #cond_sig=cond_sig[:len(cond_sig)-int(len(cond_sig)%RNN_win_len)]
        #arr_pks=np.concatenate([arr_pks.reshape(-1,1),
        #            cond_sig[:len(arr_pks),-n_classes:]],axis=-1)
        #arr_interm=np.expand_dims(arr_interm,axis=1)
        cond_sig_windows=self.sliding_window_fragmentation([cond_sig],
                                        RNN_win_len,step_size)
        #arr_pks_windows=np.expand_dims(arr_pks_windows,axis=-1)
        
        #Generate Predictions
        arr=np.zeros([*cond_sig_windows.shape[:-1],1])
        rnn_state=None
        for i in range(cond_sig_windows.shape[0]):
            arr[i:i+1]=self.gen_model.predict([cond_sig_windows[i:i+1]],
                            rnn_state=rnn_state,rnn_state_out_no=step_size-1)
            rnn_state=self.gen_model.gen.rnn_state_out #update state
        
        # Defragment windows into continous signal
        #arr=arr.reshape(-1)
        arr=self.sliding_window_defragmentation([arr],RNN_win_len,step_size)
        if show_plots:
            stress_no=np.argmax(np.mean(cond_sig[:len(arr),1:6],axis=0))
            class_no=np.argmax(np.mean(cond_sig[:len(arr),6:],axis=0))
            plt.figure();plt.plot(cond_sig[:len(arr),0]);plt.plot(arr)
            plt.legend(['Input','Output']);plt.xlabel('Sample No.')
            plt.title(f'Stress={stress_no}, Class={class_no}')
            plt.grid(True)
        return arr,cond_sig[:len(arr)]

#%% Client
if __name__=='__main__':
    # Data Helpers
    import seaborn as sns
    sns.set()
    path='D:/Datasets/WESAD/'
    ckpt_path=proj_path+'/../data/post-training/'
    curr_time='20220219-035453'#None#'20211003-112847'
    exp_no=28
    #max_ppg_val=load_data.MAX_PPG_VAL
    #max_ecg_val=load_data.MAX_ECG_VAL
    all_class_ids=copy.deepcopy(load_data.class_ids)
    #class_name='S7'
    class_name='WESAD'
    win_len_s,step_s,latent_size=8,2,2
    bsize=load_data.bsize
    ppg_gen_model_config={'rnn_units':8,'disc_f_list':[8,16,16,32,64],
                          'gru_drop':0.}
    ecg_gen_model_config={'rnn_units':8,'disc_f_list':[8,16,16,32,64],
                          'gru_drop':0.}
    
    #load_data.class_ids={'S7':all_class_ids['S7']}
    #TODO: Uncomment and indent loop for subject-specific models
    #for class_name in ['S5']:
    for class_name in list(all_class_ids.keys()):
        load_data.class_ids={class_name:all_class_ids[class_name]}
        
        #Get Train Data for simulator
        plt.close('all')
        #path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #        
        # train_data_filename='processed_RC2EP_Kalidas_train_data.pkl'
        # if not os.path.isfile(path+train_data_filename):
        #     input_list,output_list,musig_dict=load_data.get_train_data(path,mode='R2S')
        #     with open(path+train_data_filename, 'wb') as fp:
        #         pickle.dump([input_list,output_list,musig_dict], fp)
                
        # with open (path+train_data_filename, 'rb') as fp:
        #     itemlist = pickle.load(fp)
        # input_list,output_list,musig_dict=itemlist
        
        filename = (f'{proj_path}/../data/pre-training/WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{bsize}.'
                    'pickle')
        if os.path.isfile(filename):
            with open (filename, 'rb') as fp:
                _,Dsplit_mask_dict = pickle.load(fp)
            input_dict,output_dict,musig_dict,Dsplit_mask_dict=(load_data.
                    get_train_data(path,mode='R2S',win_len_s=win_len_s,
                    step_s=step_s,Dsplit_mask_dict=Dsplit_mask_dict))
        else:
            input_dict,output_dict,musig_dict,Dsplit_mask_dict=(load_data.
                    get_train_data(path,mode='R2S',win_len_s=win_len_s,
                    step_s=step_s,Dsplit_mask_dict=None))
            #Save Dsplit_masks
            with open(filename, 'wb') as handle:
                pickle.dump([musig_dict,Dsplit_mask_dict], handle)
        
        #See ECG
        #aa=output_list[0][:,0:1].reshape(-1)
        #plt.figure();plt.plot(aa)
        
        #Create Simulator using train data
        ckpt_path=proj_path+'/../data/post-training/'
    
        sim_pks2ppg=Rpeaks2Sig_Simulator(Fs_in=load_data.Fs_ppg_new,
                    Fs_out=load_data.Fs_ppg_new,input_list=input_dict['ppg'],
                    output_list=output_dict['ppg'],
                    Dsplit_mask_list=[Dsplit_mask_dict['ppg'][c] 
                            for c in Dsplit_mask_dict['Dspecs']['key_order']],
                    P_ID=class_name,path=ckpt_path,sig_id='ppg',
                    latent_size=latent_size,logging=True,epochs=500,batch_size=32,
                    aux_loss_weights=[5],RNN_win_len=win_len_s,win_step_size=step_s,
                    current_time=curr_time,exp_id=f'{ver}_{exp_no}',
                    gen_model_config=ppg_gen_model_config)
        sim_pks2ecg=Rpeaks2Sig_Simulator(Fs_in=load_data.Fs_ecg_new,
                    Fs_out=load_data.Fs_ecg_new,input_list=input_dict['ecg'],
                    output_list=output_dict['ecg'],
                    Dsplit_mask_list=[Dsplit_mask_dict['ecg'][c] 
                            for c in Dsplit_mask_dict['Dspecs']['key_order']],
                    P_ID=class_name,path=ckpt_path,sig_id='ecg',
                    latent_size=latent_size,logging=True,epochs=500,batch_size=32,
                    aux_loss_weights=[5],RNN_win_len=win_len_s,win_step_size=step_s,
                    current_time=sim_pks2ppg.current_time,
                    exp_id=sim_pks2ppg.exp_id,gen_model_config=ecg_gen_model_config)
        del sim_pks2ppg,sim_pks2ecg
#%% Inference
    filename = (f'{proj_path}/../data/pre-training/WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{bsize}.'
                'pickle')
    if os.path.isfile(filename):
        with open (filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                       'Run get_train_data in R2S mode first.')
            
    load_data.class_ids=copy.deepcopy(all_class_ids) #restore class_ids
    Fs_ppg_new,Fs_ecg_new=load_data.Fs_ppg_new,load_data.Fs_ecg_new
    #class_name='WESAD'
    #class_name='S4'
    #for class_name in [f'S{j}' for j in range(5,6)]:
    for class_name in list(all_class_ids.keys())[:9]:
        #load_data.class_ids={class_name:all_class_ids[class_name]}
        #Create Simulator using train data
        #sim_pks2sigs=Rpeaks2Sig_Simulator(input_list=[],output_list=[],P_ID=class_name,
        #        path=ckpt_path,latent_size=2)
            #Use simulator to produce synthetic output given input
        sim_pks2ppg=Rpeaks2Sig_Simulator(Fs_in=load_data.Fs_ppg_new,
                    Fs_out=load_data.Fs_ppg_new,
                    P_ID=class_name,path=ckpt_path,sig_id='ppg',
                    latent_size=latent_size,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s,
                    exp_id=f'{ver}_{exp_no}',
                    gen_model_config=ppg_gen_model_config)
        sim_pks2ecg=Rpeaks2Sig_Simulator(Fs_in=load_data.Fs_ecg_new,
                    Fs_out=load_data.Fs_ecg_new,
                    P_ID=class_name,path=ckpt_path,sig_id='ecg',
                    latent_size=latent_size,logging=False,batch_size=32,
                    RNN_win_len=win_len_s,win_step_size=step_s,
                    exp_id=sim_pks2ppg.exp_id,
                    gen_model_config=ecg_gen_model_config)
    #for class_name in list(all_class_ids.keys()):
        test_in,test_out_for_check,_,Dsplit_mask_dict=load_data.get_test_data(
                            path+class_name,mode='R2S',win_len_s=win_len_s,
                            step_s=step_s,Dsplit_mask_dict=Dsplit_mask_dict)
        
        # #test_time=int((1-load_data.test_ratio)*len(test_in['ppg']))/Fs_ppg_new
        # ppg_synth,test_in['ppg']=sim_pks2ppg(test_in['ppg'])
        # ecg_synth,test_in['ecg']=sim_pks2ecg(test_in['ecg'])
        
        # test_out_for_check['ppg']=test_out_for_check['ppg'][:len(ppg_synth)]
        # test_out_for_check['ecg']=test_out_for_check['ecg'][:len(ecg_synth)]
        # #test_out_for_check['ppg']=test_out_for_check['ppg'][:len(ppg_synth)]
        # #test_out_for_check['ecg']=test_out_for_check['ecg'][:len(ecg_synth)]
        
        class_no,test_seq_no=0,0
        cond_ppg_wins=test_in['ppg'][class_no]#[test_seq_no]
        ppg_real_wins=test_out_for_check['ppg'][class_no]#[test_seq_no]
        cond_ecg_wins=test_in['ecg'][class_no]#[test_seq_no]
        ecg_real_wins=test_out_for_check['ecg'][class_no]#[test_seq_no]
        
        # Add noise
        #plt.figure(99);plt.plot(cond_ecg_wins[0,:,0])
        #cond_ecg_wins[:,:,0]+=np.random.normal(0,0.1,cond_ecg_wins[:,:,0].shape)
        #plt.plot(cond_ecg_wins[0,:,0],'--')
        
        
        # defragment windows into continous signal
        cond_ppg,ppg_real=(load_data.sliding_window_defragmentation([
            cond_ppg_wins,ppg_real_wins],win_len_s*Fs_ppg_new,
            step_s*Fs_ppg_new))
        cond_ecg,ecg_real=(load_data.sliding_window_defragmentation([
            cond_ecg_wins,ecg_real_wins],win_len_s*Fs_ecg_new,
            step_s*Fs_ecg_new))
        
        #Generate synthetic data
        ppg_synth,cond_ppg=sim_pks2ppg(cond_ppg)
        ecg_synth,cond_ecg=sim_pks2ecg(cond_ecg)
        
        #time for plotting
        t_ppg=np.arange(len(ppg_synth))/Fs_ppg_new
        t_ecg=np.arange(len(ecg_synth))/Fs_ecg_new
        #clip real data
        ecg_real=ecg_real[:len(ecg_synth)]
        ppg_real=ppg_real[:len(ppg_synth)]
        
        # #second generation to check variety
        # ppg_synth_2,cond_ppg=sim_pks2ppg(cond_ppg)
        # ecg_synth_2,cond_ecg=sim_pks2ecg(cond_ecg)
        # plt.figure();ax1=plt.subplot(211)
        # plt.plot(t_ecg,ecg_real,t_ecg,ecg_synth,t_ecg,ecg_synth_2,'--')
        # plt.legend(['Real','Synth','Synth_2']);plt.title('ECG')
        # plt.subplot(212, sharex=ax1)
        # plt.plot(t_ppg,ppg_real,t_ppg,ppg_synth,t_ppg,ppg_synth_2,'--')
        # plt.title('PPG')
        # #plt.legend(['Real','Synth','Synth_2'])
        
        #make some plots
        #start,end=clip_ecg
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plt.plot(t_ecg,ecg_real,t_ecg,cond_ecg[:,0])
        leg_list=['True','R-peaks']+[f'Synth_{j}' for j in range(5)]
        mark_list=['r','g','c','m','k']
        for i in range(5):
            mask=cond_ecg[:,i+1].astype(bool)
            plt.plot(t_ecg[mask],ecg_synth[mask],mark_list[i])
        plt.legend(leg_list,loc='lower right')
        #plt.xlabel('Time (s.)')
        plt.title(f'ECG for {class_name}');plt.grid(True)
        
        #fig2 = plt.figure()
        #ax2 = fig2.add_subplot(212, sharex=ax1)
        plt.subplot(212, sharex=ax1)
        plt.plot(t_ppg,ppg_real,t_ppg,cond_ppg[:,0])
        leg_list=['True','R-peaks']+[f'Synth_{j}' for j in range(5)]
        mark_list=['r','g','c','m','k']
        for i in range(5):
            mask=cond_ppg[:,i+1].astype(bool)
            plt.plot(t_ppg[mask],ppg_synth[mask],mark_list[i])
        plt.legend(leg_list,loc='lower right')
        plt.xlabel('Time (s.)')
        plt.title((f'PPG for {class_name}'));plt.grid(True)
        

#%%
        #plt.suptitle(f'Test data time starts at {test_time} s.')
        
        
        delta=5
        rpeaks_ecg=np.arange(len(cond_ecg))[cond_ecg[:,0].astype(bool)]
        rpeaks_ppg=np.arange(len(cond_ppg))[cond_ppg[:,0].astype(bool)]

        #rpeaks={'ppg_R_Peaks':rpeaks}
        #t_lim=100 #limit in sec.
        #ecg_real=ecg_real
        
        #Check ecg morphs
        signal_peak, morph_features = nk.ecg_delineate(ecg_real[delta:], 
            rpeaks_ecg[1:]-delta, sampling_rate=Fs_ecg_new, show=True,
            method='peaks', show_type='peaks')
        plt.title(class_name);
        signal_peak, morph_features = nk.ecg_delineate(ecg_synth[delta:], 
            rpeaks_ecg[1:]-delta, sampling_rate=Fs_ecg_new, show=True,
            method='peaks', show_type='peaks')
        
        #Check ppg morphs
        wins_ppg=load_data.get_windows_at_peaks(rpeaks_ppg[1:-1]-delta,
                    ppg_real[delta:].flatten(),w_pk=25,w_l=10,
                    show_plots=True,n_wins=9,ylims=[-0.5,0.5])
        wins_ppg=load_data.get_windows_at_peaks(rpeaks_ppg[1:-1]-delta,
                    ppg_synth[delta:].flatten(),w_pk=25,w_l=10,show_plots=True,
                    n_wins=30,ylims=[-0.5,0.5])
        #plt.figure()
        # ppg_morphs = nk.ecg_segment(ppg_synth[delta:].flatten(), 
        #     rpeaks_ppg[1:]-delta, sampling_rate=Fs_ppg_new, show=True)
        # fig, ax = plt.subplots()
        # ppg_morphs.Label = ppg_morphs.Label.astype(int)
        # for label in ppg_morphs.Label.unique():
        #     epoch_data = ppg_morphs[ppg_morphs.Label == label]
        #     ax.plot(epoch_data.Time, epoch_data.Signal, color="grey", 
        #             alpha=0.2, label="_nolegend_")
    
        del sim_pks2sigs

    #synth_ecg_out,test_in_ecg,clip_ecg,synth_ppg_out,test_in_ppg,clip_ppg,ppg_pks,ecg_pks=sim_pks2sigs(test_in)

    # ppg_synth*=musig_dict['ppg']['sigma'] #rescale
    # ppg_synth+=musig_dict['ppg']['mu'] #add back mean
    # ecg_synth*=musig_dict['ecg']['sigma'] #rescale
    # ecg_synth+=musig_dict['ecg']['mu'] #add back mean
    
    #with open('./figures/fig_REPv3.pickle', 'wb') as file:
    #    pickle.dump([fig1,fig2],file