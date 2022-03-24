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
import copy
#import mpld3

import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from lib.simulator_for_CC import Simulator
    #from lib.data import load_data_sense as load_data
    from lib.data import load_data_wesad as load_data
    from lib.model_CondWGAN import Model_CondWGAN, downsample, upsample
    from lib.utils import copy_any, start_logging, stop_logging, check_graph
    from lib.utils import get_leading_tacho, get_uniform_tacho
else:
    from .lib.simulator_for_CC import Simulator
    #from .lib.data import load_data_sense as load_data
    from .lib.data import load_data_wesad as load_data
    from .lib.model_CondWGAN import Model_CondWGAN, downsample, upsample
    from .lib.utils import copy_any, start_logging, stop_logging, check_graph
    from .lib.utils import get_leading_tacho, get_uniform_tacho



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
ver=12 #version of the model_weights to use. Refer to README for details.
#%%
class HR2Rpeaks_Simulator(Simulator):
    '''
    Produces peak train from given smooth HR data
    '''
    def __init__(self,input_list=[],output_list=[],Dsplit_mask_list=[],
                 RNN_win_len_s=8,step_size_s=2,P_ID='',path='./',Fs_HR=25,
                 Fs_tacho=5,latent_size=2,exp_id=f'{ver}_9',epochs=2000):
        '''
        in/output format: list of numpy arrays
        '''
        super(HR2Rpeaks_Simulator,self).__init__(input_list,output_list)
        self.Dsplit_mask_list=Dsplit_mask_list
        self.P_ID=P_ID
        self.path=path
        self.Fs_HR=Fs_HR
        self.Fs_tacho=Fs_tacho
        assert (self.Fs_HR%self.Fs_tacho)==0, 'Fs_HR MUST be a multiple of Fs_tacho'

        #self.HR_win_len_s=HR_win_len_s
        self.latent_size=latent_size
        self.RNN_win_len=int(self.Fs_tacho*RNN_win_len_s)
        self.step_size=int(self.Fs_tacho*step_size_s)
        self.n_olap_wins=int(self.RNN_win_len/self.step_size)
        self.exp_id=exp_id
        #Form weights for hrv_freq_loss
        self.hrv_freqs=np.fft.rfftfreq(self.RNN_win_len, d=1/self.Fs_tacho)
        W_hrv_freq=np.ones(len(self.hrv_freqs))
        W_hrv_freq[self.hrv_freqs<=0.5]=4
        W_hrv_freq[((self.hrv_freqs>0.15) & (self.hrv_freqs<=0.4))]=6
        #W_hrv_freq[self.hrv_freqs==0.]=10
        #Sum of W=4*11+6*10+80*1=184 vs. 19 earlier
        #Sum of W=10+8*4+8*6+64*1=154 vs. 19 earlier

        self.W_hrv_freq=tf.expand_dims(tf.convert_to_tensor(W_hrv_freq,
                                        dtype=tf.float32),axis=0)
        
        
        # gen_model is a 2nd order p(RR_prev,RR_next| HR cluster) distribution
        self.gen_model_path=path+"model_weights_v{}/{}_HRV_model".format(ver,P_ID)
        
        fname=glob.glob(self.gen_model_path+'/checkpoint')
        if len(fname)==0:
            print('\n Learning HRV Gen Model... \n')
            self.gen_model=self.learn_gen_model(save_flag=True,EPOCHS = epochs, 
                                                latent_size=self.latent_size)
            del self.gen_model
        
        #Load and Test
        #self.RNN_win_len=1# Reduce RNN win_len to reduce test time wastage
        #self.step_size=1*self.RNN_win_len
        print('HRV Gen Model Exists. Loading ...')
        self.gen_model=self.load_gen_model(self.gen_model_path)
        print('Done!')
        #self.make_gen_plots(self.ecg_gen_model,int(self.w_pk*self.Fs_ecg))
        return
    
    def load_gen_model(self,path,stateful=True):
        '''
        Load a model from the disk.
        '''
        model=self.create_gen_model(stateful=stateful,
                batch_size=1,model_path=path, latent_size=self.latent_size)
        #model.load_weights(path)
        model.ckpt.restore(model.manager.latest_checkpoint)
        print("Restored from {}".format(model.manager.latest_checkpoint))
        return model
    

    def _hrv_freq_loss(self,y_true,y_hat): 
        def get_PSD(tacho):
            tacho = tf.squeeze(tacho,axis=[-1])
            fft_U=tf.signal.rfft(tacho,fft_length=[nfft])
            PSD=tf.math.divide(tf.square(tf.abs(fft_U)),norm_factr)
            #Sub-Select the freq band of interest
            #PSD=PSD_U[:,1:len_fft_out] #skipping DC value
            #PSD_U=tf.reshape(PSD_U,shape=[-1,1])
            return PSD
        # Get some common parameters
        nfft,last_dim=y_true.get_shape().as_list()[1:]
        assert last_dim==1, 'last dimension is not 1. Please check the tacho tensor'
        #dsamp_factor=int(Fs_tacho/Fs_fft)
        #len_fft_out=int(nfft/(2*dsamp_factor))#+1
        #tacho_fft=tacho[:,::dsamp_factor] #dsample tacho
        
        #Find periodogram by simple welch method
        norm_factr=tf.constant(int(nfft/2)+1,dtype=tf.float32)
        #get PSDs
        PSD_true=get_PSD(y_true)
        PSD_hat=get_PSD(y_hat)
        loss=tf.reduce_mean(self.W_hrv_freq*tf.square(PSD_true-PSD_hat))
        #return  PSD_true,PSD_hat,loss #TODO: for debugging
        return loss
        
    def create_gen_model(self,shape_in=None,shape_out=None,latent_size=4,
                     model_path='',stateful=False,batch_size=1,
                     optimizers = None,save_flag=True):
        '''
        model_path must be a dir
        '''
        # Create an instance of the model
        # Gen layers
        gen_layer_list=[downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                downsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                downsample(filters=16, kernel_size=(1,3),strides=(1,2)),
                downsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                upsample(filters=16, kernel_size=(1,5),strides=(1,5)),
                upsample(filters=16, kernel_size=(1,3),strides=(1,2)),
                upsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                upsample(filters=8, kernel_size=(1,3),strides=(1,2)),
                layers.GRU(64,return_sequences=True,stateful=stateful,dropout=0.),
                layers.Conv1D(1,1,padding='same')]
        
        #Disc Layers
        f_list=[8,16,32,64,64]
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
        
        model = Model_CondWGAN(latent_size=latent_size,model_path=model_path,
                       gen_layers=gen_layer_list,disc_layers=def_disc_layers,
                       use_x=True, mode='GAN',
                       optimizers=optimizers,save_flag=save_flag,
                       aux_losses=[self._hrv_freq_loss],
                       aux_losses_weights=[1e-3],feat_loss_weight=0,
                       Unet_reps=4)
        #model.summary()
        return model
    
    def learn_gen_model(self,latent_size=4,save_flag=True,make_plots=True,
                        EPOCHS = 100, logging=True):
        #logging stuff
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_prefix='../experiments/condWGAN/{}_{}/HR2R'.format(self.exp_id,current_time)
        train_log_dir = log_prefix + '/train'
        test_log_dir = log_prefix + '/test'
        stdout_log_file = log_prefix + '/stdout.log'
        checkpoint_dir = log_prefix + "/checkpoints"
        #figs_dir = log_prefix + "/figures"
        os.makedirs(log_prefix,exist_ok=True)
        if logging:
            origin_stdout,log_file=start_logging(stdout_log_file)
            
        input_list,output_list=self.input,self.output

        train_in_list,train_out_list=[],[]
        val_in_list,val_out_list=[],[]
        for i in range(len(input_list)):
            #data split masks
            sel_mask_train,sel_mask_val,_=self.Dsplit_mask_list[i]
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
        self.model_in=train_data[0]
        self.model_out=train_data[1]

        
        #tensorboard stuff
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        model=self.create_gen_model(latent_size=latent_size,save_flag=save_flag,
                                    model_path=checkpoint_dir)
        

        #TODO: Uncomment these to check the graphs
        print(self.model_in.dtype,self.model_out.dtype)
        print([(*self.model_in.shape[:2],latent_size),tuple(self.model_in.shape[:])])
        check_graph(model.gen,shape_list=[(int(self.model_in.shape[1]/model.z_up_factor),latent_size),
                                          tuple(self.model_in.shape[1:])],
                                              file_path=log_prefix+'/Generator.png')
        check_graph(model.disc,shape_list=[tuple(self.model_out.shape[1:]),
                                           tuple(self.model_in.shape[1:])],
                                              file_path=log_prefix+'/Discriminator.png')
        check_graph(model,shape_list=[tuple(self.model_in.shape[1:])],
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
    
    def __call__(self,cond_curve,Fs_out=100,step_size_s=None,show_plots=False):
        
        #Get complete tacho
        RNN_win_len=self.RNN_win_len
        if step_size_s is None:
            step_size=self.RNN_win_len#self.step_size
        else:
            step_size=int(self.Fs_tacho*step_size_s)
            assert step_size<=RNN_win_len, f'step_size_s must be <= {int(RNN_win_len/self.Fs_tacho)}'
        cond_tacho=cond_curve*1
        #cond_tacho[:,C_HR:C_HR+1]=cond_tacho[:,C_HR:C_HR+1]/60 #convert HR to BPS
        #cond_tacho=cond_tacho[:len(cond_tacho)-int(len(cond_tacho)%RNN_win_len)]#.reshape(-1) #clip end aptly
        cond_tacho_windows=self.sliding_window_fragmentation([cond_tacho],
                                                   RNN_win_len,step_size)
        
        if len(cond_tacho_windows.shape)==2:
            cond_tacho_windows=np.expand_dims(cond_tacho_windows,axis=-1)
        #Stitch them together
        arr_tacho=np.zeros([*cond_tacho_windows.shape[:-1],1],dtype=np.float32)
        rnn_state=None
        for i in range(cond_tacho_windows.shape[0]):
            arr_tacho[i:i+1]=self.gen_model.predict([cond_tacho_windows[i:i+1]],
                        rnn_state=rnn_state,rnn_state_out_no=step_size-1)
            rnn_state=self.gen_model.gen.rnn_state_out #update state
        
        
        # Defragment windows into continous signal
        #arr_tacho=arr_tacho.reshape(-1)
        arr_tacho=self.sliding_window_defragmentation([arr_tacho],RNN_win_len,
                                                       step_size).flatten()
        
        
        # Create a HR to tacho mapping
        #cond_curve=cond_curve[:(len(cond_tacho)-1)*factr_tacho+1]
        #cond_curve=cond_curve[:len(cond_tacho)]
        cond_curve=cond_curve[:len(arr_tacho)]

        t_steps=np.arange(len(cond_curve))/self.Fs_tacho
        get_tacho = sp.interpolate.interp1d(t_steps,arr_tacho,'cubic',axis=0)
        
        #plt.figure()
        #plt.plot(t_steps[::factr_tacho],arr_tacho,'o-',t_steps,lead_tacho(t_steps),'r--')
        #plt.legend(['True','Upsampled'])
        
        #Use the mapping to form the R-peak train
        #factr=(Fs_out/self.Fs_HR)
        
        #TODO: Changed arr_pk creation according to n_samples in input
        # There will be 1/fs length of zeros at the end as a result since 
        # 0-31.8s is the time span of tacho. This may need rethinking.
        arr_pk=np.zeros(int(len(arr_tacho)/self.Fs_tacho*Fs_out))#(i+1) 

        #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
        #plt.figure();plt.plot(gauss)
        #HR_t=HR_curve[i,0]#np.mean(HR_curve[0:i])
        #while i < (len(cond_curve)-1):
        flag_max_iters=True #for checking how loop exited
        min_RR=(1/(3.5+0.5)) #Assuming 3.5 Hz as max humanly possible HR and 0.5Hz for HRV within 3.5 Hz
        max_iters=np.ceil(t_steps[-1]/min_RR).astype(int)
        
        #TODO: Choose i randomly assuming 3.5Hz max HR
        last_t_step=int(t_steps[-1]*Fs_out)
        #i=last_t_step
        i=1+np.random.randint(int((t_steps[-1]-min_RR)*Fs_out),last_t_step)
        
        for _ in range(max_iters):
            #HR_t=HR_curve[i,0]
            #idx=int(factr*i)
            #arr_pk[idx]=1
            arr_pk[i]=1
            #RR_next=lead_tacho(i)
            RR_prev=get_tacho((i/Fs_out))

            #update for next
            #i+=int(RR_next*self.Fs_HR)
            i-=int(RR_prev*Fs_out)
            if i<0:
                flag_max_iters=False
                break
            
        #Random flip for testing only
        #flag_max_iters=np.random.uniform()>0.5 #TODO: For debugging only
        
        if flag_max_iters:
            print("Couldn't find appropriate peak-train. Returning empty arrays")
            stress_no=np.argmax(np.mean(cond_curve[:,1:6],axis=0))
            class_no=np.argmax(np.mean(cond_curve[:,6:],axis=0))
            plt.figure();plt.plot(cond_curve[:,0]);plt.plot(arr_tacho)
            plt.legend(['Input','Output']);plt.xlabel('Sample No.')
            plt.title(f'Failed RT^{-1}. Stress={stress_no}, Class={class_no}')
            plt.grid(True)
            arr_pk=np.zeros(0)
            cond_curve=np.zeros([0,*cond_curve.shape[1:]])
            arr_tacho=np.zeros(0)
            return arr_pk,cond_curve,arr_tacho
        
        if show_plots:
            stress_no=np.argmax(np.mean(cond_curve[:,1:6],axis=0))
            class_no=np.argmax(np.mean(cond_curve[:,6:],axis=0))
            plt.figure();plt.plot(cond_curve[:,0]);plt.plot(arr_tacho)
            plt.legend(['Input','Output']);plt.xlabel('Sample No.')
            plt.title(f'Stress={stress_no}, Class={class_no}')
            plt.grid(True)
        return arr_pk,cond_curve,arr_tacho

#%% Sample Client
if __name__=='__main__':
    #Get Train Data for simulator
    plt.close('all')
    path='D:/Datasets/WESAD/'
    win_len_s=8;step_s=2;Fs_tacho=5;latent_size=4
    bsize=load_data.bsize
    input_list,output_list=[],[]
    all_class_ids=copy.deepcopy(load_data.class_ids)
    ckpt_path=proj_path+'/../data/post-training/'
    P_ID='WESAD'

    # (kalidas,7306,8.6s), (nabian,7270,9.2s), (neurokit,7253,7.6), 
    # (pantompkins1985, 7346, 14.4), (hamilton2002, 7355, 11), (elgendi2010, 7332, 19.35)
# =============================================================================
#     start_time=time.time()
#     test_in,test_out_for_check=load_data.get_test_data(path+'S2',mode='HR2R',win_len_s=win_len_s
#                                              ,step_s=step_s,Fs_pks=100)
#     print(f'{time.time()-start_time} s.')
# =============================================================================
    #class_name='S10'
    
    # train_data_filename='processed_HRC2R_Kalidas_train_data.pkl'
    # if not os.path.isfile(path+train_data_filename):
    #     #load_data.class_ids={class_name:all_class_ids[class_name]}
    #     input_list,output_list=load_data.get_train_data(path,'HR2R',win_len_s,step_s,Fs_tacho)
    #     with open(path+train_data_filename, 'wb') as fp:
    #         pickle.dump([input_list,output_list], fp)
            
    # with open (path+train_data_filename, 'rb') as fp:
    #     itemlist = pickle.load(fp)
        
    #input_list,output_list=[],[]
    #input_list,output_list=itemlist
    #define constants
    #path_prefix= os.path.dirname(os.path.abspath(__file__)).replace(os.sep,'/') #'C:/Users/agarwal.270/Box/'
    
    filename = (f'{proj_path}/../data/pre-training/WESAD_musig_Dsplit_w{win_len_s}s{step_s}b{bsize}.'
                'pickle')
    if os.path.isfile(filename):
        with open (filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                       'Run get_train_data in R2S mode first.')
    
    # class_name='S7'
    # load_data.class_ids={class_name:all_class_ids[class_name]}
    
    input_list,output_list,Dsplit_mask_dict=load_data.get_train_data(path,
                        'HR2R',win_len_s,step_s,Fs_tacho,Dsplit_mask_dict)
    
    Dsplit_mask_list=[Dsplit_mask_dict['hrv'][c] 
                      for c in Dsplit_mask_dict['Dspecs']['key_order']]
            
    #Train
    sim_HR2pks=HR2Rpeaks_Simulator(input_list,output_list,
                    Dsplit_mask_list=Dsplit_mask_list,
                    RNN_win_len_s=win_len_s+(load_data.test_bsize-1)*step_s,
                    step_size_s=step_s,P_ID=P_ID,path=ckpt_path,Fs_HR=Fs_tacho,
                    Fs_tacho=Fs_tacho,latent_size=latent_size,
                    exp_id=f'{ver}_25',epochs=3000)
    
    del sim_HR2pks
    
    # %%
    #Test Simulator
    #Use simulator to produce synthetic output given input
    #load_data.class_ids={f'S{k}':v for v,k in enumerate(list(range(2,12))+list(range(13,18)))}
    all_class_ids=copy.deepcopy(load_data.class_ids)
    pred_step=32#step_s
    #Load model
    sim_HR2pks=HR2Rpeaks_Simulator(
                RNN_win_len_s=win_len_s+(load_data.test_bsize-1)*step_s,
                step_size_s=step_s,P_ID=P_ID,path=ckpt_path,Fs_HR=Fs_tacho,
                Fs_tacho=Fs_tacho,latent_size=latent_size)
    #del sim_HR2pks
    #load_data.class_ids:
    #class_name='S5'
    #for class_name in ['S5']:
    for class_name in list(all_class_ids.keys())[:2]:
        # load_data.class_ids={'S7':all_class_ids['S7']}
        # test_in,test_out_for_check=load_data.get_test_data(path+'S7',mode='HR2R',win_len_s=win_len_s
        #                                      ,step_s=step_s,Fs_pks=100)
        # HR_S7=test_in[:,0]
        
        # #load_data.class_ids={class_name:all_class_ids[class_name]}
        # test_in,test_out_for_check,test_arr_pks=load_data.get_test_data(path+class_name,
        #                                     mode='HR2R',win_len_s=win_len_s
        #                                      ,step_s=step_s,Fs_tacho=Fs_tacho)
        # lenth=min(len(HR_S7),len(test_in))
        # test_in=test_in[:lenth]
        # test_in[:,0]=HR_S7[:lenth]
        list_cond_HRV,list_HRV,Dsplit_mask_dict=load_data.get_test_data(
                        path+class_name,mode='HR2R',win_len_s=win_len_s,
                        step_s=step_s,Fs_tacho=Fs_tacho,
                        Dsplit_mask_dict=Dsplit_mask_dict)
        
        class_no,test_seq_no=0,0
        cond_HRV_wins=list_cond_HRV[class_no]#[test_seq_no]
        HRV_real_wins=list_HRV[class_no]#[test_seq_no]

        # Add noise
        #plt.figure(99);plt.plot(cond_ecg_wins[0,:,0])
        #cond_ecg_wins[:,:,0]+=np.random.normal(0,0.1,cond_ecg_wins[:,:,0].shape)
        #plt.plot(cond_ecg_wins[0,:,0],'--')
        
        
        # defragment windows into continous signal
        test_in,test_out_for_check=(load_data.sliding_window_defragmentation([
            cond_HRV_wins,HRV_real_wins],
            ((load_data.test_bsize-1)*step_s+win_len_s)*Fs_tacho,
            step_s*Fs_tacho))
    
        
        #Fs_ppg=25;Fs_ecg=100
        arr_pk_synth,test_in,arr_tacho_synth=sim_HR2pks(test_in,
                                                        step_size_s=pred_step)
        # test_out_for_check=test_out_for_check[:len(test_out_for_check)-
        #                 int(len(test_out_for_check)%sim_HR2pks.RNN_win_len)]
        test_out_for_check=test_out_for_check[:len(test_in)]
                                              
        t_tacho=np.arange(len(arr_tacho_synth))/Fs_tacho
        plt.figure(100);plt.plot(t_tacho,test_out_for_check.flatten(),'-')
        plt.plot(t_tacho,arr_tacho_synth,'--')
        
        # #Plot some stuff
        # fig=plt.figure()  
        # plt.plot(test_out_for_check[19:])
        # plt.plot(arr_pk)
        # plt.legend(['True','Synthetic'])
        # #mpld3.display(fig)
        # show_plot()
        
        #Spectral analysis of tachos
        #lenth=len(test_out_for_check)
        #load_data.plot_periodograms(test_out_for_check.flatten(),
        #                            arr_tacho_synth,test_in,Fs_tacho=5)
        load_data.plot_periodograms(test_out_for_check.flatten(),
                            arr_tacho_synth,test_in,Fs_tacho=5)
        plt.suptitle(class_name)
        # load_data.plot_STFT_tacho(test_out_for_check.flatten(),arr_tacho_synth,
        #                           test_in[:,0],Fs_tacho=5)
        
        # #HRV analysis of arr_pks
        # #TODO: May need to edit these nk functions to directly input rri/tacho
        # Fs=100
        # hrv_features = nk.hrv(test_arr_pks, sampling_rate=Fs, show=True);show_plot()
        # plt.suptitle(class_name)
        # hrv_features_synth = nk.hrv(arr_pk_synth, sampling_rate=Fs, show=True);show_plot()
        # plt.suptitle(class_name)
    
    
    #plt.figure(100);plt.legend(['S3', 'S7', 'S10', 'S11', 'S15'])
    
    #hrv_lomb = nk.hrv_frequency(test_out_for_check[19:], sampling_rate=100, show=True, psd_method="lomb");show_plot()
    #hrv_lomb_synth = nk.hrv_frequency(arr_pk, sampling_rate=100, show=True, psd_method="lomb");show_plot()

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
