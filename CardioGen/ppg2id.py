# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:14:19 2020

@author: agarwal.270a
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle
import glob
import pandas as pd
import copy

from tensorflow.keras.activations import relu
import datetime

from lib.data import load_data_wesad_eval as load_data
from lib.data.load_data_wesad import Rpeak2HR, sliding_window_fragmentation
from lib.custom_layers import upsample, downsample
from lib.utils import make_data_pipe 
from lib.utils import start_logging, stop_logging
from lib.utils import make_x2xeval_plots

from sklearn.metrics import confusion_matrix
#from pathlib import Path
#curr_dir=Path.cwd()
#root_dir=curr_dir.parents[1]

import Augmentor_ecg2ppg_st_id as augmentor
data_path='../data/pre-training/WESAD/'

augmentor.data_path=data_path
save_name='WESAD_synth_h30_m28/P2StId'
aug_suffix='s_c'
proj_path='.'
ver=12
#win_len_s,step_s,bsize=augmentor.win_len_s,augmentor.step_s,augmentor.bsize

tf.keras.backend.set_floatx('float32')
#Hyperparameter specs
morph_win_len=4
win_len_s,step_s=8,2 #in sec
bsize,bstep=2,1
Fs_new=load_data.Fs_ppg_P2St
sample_win_len=((bsize-1)*step_s+win_len_s)*Fs_new
sample_step_size=bstep*step_s*Fs_new
#TODO: Can change this later, 2/3/4 are arbitrary choices after profs suggestion
#AE_reps=2 #no. of copies of AE internally
#HR_win_len=sample_win_len*3
ppg_win_len=sample_win_len#*AE_reps
#tacho_win_len=ppg_win_len-int((win_past+win_futr)*Fs_new)

#TODO: Started using this instead of redefining the function
seq_format_function_P2St=augmentor.seq_format_function_P2St
common_batch_size=64 #32
#update_freq_TB=1000*int(64/common_batch_size)#To keep n_samples per TB update same


all_class_ids=copy.deepcopy(load_data.class_ids)

#tf.keras.backend.clear_session()
#TODO: Added to run only on CPU when needed
#tf.config.set_visible_devices([], 'GPU')
#%%

def create_model(input_dimension,output_dimension,lr=0.001,drop_rate=0.5):
    k_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
    input_tensor_feats=keras.Input(shape=input_dimension, name="feat_in")
    #add a dummy dimension for using LRN
    x=tf.expand_dims(input_tensor_feats,axis=2)
    # Blocks of Conv->ReLu->LRN->max-pool
    f_list=[16,32,64,128,256,256]
    k_list=[7,5,5,7,7,8]
    ks_list=6*[1]
    use_LRN=3*[False,True]
    use_MP=3*[True]+3*[False]
    
    p_list=2*[3]+[5]+3*[None]
    s_list=2*[2]+[5]+3*[None]
    #act_list=len(f_list)*['relu']
    for i in range(len(f_list)):
        x=layers.Conv2D(f_list[i],(k_list[i],1),strides=(ks_list[i], 1),
                        activation='relu',name=f'conv2d_{i}',padding="same",
                        kernel_initializer=k_initializer)(x)
        if use_LRN[i]:
            # x=tf.nn.local_response_normalization(x, depth_radius=5, bias=1,
            #                             alpha=2e-4, beta=0.75, name=f'LRN_{i}')
            x=layers.BatchNormalization(name=f'BN_{i}')(x)
        if use_MP[i]:
            x=layers.MaxPooling2D(pool_size=(p_list[i],1),strides=(s_list[i],1),
                                  padding="same",name=f'maxpool_{i}')(x)

    x=layers.Dropout(rate=drop_rate, name = 'drop_1')(x)
    #remove the dummy dimension
    print(x.get_shape().as_list())
    #feats=x[:,:,0,:]
    feats=layers.Flatten(name='flat_layer')(x)
    #Create a separate feature extraction model
    get_feats=keras.Model(input_tensor_feats, feats, name='get_feats')
    
    #use it for the end2end model
    input_tensor=keras.Input(shape=input_dimension, name="model_in")
    feats=get_feats(input_tensor)
    x = keras.layers.Dense(output_dimension[0],name='FC')(feats)
    x = keras.layers.Activation('softmax',name='softmax')(x)

    model = keras.Model(input_tensor, x, name='model')
    
    # learning_rate=keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate,decay_steps=lr_decay_steps, 
    # decay_rate=lr_decay_rate, staircase=True)
    op = keras.optimizers.SGD(learning_rate=lr,momentum=0.6,nesterov=True)
    
    #op = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=op, 
                  metrics=['accuracy'])
    return model,get_feats


def load_data_helper(Dsplit_mask_dict,musig_dict,data_path,win_len_s,step_s,
                     bsize,bstep,ppg_win_len,Fs_new,augmentor,
                     seq_format_function_P2St,dsampling_factor_aug=1):       

    input_list,output_list,Dsplit_mask_dict=load_data.get_train_data(data_path,
                                    win_len_s=win_len_s,step_s=step_s,
                                    bsize=bsize,bstep=bstep,
                                    Dsplit_mask_dict=Dsplit_mask_dict,
                                    musig_dict=musig_dict,mode='P2St')

    Dsplit_mask_list=[Dsplit_mask_dict['P2St'][c]
                      for c in Dsplit_mask_dict['Dspecs']['key_order']]
    
    train_in_list,train_out_list=[],[]
    val_in_list,val_out_list=[],[]
    for i in range(len(input_list)):
        #data split masks
        sel_mask_train,sel_mask_val,_=Dsplit_mask_list[i]
        
        #(N,2560,5)-->(N,5)
        # eps=1e-8
        # label=np.round(np.mean(output_list[i][:,:,:],axis=1)+eps).astype(int)
        # check_label_mask=np.invert((np.sum(label,axis=-1)==1))
        # if np.mean(check_label_mask.astype(int))!=0:
        #     print('\n Resolving ties in label assignment...\n ')
        #     label[check_label_mask,0]=0 #zero out the useless class
        #     check_label_mask=np.invert((np.sum(label,axis=-1)==1)) #recalculate
        
        # assert np.mean(check_label_mask.astype(int))==0, 'ties not resolved as expected'
        
        # sel_mask_val=((sel_mask_val) & (np.invert(label[:,0].astype(bool))))
        # sel_mask_train=((sel_mask_train) & (np.invert(label[:,0].astype(bool))))
        # label=label[:,1:] #remove first row
        label=output_list[i][:,0,5:20]
        
        print(f'Selected {np.sum(sel_mask_train)/len(sel_mask_train)} ' 
              f' ratio of samples from class {i} for training')
        
        val_in_list.append(input_list[i][sel_mask_val])
        val_out_list.append(label[sel_mask_val])
        train_in_list.append(input_list[i][sel_mask_train])
        train_out_list.append(label[sel_mask_train])
            
    
    # val_data=[np.expand_dims(np.concatenate(val_in_list,axis=0),axis=1),
    #           np.concatenate(val_out_list,axis=0)]
    # train_data=[np.expand_dims(np.concatenate(train_in_list,axis=0),axis=1),
    #           np.concatenate(train_out_list,axis=0)]
    val_data=[np.concatenate(val_in_list,axis=0),
              np.concatenate(val_out_list,axis=0)]
    train_data=[np.concatenate(train_in_list,axis=0),
                np.concatenate(train_out_list,axis=0)]
    print(val_data[0].shape,val_data[1].shape,
          train_data[0].shape,train_data[1].shape)
    
    HR_val=np.mean(val_data[0][:,:,1],axis=1)
    val_data[0]=val_data[0][:,:,0]
    HR_train=np.mean(train_data[0][:,:,1],axis=1)
    train_data[0]=train_data[0][:,:,0]

    #  Get synthetic Data
    in_data_synth,out_data_synth=augmentor.main(seq_format_function_P2St,
                        save_name=f'{save_name}',
                        show_plots=False,suffix=aug_suffix)
    #TODO: Select stress (0:5) vs. class(5:20) here
    in_data_synth=in_data_synth[::dsampling_factor_aug]
    out_data_synth_clip=out_data_synth[::dsampling_factor_aug,5:20]


    # out_data_synth_clip=np.zeros([len(out_data_synth),200])
    # for i in range(len(out_data_synth)):
    #     seq=out_data_synth[i,:,1] #[400,]
    #     avg_wins=load_data.sliding_window_fragmentation([seq],
    #         win_size=201,step_size=1,axes=None) #[200,201,]
    #     out_data_synth_clip[i]=np.mean(avg_wins,axis=1) #[200,]
    
    train_data_synth=[in_data_synth,out_data_synth_clip]
    
    # real_data_factr to balance real and synthetic data in aug data
    real_data_factr=int(len(out_data_synth_clip)/len(train_data[1]))
    print(real_data_factr)
    in_data_aug=np.concatenate([in_data_synth]+real_data_factr*[train_data[0]],axis=0)
    out_data_aug=np.concatenate([out_data_synth_clip]+real_data_factr*[train_data[1]],axis=0)
    
    #shuffle
    rand_idx=np.random.permutation(len(in_data_aug))
    in_data_aug,out_data_aug=in_data_aug[rand_idx],out_data_aug[rand_idx]
    train_data_aug=[in_data_aug,out_data_aug]
    rand_idx=np.random.permutation(len(in_data_synth))
    train_data_synth=[train_data_synth[0][rand_idx],train_data_synth[1][rand_idx]]
    
    print(train_data_synth[0].shape,train_data_synth[1].shape)
    print(train_data_aug[0].shape,train_data_aug[1].shape)
    
    misc_list=[Dsplit_mask_dict,HR_train]
    return val_data,train_data,train_data_synth,train_data_aug,misc_list

def subbed_training_helper(train_data_sub,val_ds,model,epochs,batch_size,
                           ckpt_filepath,callbacks,n_subsets=12,
                           show_plots=True):
    part_idxs=np.linspace(0,len(train_data_sub[1]),n_subsets+1).astype(int)
    train_ds_aug_list=[make_data_pipe([train_data_sub[0][start_idx:end_idx],
                                       train_data_sub[1][start_idx:end_idx]]
                        ,batch_size=batch_size,shuffle=True) for 
                       start_idx,end_idx in zip(part_idxs[:-1],part_idxs[1:])]
    losses,val_losses,min_val_loss=[],[],1e5
    acc,val_acc,max_val_acc=[],[],0

    for epoch in range(epochs):
        for n in range(n_subsets):
            train_ds_aug=train_ds_aug_list[n]
            history=model.fit(train_ds_aug,epochs=1,
                              validation_data=val_ds,callbacks=callbacks)
            #print(len(history.history['loss']))
            losses.append(history.history['loss'][0])
            val_losses.append(history.history['val_loss'][0])
            acc.append(history.history['accuracy'][0])
            val_acc.append(history.history['val_accuracy'][0])
            
            if val_acc[-1]>=max_val_acc:
                model.save_weights(ckpt_filepath.format(epoch=epoch,n=n))
                #min_val_loss=val_losses[-1]*1
                max_val_acc=val_acc[-1]
                print(f'\n Saved model with val_acc={max_val_acc} \n')
                
        print(f'\n===========Done main-epoch {epoch}==========\n')
    
    print(np.max(val_acc),int(np.argmax(val_acc)/n_subsets),
          np.argmax(val_acc)%n_subsets)
    
    if show_plots:
        #Plot loss curves
        plt.figure();ax=plt.subplot(211)
        plt.plot(losses)
        plt.plot(val_losses)
        plt.plot([np.argmin(val_losses)],[np.min(val_losses)],'ro')
        plt.legend(['Train', 'Val','min. Val'])
        plt.grid(True);plt.title('Loss')
        
        #Plot acc curves
        plt.subplot(212,sharex=ax);plt.plot(acc)
        plt.plot(val_acc)
        plt.plot([np.argmax(val_acc)],[np.max(val_acc)],'ro')
        plt.legend(['Train', 'Val','max. Val'])
        plt.grid(True);plt.title('Accuracy')
    return losses,val_losses,acc,val_acc


def create_and_train_model(log_prefix,common_batch_size,exp_sub_id,n_subsets,
                           epochs,dataset,val_data):
    #logging specs
    ckpt_dir=log_prefix+f'/{exp_sub_id}/checkpoints'
    tflogs_dir=(log_prefix+f'/{exp_sub_id}/tflogs').replace('/','\\')
    stdout_log_file = log_prefix + f'/{exp_sub_id}/stdout.log'
    training_log_file = log_prefix + f'/{exp_sub_id}/training.log'
    ckpt_filepath=ckpt_dir+'/ckpt-{epoch:03d}'#'-{n:02d}'#'/cp.ckpt'
    os.makedirs(ckpt_dir,exist_ok=True)
    os.makedirs(tflogs_dir,exist_ok=True)
    file_path=log_prefix+f'/{exp_sub_id}'
    plot_model = True
    logging = True
    
    model,get_feats = create_model(input_dimension=(morph_win_len*Fs_new,1),
                                   output_dimension=(load_data.n_classes,),
                                   lr=0.001,drop_rate=0.5)
    
    # # if loading pre-existing weights
    # weights_dir_ssl=data_path+f'../{save_name}/SSL_model'
    # latest_ckpt = tf.train.latest_checkpoint(weights_dir_ssl)
    # print('Loading model from ckpt {}'.format(latest_ckpt))
    # model_ssl.load_weights(latest_ckpt)
    
    #Verify model graph
    print(model.summary())
    #model_ssl.trainable=False #freeze model_HR weights
    #print(model_ssl.summary())
    #model_ssl.trainable=True #freeze model_HR weights

    # Include the epoch in the file name (uses `str.format`)
    if plot_model:
        keras.utils.plot_model(model,to_file=file_path+'/model.png', 
        dpi=200, show_shapes=True, show_layer_names=True, expand_nested=True)
        

    #callbacks
    callbacks=[]
    
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_filepath,
    #                 save_weights_only=True,save_best_only=True,
    #                 monitor="val_accuracy",mode='max'))
    if logging:
        callbacks.append(tf.keras.callbacks.CSVLogger(training_log_file,
                                                separator=',',append=True))
        #callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tflogs_dir))
    
    # Train
    
    #Form data pipes
    val_ds=make_data_pipe(val_data,batch_size=common_batch_size,shuffle=False)
    # train_ds=make_data_pipe(train_data,batch_size=common_batch_size,
    #                             shuffle=True)
    
    #When very large dataset size
    #train_data_aug=train_data_synth #TODO: if all synthetic to be used
    losses,val_losses,acc,val_acc=subbed_training_helper(dataset,val_ds,model,
                            epochs,common_batch_size,ckpt_filepath,callbacks,
                            n_subsets=n_subsets,show_plots=True)
    # history=model.fit(train_ds,epochs=epochs,
    #                       validation_data=val_ds,callbacks=callbacks)
    # losses=history.history['loss']
    # val_losses=history.history['val_loss']
    # print(min(val_losses))
    # #Plot loss curves
    # plt.figure();ax=plt.subplot(211)
    # plt.plot(losses)
    # plt.plot(val_losses)
    # plt.plot([np.argmin(val_losses)],[np.min(val_losses)],'ro')
    # plt.legend(['Train', 'Val','min. Val'])
    # plt.grid(True);plt.title('Loss')
    
    # losses=history.history['accuracy']
    # val_losses=history.history['val_accuracy']
    # print(max(val_losses))
    # #Plot loss curves
    # plt.subplot(212,sharex=ax);plt.plot(losses)
    # plt.plot(val_losses)
    # plt.plot([np.argmax(val_losses)],[np.max(val_losses)],'ro')
    # plt.legend(['Train', 'Val','max. Val'])
    # plt.grid(True);plt.title('Accuracy')
    del model, get_feats
    return
#%%
def main():
    #Load Data
    ver=13
    plt.close('all')
    path_prefix='../data/pre-training'
    exp_id=f'{ver}_1'
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_prefix='../experiments/{}_{}'.format(exp_id,current_time)
    log_prefix='../experiments/{}'.format(exp_id)
    #path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #

    #class_name_list=['S7'
    #load_data.class_ids={class_name:all_class_ids[class_name]}
    # load_data.class_ids={k:all_class_ids[k] 
    #                       for k in list(all_class_ids.keys())[10:11]}
    # Load Dsplit_mask
    Dsplit_filename = (f'{proj_path}/../data/pre-training/old/'
                f'WESAD_musig_Dsplit_w{augmentor.win_len_s}s{augmentor.step_s}'
                f'b{load_data.test_bsize}.pickle')
    if os.path.isfile(Dsplit_filename):
        with open (Dsplit_filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                        'Run get_train_data in R2S mode first.')

    
    #%%
    #Load Data
    d_list=load_data_helper(Dsplit_mask_dict,musig_dict,data_path,win_len_s,
                            step_s,bsize,bstep,ppg_win_len,Fs_new,augmentor,
                            seq_format_function_P2St,dsampling_factor_aug=4)
    
    
    val_data,train_data,train_data_synth,train_data_aug,misc_list=d_list
    Dsplit_mask_dict,HR_train=misc_list
    for d in [val_data,train_data,train_data_synth,train_data_aug]:
        d[0]=np.concatenate([d[0][:,:morph_win_len*Fs_new],
                             d[0][:,-morph_win_len*Fs_new:]],axis=0)
        d[0]=np.expand_dims(d[0],axis=-1)
        d[1]=np.concatenate(2*[d[1]],axis=0)
        
    #Visualize
    idx=19000
    sample_data=train_data_aug#_synth#val_data
    x_vec=np.arange(sample_data[0].shape[1])
    #plt.figure();ax=plt.subplot(211);plt.plot(x_vec,val_data[0][idx,0,:,:])
    plt.figure();ax=plt.subplot(211);plt.plot(x_vec,sample_data[0][idx,:])
    plt.subplot(212,sharex=ax)
    plt.plot(x_vec,sample_data[0][idx+5,:])
    
    #======================================
    


    exp_configs=[['ppg2id_synth_full_interbatchsave',8,100,train_data_synth],
                 ['ppg2id_aug_full_interbatchsave',8,25,train_data_aug],
                 ['ppg2id',1,600,train_data]]
    
    
    
    for k in range(len(exp_configs)):
        exp_sub_id,n_subsets,epochs,dataset=exp_configs[k]
        create_and_train_model(log_prefix,common_batch_size,exp_sub_id,
                               n_subsets,epochs,dataset,val_data)
    
    #%% Create inference model
    model,get_feats = create_model(input_dimension=(morph_win_len*Fs_new,1),
                               output_dimension=(load_data.n_classes,),
                               lr=0.001,drop_rate=0.5)
    #TODO: Change according to need
    #weights_path_prefix='../data/post-training/model_weights' #for saved weights
    plt.close('all')
    load_data.class_ids=copy.deepcopy(all_class_ids)
    show_plots=False
    #expid_list=['ppg2stress','ppg2stress_aug']#['2_17','2_16']
    #expid_list=['ppg2subject','ppg2subject_aug']#['2_17','2_16']
    expid_list=['ppg2id','ppg2id_synth_full_interbatchsave','ppg2id_aug_full_interbatchsave']
    #confusion_keys=['baseline', 'stress', 'amusement','meditation']
    confusion_keys=list(load_data.class_ids.keys())
    save_data=False

    label_list=['Original','Synthetic','Augmented']
    total_err_list,handle_list=[],[]
    # Using max size to avoid defragmentation step
    test_bsize,test_bstep=1,1
    
    # Get test Data
    ppg_test_list=[]
    y_test_list=[]
    #class_name='S15'
    for class_name in list(load_data.class_ids.keys())[:]:
        ppg_list,y_list,Dsplit_mask_dict=load_data.get_test_data(
                                        data_path+class_name,
                                        win_len_s=win_len_s,step_s=step_s,
                                        bsize=test_bsize,bstep=test_bstep,
                                        Dsplit_mask_dict=Dsplit_mask_dict,
                                        musig_dict=musig_dict,mode='P2St')
        ppg_test_list.append(np.concatenate(ppg_list,axis=0))
        #ppg_test_list.append(np.concatenate([ppg[:,:-1] for ppg in ppg_list],axis=0))
        y_test_list.append(np.concatenate(y_list,axis=0))
    
    ppg_test=np.concatenate(ppg_test_list,axis=0)
    y_test=np.concatenate(y_test_list,axis=0)
    
    # eps=1e-8
    # label=np.round(np.mean(y_test,axis=1)+eps).astype(int)
    # check_label_mask=np.invert((np.sum(label,axis=-1)==1))
    # if np.mean(check_label_mask.astype(int))!=0:
    #     print('\n Resolving ties in label assignment...\n ')
    #     label[check_label_mask,0]=0 #zero out the useless class
    #     check_label_mask=np.invert((np.sum(label,axis=-1)==1)) #recalculate
    # assert np.mean(check_label_mask.astype(int))==0, 'ties not resolved as expected'
    
    # sel_mask_label=np.invert(label[:,0].astype(bool))
    # label=label[:,1:] #remove first row
    # ppg_test,y_test=ppg_test[sel_mask_label],label[sel_mask_label]
    
    # TODO: currently based on morph_window=4. Change this as needed.
    plt.figure();plt.plot(ppg_test[10,:,0])
    ppg_test=np.concatenate([ppg_test[:,:4*Fs_new,:],ppg_test[:,4*Fs_new:,:]],
                            axis=0)
    y_test=np.concatenate(2*[y_test[:,0,5:20]],axis=0)
    

    ppg_test,HR_test=ppg_test[:,:,0:1],np.mean(ppg_test[:,:,1],axis=1)
    
    data_test=[ppg_test,y_test,HR_test]
    title='Test Error for Identity Recognition using PPG'
    fig=make_x2xeval_plots(data_test,model,log_prefix,expid_list,label_list,
                       confusion_keys,title,save_fig=True)
    return
    
if __name__=='__main__':
    main()
