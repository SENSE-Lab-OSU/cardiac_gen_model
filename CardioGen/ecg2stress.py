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
#gpus = tf.config.experimental.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(gpus[0], True)


from tensorflow.keras.activations import relu

import datetime

from lib.data import load_data_wesad_eval as load_data
from lib.data.load_data_wesad import Rpeak2HR, sliding_window_fragmentation
from lib.custom_layers import upsample, downsample
from lib.utils import make_data_pipe 
from lib.utils import start_logging, stop_logging
from lib.utils import make_x2xeval_plots
#from pathlib import Path
#curr_dir=Path.cwd()
#root_dir=curr_dir.parents[1]

import Augmentor_ecg_st_id as augmentor
data_path='../data/pre-training/WESAD/'

augmentor.data_path=data_path
save_name='WESAD_synth_h30_m28/E2StId'
aug_suffix='s_c'
proj_path='.'
ver=12
#win_len_s,step_s,bsize=augmentor.win_len_s,augmentor.step_s,augmentor.bsize

tf.keras.backend.set_floatx('float32')
lrelu_pt2=lambda x: relu(x, alpha=0.2)

#tf.keras.backend.clear_session()
#TODO: Added to run only on CPU when needed
#tf.config.set_visible_devices([], 'GPU')

def conv_block(input_tensor,  filter_size, kernel_size, stride, batch_norm, 
               dropout, dropout_rate, name):
  
    conv  = layers.Conv1D(filters = filter_size, 
                          kernel_size = kernel_size, strides = stride, 
                          padding='same', name=name)(input_tensor)
    if batch_norm:
        conv = layers.Batch_Normalization(name = name)(conv)
    conv = tf.nn.leaky_relu(conv,alpha=0.2,name = name)
    if dropout:
        conv = layers.Dropout(rate=dropout_rate, name = name)(conv)
    return conv

def create_model_ssl(input_dimension,drop_rate,hidden_nodes=128,stride_mp=4):
    
    input_tensor=keras.Input(shape=input_dimension, name="ssl_in")
    main_branch = conv_block(input_tensor,  filter_size = 32, kernel_size = 32,
                             stride = 1, batch_norm = False, dropout = False, 
                             dropout_rate = drop_rate * 0.5, 
                             name = 'conv_layer_1')
    main_branch = conv_block(main_branch,  filter_size = 32, kernel_size = 32, 
                             stride = 1, batch_norm = False, dropout = False, 
                             dropout_rate = drop_rate * 0.5,
                             name = 'conv_layer_2')
    
    ## conv block 1
    conv1     = main_branch 
    conv1     =layers.MaxPool1D(pool_size=conv1.get_shape()[1],
                                    strides=stride_mp, padding='valid', 
                                    name = 'GAP1')(conv1)
    conv1     = layers.Flatten(name = 'flat_layer1')(conv1)
    
    main_branch = layers.MaxPool1D(pool_size = 8, strides=2, 
                                   padding='valid', name = 'mp1')(main_branch)
    main_branch = conv_block(main_branch,filter_size = 64, kernel_size = 16, 
                             stride = 1, batch_norm = False, dropout = False, 
                             dropout_rate = drop_rate * 0.5, 
                             name = 'conv_layer_3')
    main_branch = conv_block(main_branch,filter_size = 64, kernel_size = 16,
                             stride = 1, batch_norm = False, dropout = False,
                             dropout_rate = drop_rate * 0.5, 
                             name = 'conv_layer_4')
    
    ## conv block 2
    conv2     = main_branch 
    conv2     =layers.MaxPool1D(pool_size=conv2.get_shape()[1],
                                    strides=stride_mp, padding='valid', 
                                    name = 'GAP2')(conv2)
    conv2     = layers.Flatten(name = 'flat_layer2')(conv2)
    
    main_branch = layers.MaxPool1D(pool_size = 8, strides=2, 
                                   padding='valid', name = 'mp2')(main_branch)      
    main_branch = conv_block(main_branch,filter_size = 128, kernel_size = 8, 
                             stride = 1, batch_norm = False, dropout = False, 
                             dropout_rate = drop_rate * 0.5, 
                             name = 'conv_layer_5')
    main_branch = conv_block(main_branch,filter_size = 128, kernel_size = 8, 
                             stride = 1, batch_norm = False, dropout = False, 
                             dropout_rate = drop_rate * 0.5, 
                             name = 'conv_layer_6')
    
    ## conv block 3
    conv3     = main_branch 
    conv3     =layers.MaxPool1D(pool_size=conv3.get_shape()[1],
                                    strides=stride_mp, padding='valid', 
                                    name = 'GAP3')(conv3)
    conv3     = layers.Flatten(name = 'flat_layer3')(conv3)
    
    gap_pool_size   = main_branch.get_shape()[1]
    main_branch     = layers.MaxPool1D(pool_size = gap_pool_size, 
                                          strides=1, padding='valid', 
                                          name = 'GAP')(main_branch)
    ## final conv block output
    main_branch     = layers.Flatten(name = 'flat_layer')(main_branch)
    model = keras.Model(input_tensor, main_branch, name='model_ssl')
    
    #compile and prep fo training
    initial_learning_rate = 0.001
    lr_decay_steps = 10000
    lr_decay_rate = 0.9
    learning_rate=keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,decay_steps=lr_decay_steps, 
        decay_rate=lr_decay_rate, staircase=True)
    op = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = op, loss = "mse", metrics = ["mse"])
    return model

def create_model_sl(input_dimension,output_dimension,model_ssl,lr_super=0.001,
                    hidden_nodes=512,dropout=0.2,L2=0):
    input_tensor=keras.Input(shape=input_dimension, name="al_in")
    feat=model_ssl(input_tensor)
    x = keras.layers.Dense(hidden_nodes,activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(L2))(feat)
    
    x = keras.layers.Dense(hidden_nodes, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(L2))(x)
    x = keras.layers.Dense(output_dimension[0])(x)
    x = keras.layers.Activation('softmax')(x)
    model = keras.Model(input_tensor, x, name='model_sl')
    op = keras.optimizers.Adam(lr=lr_super)
    #op = keras.optimizers.SGD(learning_rate=0.001,momentum=0.6,nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=op, 
                  metrics=['accuracy'])
    return model

def load_data_helper(Dsplit_mask_dict,musig_dict,data_path,win_len_s,step_s,
                     bsize,bstep,ecg_win_len,Fs_new,augmentor,
                     seq_format_function_E2St,dsampling_factor_aug=1):        

    input_list,output_list,Dsplit_mask_dict=load_data.get_train_data(data_path,
                                    win_len_s=win_len_s,step_s=step_s,
                                    bsize=bsize,bstep=bstep,
                                    Dsplit_mask_dict=Dsplit_mask_dict,
                                    musig_dict=musig_dict,mode='E2St')

    Dsplit_mask_list=[Dsplit_mask_dict['E2St'][c]
                      for c in Dsplit_mask_dict['Dspecs']['key_order']]
    
    train_in_list,train_out_list=[],[]
    val_in_list,val_out_list=[],[]
    for i in range(len(input_list)):
        #data split masks
        sel_mask_train,sel_mask_val,_=Dsplit_mask_list[i]
        
        #(N,2560,5)-->(N,5)

        eps=1e-8
        label=np.round(np.mean(output_list[i][:,:,0:5],axis=1)+eps).astype(int)
        check_label_mask=np.invert((np.sum(label,axis=-1)==1))
        if np.mean(check_label_mask.astype(int))!=0:
            print('\n Resolving ties in label assignment...\n ')
            label[check_label_mask,0]=0 #zero out the useless class
            check_label_mask=np.invert((np.sum(label,axis=-1)==1)) #recalculate
        
        assert np.mean(check_label_mask.astype(int))==0, 'ties not resolved as expected'
        
        sel_mask_val=((sel_mask_val) & (np.invert(label[:,0].astype(bool))))
        sel_mask_train=((sel_mask_train) & (np.invert(label[:,0].astype(bool))))
        label=label[:,1:] #remove first row
        #label=output_list[i][:,0,:]
        
        print(f'Selected {np.sum(sel_mask_train)/len(sel_mask_train)} ' 
              f' ratio of samples from class {i} for training')
        
        val_in_list.append(input_list[i][sel_mask_val])
        val_out_list.append(label[sel_mask_val])
        train_in_list.append(input_list[i][sel_mask_train])
        train_out_list.append(label[sel_mask_train])
            
    val_data=[np.concatenate(val_in_list,axis=0),
              np.concatenate(val_out_list,axis=0).astype(np.float32)]
    train_data=[np.concatenate(train_in_list,axis=0),
                np.concatenate(train_out_list,axis=0).astype(np.float32)]
    print(val_data[0].shape,val_data[1].shape,
          train_data[0].shape,train_data[1].shape)
    
    HR_val=np.mean(val_data[0][:,:,1],axis=1)
    val_data[0]=val_data[0][:,:,0]
    HR_train=np.mean(train_data[0][:,:,1],axis=1)
    train_data[0]=train_data[0][:,:,0]

    #  Get synthetic Data
    in_data_synth,out_data_synth=augmentor.main(seq_format_function_E2St,
                        save_name=f'{save_name}',
                        show_plots=False,suffix=aug_suffix)
    
    #TODO: Select stress (0:5) vs. class(5:20) here
    in_data_synth=in_data_synth[::dsampling_factor_aug]
    out_data_synth_clip=out_data_synth[::dsampling_factor_aug,1:5]
    
    
    train_data_synth=[in_data_synth,out_data_synth_clip]
    
    # real_data_factr to balance real and synthetic data in aug data
    real_data_factr=int(len(out_data_synth_clip)/len(train_data[1]))#1
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

def create_model(input_dim_ssl=(2560,1),output_dim_sl=(4,),
                 plot_model=False,file_path=None):
    # Make models
    
    #output_dim_sl=(load_data.n_classes,)
    #input_dim_sl=(128,)
    model_ssl = create_model_ssl(input_dim_ssl,drop_rate=0.6,
                                  hidden_nodes=128,stride_mp=4)
    model_sl = create_model_sl(input_dim_ssl,output_dim_sl,model_ssl,
                                lr_super=0.001,hidden_nodes=512,dropout=0.2,
                                L2=0)
    
    # # if loading pre-existing weights
    # weights_dir_ssl=data_path+f'../{save_name}/SSL_model'
    # latest_ckpt = tf.train.latest_checkpoint(weights_dir_ssl)
    # print('Loading model from ckpt {}'.format(latest_ckpt))
    # model_ssl.load_weights(latest_ckpt)
    
    #Verify model graph
    print(model_ssl.summary())
    print(model_sl.summary())
    #model_ssl.trainable=False #freeze model_HR weights
    #print(model_ssl.summary())
    #model_ssl.trainable=True #freeze model_HR weights

    # Include the epoch in the file name (uses `str.format`)

    if plot_model:
        keras.utils.plot_model(model_ssl,to_file=file_path+'/SSL.png', 
        dpi=200, show_shapes=True, show_layer_names=True, expand_nested=True)
        keras.utils.plot_model(model_sl,to_file=file_path+'/SL.png', 
        dpi=200, show_shapes=True, show_layer_names=True, expand_nested=True)
        
    return model_ssl,model_sl
    

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
        
    model_ssl,model_sl=create_model(input_dim_ssl=(2560,1),output_dim_sl=(4,),
                                    plot_model=plot_model,file_path=file_path)
    
    #callbacks
    #epochs=25#120

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
    
    # When very large dataset size
    #train_data_aug=train_data_synth #TODO: if all synthetic to be used
    losses,val_losses,acc,val_acc=subbed_training_helper(dataset,val_ds,model_sl,
                            epochs,common_batch_size,ckpt_filepath,callbacks,
                            n_subsets=n_subsets,show_plots=True)
    
    # history=model_sl.fit(train_ds,epochs=epochs,initial_epoch=0,
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
    
    del model_ssl,model_sl
    return
#%%
def main():
    #path specs
    ver=13
    plt.close('all')
    path_prefix='../data/pre-training'
    exp_id=f'{ver}_1'

    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_prefix='../experiments/{}_{}'.format(exp_id,current_time)
    log_prefix='../experiments/{}'.format(exp_id)
    #path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #
    
    #Hyperparameter specs
    win_len_s,step_s=8,2 #in sec
    bsize,bstep=2,1
    Fs_new=load_data.Fs_ecg_E2St
    sample_win_len=((bsize-1)*step_s+win_len_s)*Fs_new
    sample_step_size=bstep*step_s*Fs_new
    ecg_win_len=sample_win_len#*AE_reps
    
    #TODO: Started using this instead of redefining the function
    seq_format_function_E2St=augmentor.seq_format_function_E2St
    common_batch_size=128 #128 #32
    
    all_class_ids=copy.deepcopy(load_data.class_ids)
    #class_name_list=['S7'
    #load_data.class_ids={class_name:all_class_ids[class_name]}
    # load_data.class_ids={k:all_class_ids[k] 
    #                       for k in list(all_class_ids.keys())[10:11]}
    # Load Dsplit_mask
    Dsplit_filename = (f'{proj_path}/../data/pre-training/'
                f'WESAD_musig_Dsplit_w{augmentor.win_len_s}s{augmentor.step_s}'
                f'b{augmentor.bsize}.pickle')
    if os.path.isfile(Dsplit_filename):
        with open (Dsplit_filename, 'rb') as fp:
            musig_dict,Dsplit_mask_dict = pickle.load(fp)
    else:
        assert False, ('Could not find existing Dsplit_mask_dict. '
                        'Run get_train_data in R2S mode first.')
    
    #%%
    #Load Data
    #======Don't Run this block during inference=========
    d_list=load_data_helper(Dsplit_mask_dict,musig_dict,data_path,win_len_s,
                            step_s,bsize,bstep,ecg_win_len,Fs_new,augmentor,
                            seq_format_function_E2St,dsampling_factor_aug=4)
    
    
    val_data,train_data,train_data_synth,train_data_aug,misc_list=d_list
    Dsplit_mask_dict,HR_train=misc_list
    for d in [val_data,train_data,train_data_synth,train_data_aug]:
        d[0]=np.expand_dims(d[0],axis=-1)
        
    #Visualize
    idx=190
    sample_data=train_data_synth#val_data
    x_vec=np.arange(sample_data[0].shape[1])
    #plt.figure();ax=plt.subplot(211);plt.plot(x_vec,val_data[0][idx,0,:,:])
    plt.figure();ax=plt.subplot(211);plt.plot(x_vec,sample_data[0][idx,:])
    plt.subplot(212,sharex=ax)
    plt.plot(x_vec,sample_data[0][idx+5,:])
    
    #======================================
    
    
    exp_configs=[['ecg2stress_synth_full_interbatchsave',8,100,train_data_synth],
                 ['ecg2stress_aug_full_interbatchsave',8,25,train_data_aug],
                 ['ecg2stress',1,600,train_data]]
    
    
    
    for k in range(len(exp_configs)):
        exp_sub_id,n_subsets,epochs,dataset=exp_configs[k]
        create_and_train_model(log_prefix,common_batch_size,exp_sub_id,
                               n_subsets,epochs,dataset,val_data)

    
    #%% Create inference model
    model_ssl,model_sl=create_model(input_dim_ssl=(2560,1),output_dim_sl=(4,))
    #TODO: Change according to need
    #weights_path_prefix='../data/post-training/model_weights' #for saved weights
    plt.close('all')
    load_data.class_ids=copy.deepcopy(all_class_ids)
    show_plots=False
    expid_list=['ecg2stress','ecg2stress_synth_full_interbatchsave',
                'ecg2stress_aug_full_interbatchsave']
    #expid_list=['ecg2stress_aug_full_interbatchsave']
    label_list=['Original','Synthetic','Augmented']
    #label_list=['Augmented']
    
    confusion_keys=['baseline', 'stress', 'amusement','meditation']
    #confusion_keys=list(load_data.class_ids.keys())
    save_data=False



    # Using max size to avoid defragmentation step
    test_bsize,test_bstep=bsize,bstep
    
    # Get test Data
    ecg_test_list=[]
    y_test_list=[]
    #class_name='S15'
    for class_name in list(load_data.class_ids.keys())[:]:
        ecg_list,y_list,Dsplit_mask_dict=load_data.get_test_data(
                                        data_path+class_name,
                                        win_len_s=win_len_s,step_s=step_s,
                                        bsize=test_bsize,bstep=test_bstep,
                                        Dsplit_mask_dict=Dsplit_mask_dict,
                                        musig_dict=musig_dict,mode='E2St')
        ecg_test_list.append(np.concatenate(ecg_list,axis=0))
        #ecg_test_list.append(np.concatenate([ecg[:,:-1] for ecg in ecg_list],axis=0))
        y_test_list.append(np.concatenate(y_list,axis=0))
    
    ecg_test=np.concatenate(ecg_test_list,axis=0)
    y_test=np.concatenate(y_test_list,axis=0)
    
    eps=1e-8
    label=np.round(np.mean(y_test[:,:,0:5],axis=1)+eps).astype(int)
    check_label_mask=np.invert((np.sum(label,axis=-1)==1))
    if np.mean(check_label_mask.astype(int))!=0:
        print('\n Resolving ties in label assignment...\n ')
        label[check_label_mask,0]=0 #zero out the useless class
        check_label_mask=np.invert((np.sum(label,axis=-1)==1)) #recalculate
    assert np.mean(check_label_mask.astype(int))==0, 'ties not resolved as expected'
    
    sel_mask_label=np.invert(label[:,0].astype(bool))
    label=label[:,1:] #remove first row
    ecg_test,y_test=ecg_test[sel_mask_label],label[sel_mask_label]
    #y_test=y_test[:,0,:]

    ecg_test,HR_test=ecg_test[:,:,0:1],np.mean(ecg_test[:,:,1],axis=1)
    
    data_test=[ecg_test,y_test,HR_test]
    title='Test Error for Emotion Recognition using ECG'
    fig=make_x2xeval_plots(data_test,model_sl,log_prefix,expid_list,label_list,
                       confusion_keys,title,save_fig=True)
    return
    
    

if __name__=='__main__':
    main()
