# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:35:59 2020

@author: agarwal.270a
"""

#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras import layers

#from tensorflow.keras import initializers as initizers

from .utils import find_batch_size, make_data_pipe
#from .networks_GAN import get_GAN_layers_default as get_GAN_layers
from .networks_GAN import get_GAN_layers_conv as get_GAN_layers
from .networks_GAN import downsample, upsample
eps=1e-9 #smoothening wherever division by zero may occur in autograd

tf.keras.backend.set_floatx('float32')
tf_dtype=tf.float32

#%% Default GAN layers





#%%
class Generator(tf.keras.layers.Layer):
    def __init__(self,layer_list,optimizer,use_x=True,n_classes=0):
        super(Generator, self).__init__()
        self.layer_list=layer_list
        self.class_layer_list=n_classes*[upsample(filters=1, kernel_size=(1,3),strides=(1,2))]
        self.n_classes=None if n_classes<=0 else n_classes
        self.use_x=use_x
        
        if optimizer is not None:
            self.optimizer=optimizer
        else:
            self.optimizer=tf.keras.optimizers.Adam(1e-4)
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
    def loss(self, fake_logit):
        return -tf.reduce_mean(fake_logit)
        #return self.bc(tf.ones_like(fake_output), fake_output)

    def call(self,input_list,training=None):
        z,x=input_list
        if self.use_x:
            x=tf.concat([z,x],axis=-1)
        else:
            x=z
        for lay in self.layer_list:
            x=lay(x,training=training)
            #print(x.shape.as_list())
        return x
    
class Discriminator(tf.keras.layers.Layer):
    def __init__(self,layer_list,optimizer,use_x=True):
        super(Discriminator, self).__init__()
        self.layer_list=layer_list
        self.use_x=use_x
        if optimizer is not None:
            self.optimizer=optimizer
        else:
            self.optimizer=tf.keras.optimizers.Adam(1e-4)
            
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(self,real_logit, fake_logit):
        real_loss = tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
        total_loss = fake_loss - real_loss
        #real_loss = self.bc(tf.ones_like(real_logit), real_logit)
        #fake_loss = self.bc(tf.zeros_like(fake_logit), fake_logit)
        #total_loss = fake_loss + real_loss
        return total_loss

    
    def call(self,input_list,training=None, return_all_out=False):
        y,x=input_list
        y=tf.cast(y, tf.float32)
        if self.use_x:
            x=tf.concat([y,x],axis=-1)
        else:
            x=y
        
        out_list=[]
        #TODO: Inserted dummy dimension after using downsample layers
        #x = tf.expand_dims(x, axis=1) #insert a dummy dim to use conv2d
        for lay in self.layer_list:
            x=lay(x,training=training)
            out_list.append(x)
            #print(x.shape.as_list())
        #x = tf.squeeze(x, axis=1) #remove the dummy dim 
        
        if return_all_out:
            return x, out_list.pop()
        else:
            return x

#%%
# =============================================================================
# class Generator_pix2pix(Generator):
#     def call(self,input_list,training=None):
#         z,x=input_list
#         if self.use_x:
#             x=tf.concat([z,x],axis=-1)
#         else:
#             x=z
#         
#         #x=self.layer_list[-3](x,training=training)
#         x = tf.expand_dims(x, axis=1) #insert a dummy dim to use conv2d
#         
#         # Downsampling
#         skips=[]
#         for i in range(4):
#             x=self.layer_list[i](x,training=training)
#             skips.append(x)
#             
#         
#         #skips = reversed(skips[:-1])
#         skips.pop() #remove last element
#         skips.reverse()
#         
#         # Upsampling
#         for i in range(4-1):
#             x=self.layer_list[i+4](x,training=training)
#             x=tf.concat([x,skips[i]],axis=-1)
#             #if i!=0: x=tf.concat([x,skips[i]],axis=-1)
#             
#         x=self.layer_list[(4-1)+4](x,training=training) #last unet layer
#         
#         x = tf.squeeze(x, axis=1) #remove the dummy dim
#         
#         for i in range(-2,0):
#             x=self.layer_list[i](x,training=training)
#         return x
# =============================================================================
    
class Generator_pix2pix(Generator):
    def __init__(self,layer_list,optimizer,latent_size,z_up_factor=8,
                 use_x=True,n_classes=0,Unet_reps=4):
        super(Generator, self).__init__()
        self.layer_list=layer_list
        self.class_layer_list=[upsample(filters=1,kernel_size=(1,3),
                                strides=(1,2)) for _ in range(n_classes)]
        self.n_classes=None if n_classes<=0 else n_classes
        self.z_up_factor=z_up_factor
        self.z_up_layer_list=[upsample(filters=latent_size,kernel_size=(1,3),
                    strides=(1,2)) for _ in range(int(np.log2(z_up_factor)))]
        self.use_x=use_x
        self.Unet_reps=Unet_reps
        
        
        if optimizer is not None:
            self.optimizer=optimizer
        else:
            self.optimizer=tf.keras.optimizers.Adam(1e-4)
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    def Unet(self,input_list,training,rnn_state_out_no,
             update_rnn_state_out=True):
        x,rnn_state_in=input_list
        # Downsampling
        skips=[]
        for i in range(4):
            x=self.layer_list[i](x,training=training)
            skips.append(x)
        #skips = reversed(skips[:-1])
        skips.pop() #remove last element
        skips.reverse()
        # Upsampling
        for i in range(4-1):
            x=self.layer_list[i+4](x,training=training)
            x=tf.concat([x,skips[i]],axis=-1)
            #if i!=0: x=tf.concat([x,skips[i]],axis=-1)
            
        x=self.layer_list[(4-1)+4](x,training=training) #last unet layer
        
        x = tf.squeeze(x, axis=1) #remove the dummy dim
        
        x=self.layer_list[-2](x,training=training,initial_state=rnn_state_in)
        if update_rnn_state_out: self.rnn_state_out=x[:,rnn_state_out_no,:]
        #self.rnn_state_out=x[:,-1,:]
        rnn_state_out_internal=x[:,-1,:]
        x=self.layer_list[-1](x,training=training)
        return x,rnn_state_out_internal
        
    def call(self,input_list,training=None,rnn_state=None,rnn_state_out_no=-1):
        z,x=input_list
        
        #Transform z to create correlated samples
        if len(self.z_up_layer_list)!=0:
            z = tf.expand_dims(z, axis=1) #insert a dummy dim to use conv2d
            for lay in self.z_up_layer_list:
                z=lay(z,training=training)
            z = tf.squeeze(z, axis=1) #remove the dummy dim
        
        #Append condition
        if self.use_x:
            x=tf.concat([z,x],axis=-1)
        else:
            x=z
        
        #x=self.layer_list[-3](x,training=training)
        x = tf.expand_dims(x, axis=1) #insert a dummy dim to use conv2d
        in_shape_Unet=x.get_shape().as_list() #[N,1,T,C]
        if self.n_classes is not None:
            assert in_shape_Unet[-1]>=self.n_classes,('input condition shape'
                                                'must be >= no. of classes')
        in_shape_Unet[0]=self.Unet_reps*1
        in_shape_Unet[2]=int(in_shape_Unet[2]/self.Unet_reps)
        if rnn_state_out_no!=-1:
            assert rnn_state_out_no>=0, 'rnn_state_out_no must be non-negative or -1'
            update_rnn_idx=int(rnn_state_out_no/in_shape_Unet[2])
            rnn_state_out_no%=in_shape_Unet[2]
        else:
            update_rnn_idx=self.Unet_reps-1
            
        #in_shape_Unet = (in_shape[1int(in_shape[2]/self.Unet_reps),in_shape[3])

        inputs = layers.Reshape(in_shape_Unet)(x) #Check if reshaping in desired fashion
        #tf.zeros([tf.shape(inputs)[0],8])
        out_list=[]
        for i in range(self.Unet_reps):
            Unet_out,rnn_state=self.Unet([inputs[:,i,:,:,:],rnn_state],
                        training=training,rnn_state_out_no=rnn_state_out_no,
                        update_rnn_state_out=(i==update_rnn_idx))
            out_list.append(Unet_out)
        #dec_mem = z[:,-mem_shape[0]:]
        out = tf.stack(out_list,axis=1)
        out = layers.Reshape((int(in_shape_Unet[2]*self.Unet_reps),-1))(out)#Check if reshaping in desired fashion
        return out

class Generator_pix2pix_mod(Generator_pix2pix):
    def Unet(self,input_list,training):
        x,rnn_state_in=input_list
        x,class_signal=x[:,:,:,:-self.n_classes],x[:,0:1,0:1,-self.n_classes:]
        # Downsampling
        skips=[]
        for i in range(4):
            x=self.layer_list[i](x,training=training)
            skips.append(x)
        #skips = reversed(skips[:-1])
        skips.pop() #remove last element
        skips.reverse()
        # Upsampling
        for i in range(4-1):
            x=self.layer_list[i+4](x,training=training)
            x=tf.concat([x,skips[i]],axis=-1)
            #if i!=0: x=tf.concat([x,skips[i]],axis=-1)
        
        #last unet layer
        #x=self.layer_list[(4-1)+4](x,training=training)
        for j in range(len(self.class_layer_list)):
            if j==0:
                y=(self.class_layer_list[j](x,training=training)*
                    class_signal[:,:,:,j:j+1])
            else:
                y+=(self.class_layer_list[j](x,training=training)*
                    class_signal[:,:,:,j:j+1])
        x = tf.squeeze(y, axis=1) #remove the dummy dim
        
        x=self.layer_list[-2](x,training=training,initial_state=rnn_state_in)
        self.rnn_state_out=x[:,-1,:]
        x=self.layer_list[-1](x,training=training)
        return x,self.rnn_state_out
#%%

class Model_CondWGAN(tf.keras.Model):
    def __init__(self,model_path,gen_layers=None,disc_layers=None,
                 save_flag=True,optimizers=None,aux_losses=None,
                 aux_losses_weights=None,in_shape=None,out_shape=None,
                 latent_size=10,mode='GAN',use_x=True, T_steps=5, n_classes=1,
                 feat_loss_weight=0,Unet_reps=2,z_up_factor=1):
        '''
            Setting all the variables for our model.
        '''
        super(Model_CondWGAN, self).__init__()
        self.model_path=model_path
        self.save_flag=save_flag
        self.mode=mode
        self.latent_size=latent_size
        self.use_x=use_x
        self.z_up_factor=z_up_factor

        
        self.n_dims_x=None
        self.n_dims_y=None
        #self.optimizer=self.net.optimizer
        #self.get_data=modify_get_data(get_data_old)
        
        
        if optimizers is None:
            optimizers=[tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5,
                        beta_2=0.9,epsilon=1e-7,amsgrad=False),
                        tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5,
                        beta_2=0.9,epsilon=1e-7,amsgrad=False)]
            
        if type(optimizers)!=type([]):
            raise AssertionError(('optimizers must be a list of 2 optimizers'
                                 ', one each for generator and discrimator'
                                 'respectively.'))

        self.optimizers=optimizers
        
        if gen_layers is None:
            gen_layers=get_GAN_layers(req_list=['gen_layers'])
            
        self.gen_layers=gen_layers
        self.gen=Generator_pix2pix(self.gen_layers,self.optimizers[0],
                                   latent_size,self.z_up_factor,use_x=use_x,
                                   Unet_reps=Unet_reps)#,
                                   #n_classes=n_classes)
        #self.gen=Generator(self.gen_layers,self.optimizers[0],use_x=use_x)


        if disc_layers is None:
            disc_layers=get_GAN_layers(req_list=['disc_layers'])

        self.disc_layers=disc_layers
        self.disc=Discriminator(self.disc_layers,self.optimizers[1],use_x=use_x)
        
        
        self.aux_losses=aux_losses
        if ((aux_losses is not None) and (aux_losses_weights is None)):
            aux_losses_weights=len(aux_losses)*[1]
        self.aux_losses_weights=aux_losses_weights
        self.feat_loss_weight=feat_loss_weight#5e-4 #TODO: check this
        self.grad_penalty_weight = 10
        
        #'Stateful' Metrics
        self.train_loss1 = tf.keras.metrics.Mean(name='train_loss1')
        self.train_loss2 = tf.keras.metrics.Mean(name='train_loss2')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        #self.test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
        #self.test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        #'Stateless' Losses
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        #self.l1_loss=lambda z: tf.reduce_mean(tf.abs(z))
        self.acc= lambda y,y_hat: tf.reduce_mean(tf.cast(tf.equal(
                    tf.argmax(y,axis=1),tf.argmax(y_hat,axis=1)),tf.float32))

        
        #Checkpoint objects
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), 
                                        optimizer=self.gen.optimizer,
                                        model=self)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.model_path
                                                  ,max_to_keep=100,
                                                  keep_checkpoint_every_n_hours=(10/60))
                                                  
        
        #For fit function initialization
        self.fit_init=False
        
        #For switching D and G mechanism
        #self.switch_loss = tf.keras.metrics.Mean(name='switch_loss')
        self.switch_loss = tf.keras.metrics.Sum(name='switch_loss')

        self.D_switch=True #start with Disc training
        self.switch_loss_thres=-1. #0.5 #decide aptly
        self.val_loss_min=1e7
        #self.switch_loss_reduction=0.02
        self.last_preserved_timestamp=-10
        self.D_G_iter_ratio = 5
        #self.trans2dB=lambda x: 20*(rescale_values(x,-30,5,-1,1))*np.log10(2)
        
        self.T_steps = T_steps

        return
    
    def find_z_shape(self,x):
        x_shape=tf.shape(x)
        
        if type(x.shape)==type((1,)):
            self.T_dim=x.shape[-2]
        else:
            self.T_dim=x.shape.as_list()[-2]
        
        assert (self.T_dim%self.z_up_factor==0), ("z_up_factor"
                                " must be a factor of input's time dimension")
        if self.n_dims_x is None:
            self.n_dims_x=x_shape.shape.as_list()[0]
        z_shape = [x_shape[i] for i in range(self.n_dims_x)]
        #z_shape = [x_shape[i] for i in range(x_shape.shape.as_list()[0])]
        z_shape[-2]=int(self.T_dim/self.z_up_factor) #(For correlated z) 
        z_shape[-1]=self.latent_size*1
        return z_shape
    
    def sample_z_like(self,x):
        #z_shape=[x_shape[0],x_shape[1],self.latent_size]#TODO: change this base on need
        z=tf.random.normal(shape=self.z_shape,mean=0.,stddev=1.,
                             dtype=tf.dtypes.float32)
        return z
    
    def update_D_switch(self):
        if self.switch_loss.result()<=self.switch_loss_thres:
            #print('\n Switch loss = {} is below threshold. Switching between G<->D\n'.format(self.switch_loss.result()))
            
            #values=([int(self.ckpt.step)]+[m.result() for m in self.metrics_list[:3]]
            #            +[time.time()-self.start])
            #print(self.template.format(*values))
            
            self.D_switch=not(self.D_switch) #flip the switch
            #self.train_loss1.reset_states()
            #self.train_loss2.reset_states()
# =============================================================================
#             if self.D_switch:
#                 print('Switch loss = {} is below threshold. Training Disc...\n'.format(self.switch_loss.result()))
#             else:
#                 print('Switch loss = {} is below threshold. Training Gen...\n'.format(self.switch_loss.result()))
#             
# =============================================================================
            self.switch_loss.reset_states() #start new

        return
    
    @tf.function
    def train_step_recon(self,data):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input. We move forward first, then calculate gradients 
            with Gradient Tape to move backwards.
        '''
        #noise = tf.random.normal([BATCH_SIZE, noise_dim])
        x,y=data
        z=self.sample_z_like(x)
        generator=self.gen
        
        with tf.GradientTape() as gen_tape:
            #print(z.get_shape().as_list(),cond_sig.get_shape().as_list())

            sig_hat = generator([z,x], training=True)
            
            recon_loss = self.mse(y,sig_hat)
    
        gradients = gen_tape.gradient(recon_loss, generator.trainable_variables)    
        generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        self.train_loss1(recon_loss)
        return
    
    def train_step(self,data):
        x,y=data
        if self.D_switch:
            #Update Discriminator
            gen_loss, disc_loss, cost=self.train_disc(x, y)
            self.switch_loss(-1/self.D_G_iter_ratio)
        else:
            #print(x.get_shape().as_list(),self.var_z.get_shape().as_list())
            #Update Generator
            gen_loss, disc_loss, cost=self.train_gen(x, y)
            self.switch_loss(-1.)
            self.train_loss(cost)
            
            
        self.update_D_switch()
        self.train_loss1(gen_loss)
        self.train_loss2(-disc_loss)
        #print('batch_no. = {}'.format(self.batch_counter))
        return
    
    @tf.function
    def train_gen(self,x,y):
        z=self.sample_z_like(x)
        #z = random.normal((self.batch_size, 1, 1, self.z_dim))
        #self.train_inverse_program(y,x,n_steps=self.T_steps)#updates self.var_z
        
        with tf.GradientTape() as t:
            y_hat = self.gen([z,x], training=True)
            fake_logits, fake_out_list = self.disc([y_hat,x], training=False,
                                                   return_all_out=True)
            gen_loss = self.gen.loss(fake_logits)
            
            #y_hat_opt = self.gen([self.var_z,x], training=True)
            #maml_loss=self.mse(y,y_hat_opt)
            #cost = gen_loss# + 1e3*maml_loss
            
            cost=[gen_loss]
            #print(f'At gen_loss={np.mean(~np.isnan(gen_loss.numpy()))}')
            if self.aux_losses is not None:
                aux_loss=[self.aux_losses_weights[i]*
                          self.aux_losses[i](y, y_hat) 
                          for i in range(len(self.aux_losses))]
                cost+=aux_loss
            cost = sum(cost)
            

            #No grad tracking needed for disc loss
            real_logits, real_out_list = self.disc([y,x], training=False, 
                                                    return_all_out=True)
            feat_loss=sum([self.mse(real_out_list[i],fake_out_list[i]) 
                            for i in range(len(fake_out_list))])
            cost+=(self.feat_loss_weight*feat_loss)

            disc_loss = self.disc.loss(real_logits,fake_logits)

        grad = t.gradient(cost, self.gen.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(grad, self.gen.trainable_variables))
        return gen_loss, disc_loss, cost

    @tf.function
    def train_disc(self, x, y):
        z=self.sample_z_like(x)
        #z = random.normal((self.batch_size, 1, 1, self.z_dim))
        f_op4gp=lambda y_interp: self.disc([y_interp,x], training=True)
        with tf.GradientTape() as t:
            y_hat = self.gen([z,x], training=True)
            #print(f'At y_hat={np.mean(~np.isnan(y_hat.numpy()))}')

            fake_logits = self.disc([y_hat,x], training=True)
            #print(f'At fake_logits={np.mean(~np.isnan(fake_logits.numpy()))}')

            real_logits = self.disc([y,x], training=True)
            disc_loss = self.disc.loss(real_logits,fake_logits)
            gp = self.gradient_penalty(f_op4gp, y, y_hat)
            #print(type(disc_loss),type(gp))
            cost = disc_loss + self.grad_penalty_weight * gp
            #print(f'At disc_loss={np.mean(~np.isnan(disc_loss.numpy()))}')

            
            
        #No grad tracking needed for gen loss
        gen_loss = self.gen.loss(fake_logits)
        
        grad = t.gradient(cost, self.disc.trainable_variables)
        #print(f'At grad={np.mean(np.array([np.mean(~np.isnan(grad[i].numpy())) for i in range(len(grad))]))}')

        self.disc.optimizer.apply_gradients(zip(grad, self.disc.trainable_variables))
        return gen_loss, disc_loss, cost

    def gradient_penalty(self, f, real, fake):
        real_shape=tf.shape(real)
        n_dims= self.n_dims_y*1#real_shape.shape.as_list()[0]
        alpha_shape=n_dims*[1]
        alpha_shape[0]=real_shape[0]*1
        alpha = tf.random.uniform(alpha_shape, 0., 1.,dtype=tf_dtype)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.reduce_sum(tf.square(grad), axis=list(range(1,n_dims)))
        slopes=tf.sqrt(tf.maximum(slopes, eps))
        
        gp = tf.reduce_mean((slopes - 1.)**2)
        #print(gp.shape,gp)
        return gp
    
    def test_step_recon(self,data, in_prediction=False):
        '''
            x is either a condition or just a batch size indicator
        '''
        x,y=data
        z=self.sample_z_like(x)
        generator=self.gen
        y_hat = generator([z,x], training=False)
        if  in_prediction:
            return y_hat
        recon_loss = self.mse(y,y_hat)
        self.test_loss(recon_loss)
        #self.test_metric(x, predictions)
        return
    
    def test_step(self,data, in_prediction=False, rnn_state=None, 
                  rnn_state_out_no=-1):
        '''
            x is either a condition or just a batch size indicator
        '''
        x,y=data
        z=self.sample_z_like(x)
        y_hat = self.gen([z,x], training=False, rnn_state=rnn_state,
                         rnn_state_out_no=rnn_state_out_no)
        if  in_prediction:
            return y_hat
        fake_logits = self.disc([y_hat,x], training=False)
        real_logits = self.disc([y,x], training=False)
        disc_loss = self.disc.loss(real_logits,fake_logits)
        self.test_loss(-disc_loss)
        #self.test_metric(x, predictions)
        return
    
    @tf.function
    def val_step_recon(self, data,in_prediction=False):
        return self.test_step_recon(data,in_prediction=in_prediction)

    @tf.function
    def val_step(self, data,in_prediction=False):
        return self.test_step(data,in_prediction=in_prediction)
    

    def fit(self, data, summaries, epochs, batch_size=16, path2figs='.',
            local_save_flag=True):
        '''
            This fit function runs training and testing.
        '''
        if self.mode=='GAN_recon':
            train_step,val_step=self.train_step_recon,self.val_step_recon
            self.template = ('Epoch {}, Train_Loss: {}, Val_Loss: {},'
                        'Time used: {} \n')
            #loss_eval= lambda p,q: q.result()
            self.metrics_list = [self.train_loss1,self.test_loss]

        elif self.mode=='GAN':
            train_step,val_step=self.train_step,self.val_step
            self.template = ('Epoch {}, Gen_Loss: {}, neg_Disc_Loss: {},'
                             ' net_Gen_loss: {}, Val_Loss: {},'
                        'Time used: {} \n')
            self.metrics_list = [self.train_loss1,self.train_loss2,
                                 self.train_loss,self.test_loss]
                                 #self.switch_loss]
        else:
            assert False, 'mode can only be GAN or GAN_recon\n'
            
        train, val=data
        self.n_dims_x,self.n_dims_y=len(val[0].shape),len(val[1].shape)
        
        #batch_size_train,N=find_batch_size(train[0].shape[0],thres=1000)
        self.batch_size_train = batch_size*1
        #batch_size_val,N_test=find_batch_size(val[0].shape[0],thres=100,
        #                                      mode='val')
        batch_size_val = batch_size*1
        
        #n_tsteps=train[0].shape[1]
        z_shape=list(train[0].shape)
        z_shape[0]=self.batch_size_train*1
        z_shape[-1]=self.latent_size*1
        
        
        self.var_z=tf.Variable(tf.zeros(z_shape,dtype=tf.dtypes.float32))
        #TODO: Overridden stuff here
        #batch_size_train*=8
        #batch_size_val=int(batch_size_val/2)
        
        print(self.n_dims_x,self.n_dims_y,self.batch_size_train,batch_size_val,'\n')
        train_ds=make_data_pipe(train,self.batch_size_train)
        val_ds=make_data_pipe(val,batch_size_val,shuffle=False,
                              drop_remainder=True)
        
        # Some logging utilities
        train_summary_writer, test_summary_writer=summaries
        
# =============================================================================
#         steps2gen_samples = 5 #save samples to disk for visualizing, during training
#         n_vis_samples=12
#         rand_val_idx=np.random.randint(0,val[0].shape[0],size=n_vis_samples)
#         vis_data=[val[0][rand_val_idx],val[1][rand_val_idx]]
#         samples = self.trans2dB(vis_data[1])
#         vmax=np.max(samples.reshape(-1))
#         vmin=vmax-60
#         save_suf ='true'
#         os.makedirs(path2figs,exist_ok=True)
#         plot_images(samples,vmin=vmin,vmax=vmax,save_suf=save_suf,
#                             path2figs=path2figs)
# =============================================================================
        
        for epoch in range(epochs):
            self.start = time.time()
            self.batch_counter = 0
            # Reset the metrics for the next epoch
            for m in self.metrics_list:
                m.reset_states()
            
            for images in train_ds:
                if epoch==0:
                    x,y=images
                    self.z_shape=self.find_z_shape(x) #update z_shape
                train_step(images)
                self.batch_counter+=1
                #if epoch==0: print(self.gen.layer_list[0].get_weights()[0])
            for test_images in val_ds:
                val_step(test_images)

            
            self.ckpt.step.assign_add(1) #increment ckpt counter at every epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss2.result(), 
                                  step=int(self.ckpt.step))
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), 
                                  step=int(self.ckpt.step))

            
            if self.mode=='GAN':
                values=([int(self.ckpt.step)]+[m.result() for m in self.metrics_list]
                        +[time.time()-self.start])
                print(self.template.format(*values))
                
            elif self.mode=='GAN_recon':
                values=([int(self.ckpt.step)]+[m.result() for m in self.metrics_list[:2]]
                        +[time.time()-self.start])

                print(self.template.format(*values))
            else:
                assert False, 'mode can only be GAN or GAN_recon\n'
                
            val_loss=self.test_loss.result()
            if ((self.save_flag) and (local_save_flag)):
                #cond2reset_val_loss_state=(time.time()-
                 #   self.manager._keep_checkpoint_every_n_hours * 3600.
                  #  >= (self.last_preserved_timestamp+10))
                cond2reset_val_loss_state=(int(self.ckpt.step)%50==0)
                #This reset delay MUST be greater than keep_checkpoint_every_n_hours
                #cond2reset_val_loss_state=(int(self.ckpt.step.numpy())%200==0)
                if cond2reset_val_loss_state:
                    print('Resetting val_loss monitoring state... \n')
                    self.val_loss_min=1e7 #reset val_loss_state for saving
                if val_loss<=self.val_loss_min:
                    print('Saving Model...\n')
                    #self.save_weights(self.model_path,save_format='tf')
                    self.manager.save(checkpoint_number=int(self.ckpt.step))
                    self.val_loss_min=val_loss
                    self.last_preserved_timestamp=time.time()
            
# =============================================================================
#             if epoch%(steps2gen_samples)==0:
#                 save_suf ='gen_{}'.format(int(self.ckpt.step))
#                 samples = self.test_step(vis_data,in_prediction=True)
#                 samples = self.trans2dB(samples)
#                 plot_images(samples,vmin=vmin,vmax=vmax,save_suf=save_suf,
#                             path2figs=path2figs)
# =============================================================================
                
            sys.stdout.flush() #for enabling real-time log file printing
        print('Saving Final epoch Model...\n')
        #self.save_weights(self.model_path,save_format='tf')
        self.manager.save(checkpoint_number=int(self.ckpt.step))
    
    
    def predict(self,test_data,rnn_state=None,rnn_state_out_no=-1):
        if self.mode=='GAN_recon':
            test_step=self.test_step_recon
        else:
            test_step=self.test_step

        self.test_loss.reset_states()
        test_y_hat_list=[]
        #TODO: Reomve fixed batch_size=1 when needed
        batch_size_test,N_test=find_batch_size(test_data[0].shape[0],thres=1024
                                              ,mode='val')
        #print(batch_size_test)
        #batch_size_test,N_test=1,test_data[0].shape[0]
        self.z_shape=self.find_z_shape(test_data[0][0:batch_size_test])
        for i in range(N_test):
            # Reset the metrics for the next batch and test z values
            y_hat=test_step([test_data[0][i:i+batch_size_test],
                            test_data[0][i:i+batch_size_test]] #Dummy y to test_step
                            ,in_prediction=True,rnn_state=rnn_state,
                            rnn_state_out_no=rnn_state_out_no)
            test_y_hat_list.append(y_hat)

        #test_data.append(np.concatenate(test_y_hat_list,axis=0))
        #return test_data
        return np.concatenate(test_y_hat_list,axis=0)
    
    def train_inverse_program(self,y,x,F=None,n_steps=5,
                       optimizer=tf.keras.optimizers.Adam(1e-4)):
        if F is None:
            F=self.gen
        #self.var_z.assign(self.sample_z_like(x))
        #self.var_z.assign(tf.zeros([self.batch_size_train,self.latent_shape]))
        self.var_z.assign(tf.zeros_like(self.var_z))

        #define the optimization step
        def optimize_step():
            with tf.GradientTape() as tape:
                y_hat=F([self.var_z,x],training=False)  # Forward pass
                # Compute the loss value
                loss=self.mse(y,y_hat)
            # Compute gradients
            trainable_vars = [self.var_z]
            gradients = tape.gradient(loss, trainable_vars)
            # Update vars
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            return
        
        #Run optimization
        for step in np.arange(n_steps):
            optimize_step()
            #print('Step={}, Loss={}\n'.format(step,loss))
        #print('Done running inverse program \n')
        #z=self.var_z.numpy()
        return
    
    def call(self,x,training=False):
        '''
        x is either a condition or just a batch size indicator
        '''
        self.z_shape=self.find_z_shape(x)
        z=self.sample_z_like(x)
        y_hat=self.gen([z,x],training=training)
        d_out=self.disc([y_hat,x],training=training)
        return [y_hat,d_out]

#%%
class Model_CondWGAN_stitch(Model_CondWGAN):
    
    @tf.function
    def train_gen(self,x,y):
        #z = random.normal((self.batch_size, 1, 1, self.z_dim))
        #self.train_inverse_program(y,x,n_steps=self.T_steps)#updates self.var_z
        alw=x[:,:,-1:]
        x=x[:,:,:-1]
        
        z=self.sample_z_like(x)

        
        with tf.GradientTape() as t:
            y_hat = self.gen([z,x], training=True)
            fake_logits = self.disc([y_hat,x], training=True)
            gen_loss = self.gen.loss(fake_logits)
            
            #y_hat_opt = self.gen([self.var_z,x], training=True)
            #maml_loss=self.mse(y,y_hat_opt)
            #cost = gen_loss# + 1e3*maml_loss
            
            cost=[gen_loss]
            if self.aux_losses is not None:
                #print(alw.shape,y.shape,y_hat.shape)
                aux_loss=[self.aux_losses_weights[i]*
                          self.aux_losses[i](y*alw, y_hat*alw) 
                          for i in range(len(self.aux_losses))]
                cost+=aux_loss
            cost = sum(cost)
        #No grad tracking needed for disc loss
        real_logits = self.disc([y,x], training=True)
        disc_loss = self.disc.loss(real_logits,fake_logits)

        grad = t.gradient(cost, self.gen.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(grad, self.gen.trainable_variables))
        return gen_loss, disc_loss, cost
    
    @tf.function
    def train_disc(self, x, y):
        x=x[:,:,:-1]
        z=self.sample_z_like(x)
        #z = random.normal((self.batch_size, 1, 1, self.z_dim))
        f_op4gp=lambda y_interp: self.disc([y_interp,x], training=True)
        with tf.GradientTape() as t:
            y_hat = self.gen([z,x], training=True)
            fake_logits = self.disc([y_hat,x], training=True)
            real_logits = self.disc([y,x], training=True)
            disc_loss = self.disc.loss(real_logits,fake_logits)
            gp = self.gradient_penalty(f_op4gp, y, y_hat)
            #print(type(disc_loss),type(gp))
            cost = disc_loss + self.grad_penalty_weight * gp
            
        #No grad tracking needed for gen loss
        gen_loss = self.gen.loss(fake_logits)
        
        grad = t.gradient(cost, self.disc.trainable_variables)
        self.disc.optimizer.apply_gradients(zip(grad, self.disc.trainable_variables))
        return gen_loss, disc_loss, cost
    
    @tf.function
    def val_step(self,data, in_prediction=False):
        '''
            x is either a condition or just a batch size indicator
        '''
        x,y=data
        x=x[:,:,:-1]
        
        z=self.sample_z_like(x)
        y_hat = self.gen([z,x], training=False)
        if  in_prediction:
            return y_hat
        fake_logits = self.disc([y_hat,x], training=False)
        real_logits = self.disc([y,x], training=False)
        disc_loss = self.disc.loss(real_logits,fake_logits)
        self.test_loss(-disc_loss)
        #self.test_metric(x, predictions)
        return