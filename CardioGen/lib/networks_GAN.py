import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.keras.activations import relu
#from .ops import Res_Block

tf.keras.backend.set_floatx('float32')
#%%
class downsample(tf.keras.layers.Layer):
    def __init__(self,filters, kernel_size, strides,activation='Lrelu',
                 apply_batchnorm=True):
        super(downsample, self).__init__()
        
        if activation=='Lrelu':
            activation=None
            Lrelu_flag=True
            alpha=0.3
        elif activation=='relu':
            activation=None
            Lrelu_flag=True
            alpha=0.
        else:
            Lrelu_flag=False
            
        initializer = tf.random_normal_initializer(0., 0.02)
        self.layer_list=[layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                        strides=strides,activation=activation,padding='same',
                        kernel_initializer=initializer)]
        
        if apply_batchnorm:
            self.layer_list.append(layers.BatchNormalization())
        
        if Lrelu_flag:
            self.layer_list.append(layers.LeakyReLU(alpha=alpha))
        
    def call(self,x,training=None):
        for lay in self.layer_list:
            x=lay(x,training=training)
            #print(x.shape.as_list())
        return x
    
class upsample(tf.keras.layers.Layer):
    def __init__(self,filters, kernel_size, strides,activation=None, 
                 apply_batchnorm=True, apply_dropout=False):
        super(upsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.layer_list=[layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, 
                        strides=strides,activation=activation,padding='same',
                        kernel_initializer=initializer)]
        
        if apply_batchnorm:
            self.layer_list.append(layers.BatchNormalization())
        if apply_dropout:
            self.layer_list.append(layers.Dropout(0.25))
        self.layer_list.append(layers.ReLU())
        
    def call(self,x,training=None):
        for lay in self.layer_list:
            x=lay(x,training=training)
            #print(x.shape.as_list())
        return x

#%%
def get_GAN_layers_default(latent_size=256,req_list=['gen_layers','disc_layers']):
    def get_gen_layers():
        def_gen_layers=[]
        def_gen_layers.append(layers.GRU(64,return_sequences=True,name='gru_gen_1'))
        def_gen_layers.append(layers.Conv1D(1,1,name='conv1d_gen_1'))
        return def_gen_layers
    
    def get_disc_layers():
        cntr=0
        def_disc_layers=[]
        #def_disc_layers.append(layers.GRU(64,return_sequences=True,name='gru_disc_1'))#,unroll=True))
        def_disc_layers.append(tf.keras.layers.RNN(tf.keras.layers.GRUCell(64),return_sequences=True))
        def_disc_layers.append(layers.Dropout(0.05,name='drop_disc_1',noise_shape=[None, 1, 64]))
        def_disc_layers.append(layers.Conv1D(4,5,name='conv1d_disc_1',padding="same"))
        def_disc_layers.append(layers.MaxPooling1D(pool_size=5, strides=5, padding="same"))
        def_disc_layers.append(layers.Conv1D(2,2,name='conv1d_disc_2',padding="same"))
        def_disc_layers.append(layers.MaxPooling1D(pool_size=2, strides=2, padding="same"))
        def_disc_layers.append(layers.Flatten(name='flat_disc_1'))
        
        def_disc_layers.append(layers.Dense(1,name='fc_disc_1'))
        
        #def_disc_layers[0]._could_use_gpu_kernel = False
        return def_disc_layers
    
    return_list=[]
    
    for req in req_list:
        if req=='gen_layers':
            def_gen_layers=get_gen_layers()
            return_list.append(def_gen_layers)
        elif req=='disc_layers':
            def_disc_layers=get_disc_layers()
            return_list.append(def_disc_layers)
        else:
            assert False, "Requirements must be in ['gen_layers','disc_layers']"
    
    if len(return_list)==0:
        return None
    elif len(return_list)==1:
        return return_list[0]
    else:
        return return_list

#%%
def get_GAN_layers_conv(latent_size=256,req_list=['gen_layers','disc_layers']):
    def get_gen_layers():
        def_gen_layers=[]
        def_gen_layers.append(layers.GRU(64,return_sequences=True,name='gru_gen_1'))
        def_gen_layers.append(layers.Conv1D(1,1,name='conv1d_gen_1'))
        return def_gen_layers
    
    def get_disc_layers():
        f_list=[8,16,16,32,64]
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
            
            # def_disc_layers.append(downsample(filters=f_list[i], 
            #                         kernel_size=(1,k_list[i]),strides=(1,1),
            #                         activation=None))
            # def_disc_layers.append(layers.MaxPooling2D(pool_size=(1,s_list[i]), 
            #                         strides=(1,s_list[i]), padding="same",
            #                         name='maxpool_disc_{}'.format(i)))
            # def_disc_layers.append(layers.Activation(act_list[i],
            #                         name='act_disc_{}'.format(i)))
            
        def_disc_layers.append(layers.Flatten(name='flat_disc_1'))
        def_disc_layers.append(layers.Dense(1,name='fc_disc_1'))
        
        #def_disc_layers[0]._could_use_gpu_kernel = False
        return def_disc_layers
    
    return_list=[]
    
    for req in req_list:
        if req=='gen_layers':
            def_gen_layers=get_gen_layers()
            return_list.append(def_gen_layers)
        elif req=='disc_layers':
            def_disc_layers=get_disc_layers()
            return_list.append(def_disc_layers)
        else:
            assert False, "Requirements must be in ['gen_layers','disc_layers']"
    
    if len(return_list)==0:
        return None
    elif len(return_list)==1:
        return return_list[0]
    else:
        return return_list
    
#%%
def get_GAN_layers_conv_dil(latent_size=256,req_list=['gen_layers','disc_layers']):
    def get_gen_layers():
        def_gen_layers=[]
        def_gen_layers.append(layers.GRU(64,return_sequences=True,name='gru_gen_1'))
        def_gen_layers.append(layers.Conv1D(1,1,name='conv1d_gen_1'))
        return def_gen_layers
    
    def get_disc_layers():
        f_list=[16,16,32,32,64]
        k_list=[4,4,3,3,3]
        s_list=[2,2,2,2,2]
        act_list=5*['relu']
        def_disc_layers=[]
        for i in range(len(act_list)):
            def_disc_layers.append(layers.Conv1D(f_list[i],k_list[i],
                            name='conv1d_disc_{}'.format(i),padding="same",
                            dilation_rate=s_list[i],activation=act_list[i]))
        def_disc_layers.append(layers.GlobalAveragePooling1D(name='GAP_disc_1'))
        #def_disc_layers.append(layers.Flatten(name='flat_disc_1'))
        def_disc_layers.append(layers.Dense(1,name='fc_disc_1'))
        
        #def_disc_layers[0]._could_use_gpu_kernel = False
        return def_disc_layers
    
    return_list=[]
    
    for req in req_list:
        if req=='gen_layers':
            def_gen_layers=get_gen_layers()
            return_list.append(def_gen_layers)
        elif req=='disc_layers':
            def_disc_layers=get_disc_layers()
            return_list.append(def_disc_layers)
        else:
            assert False, "Requirements must be in ['gen_layers','disc_layers']"
    
    if len(return_list)==0:
        return None
    elif len(return_list)==1:
        return return_list[0]
    else:
        return return_list