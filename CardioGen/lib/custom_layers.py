import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float32')

class downsample(tf.keras.layers.Layer):
    def __init__(self,filters, kernel_size, strides,activation='relu',
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
                        strides=(1,1),activation=activation,padding='same',
                        kernel_initializer=initializer),
                        layers.MaxPooling2D(pool_size=strides,padding='same')]
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