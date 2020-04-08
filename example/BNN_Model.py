##
#
#
##

import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time
from tensorflow import keras
import pickle
import tensorflow_probability as tfp
import sys

try:
  import seaborn as sns 
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfp_layers = tfp.layers
tf.logging.set_verbosity(tf.logging.INFO)
tfd = tfp.distributions



def BNN_FC_model(neurons,Num_class):
    """
    3-layer Denseflipout model.
    Note: The input shape is specific to MNIST data-set. 

    Input:
        @neurons: Number of neurons in each hidden layer
        @Num_class: Dimensions of the output. 
    Output:
        @model

    """
    # Define the Model structure 
    model = tf.keras.Sequential([
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,input_shape=(28*28,), name="den_1" )),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,  name="den_2")),
        (tfp_layers.DenseFlipout(Num_class,name="den_3"))])
    return model


def BNN_conv_model_conv2(feature_shape, Num_class,filter_size):
    """
    2-layer convolution model and 1 DenseFlipout Layer second.
    
    Input:
        @feature_shape: Input feature shape. 
        @Num_class: Output dimension from the last layer.
        @filter_size: Size of filter in each layer.
    Note: Default activation function except last layer is relu

    Output:
        @model
    """
    
    #   Define the Model structure 
    model = tf.keras.Sequential(
        [
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DFlipout(filters=filter_size,kernel_size=[5, 5],activation=tf.nn.relu,padding="SAME",name="Conv_1")),   
        (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME",name="Max_1")),    
        (tfp_layers.Convolution2DFlipout(filter_size,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_2")),         
        (tf.keras.layers.Flatten()),
        (tfp.layers.DenseFlipout(Num_class))
        ]
        )
    return model
