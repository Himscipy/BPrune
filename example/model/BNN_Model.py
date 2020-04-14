import os
import errno
import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
import pickle
import tensorflow_probability as tfp
import sys
from absl import flags 
import bprune.src.utils as UT



try:
  import seaborn as sns 
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfp_layers = tfp.layers
tf.logging.set_verbosity(tf.logging.INFO)
tfd = tfp.distributions


flags.DEFINE_string('data_path', default="/home/hsharma/WORK/Project_BNN/Theta_data", help="path to the dataset")
flags.DEFINE_string("model_dir", default='./results', help="Directory to put the model's fit.")
flags.DEFINE_integer("batch_size", default=64, help="Batch size. #16,32,128")
flags.DEFINE_integer("iterations", default=1000, help="Training iterations")
flags.DEFINE_float('learning_rate',default=0.001, help = "Learning rate for optimizer")
flags.DEFINE_bool("FC", default=True, help= "Fully connected network as default & False will be Conv")
flags.DEFINE_bool("verbose",default=False, help="To print model summary")

flags = flags.FLAGS

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

def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def main(_):
    
    class Dummy():
        pass

    
    dirmake = "result_" + "BNN_Test_iter"+str(flags.iterations)+"Model_FC_"+ str(flags.FC)
    dirmake = os.path.join(flags.model_dir , dirmake)
    #logdir = os.pathjoin(dirmake,("LOG" + str(hvd.size())+ "/"))
    if not os.path.exists(dirmake):
        os.makedirs(dirmake)        
   
    # Load MNIST dataset.
    # (60000,28,28), (60000)
    # (10000,28,28), (10000)
    with np.load( os.path.join(flags.data_path,'mnist.npz') ) as f:
        x_train,y_train = f['x_train'],f['y_train']
        x_test,y_test = f['x_test'],f['y_test']
        
    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    
    x_test = np.reshape(x_test, (-1, 784)) / 255.0
    
    # Build model...
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None])
    K = 10 # number of classes
    feature_shape = [28,28,1]

    filter_size = 256 
    
    if flags.FC:
        model = BNN_FC_model(filter_size,K) 
    else:
        model = BNN_conv_model_conv2(feature_shape,K,filter_size)
      
    

    logits = model(image)
    
    
    # %% Loss
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    labels_distribution = tfd.Categorical(logits=logits,name='label_dist')
    
    
    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(label))
    
    
    N = x_train.shape[0]
    kl = sum(model.losses) / N
    
    KLscale=1.0
    elbo_loss = neg_log_likelihood + KLscale * kl
    
    #predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)
    predictions = tf.argmax(input=logits, axis=1)
    
    train_accuracy, train_accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions)
    
    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(flags.learning_rate)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(elbo_loss, global_step=global_step)

    hooks = [

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step= flags.iterations ),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': elbo_loss},
                                   every_n_iter=500),
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = os.path.join(dirmake,'checkout')
    
    training_batch_generator = train_input_generator(x_train,y_train, batch_size=flags.batch_size)
    
    ## Creating some variable for runtime writing
    net = Dummy()
    net.plot = Dummy()
    net.Totalruntimeworker = []
    net.plot.RuntimeworkerIter = []
    net.plot.Loss= []
    net.plot.Accuracy= []
    net.plot.Iter_num= []

    #######################################################################
    # Storing mean and standard deviations for the trained weights
    #######################################################################
    qmeans = []
    qstds = []
    names = []
    for i, layer in enumerate(model.layers):
        try:
            # print(layer)
            q = layer.kernel_posterior
        except AttributeError:
            #print ("I am continuing layer has no attribute")
            continue
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())    

    
    # Writing all the graph details...
    UT.Graph_Info_Writer(dirmake)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        #Train Time start 
        start_Train_time = time.time()
        iter_num = 0
        if flags.verbose:
            model.summary()
        
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)

            _, Acc_,Up_opt,loss_ = mon_sess.run([train_op,train_accuracy,train_accuracy_update_op,elbo_loss],feed_dict={image: image_, label: label_})
            

            iter_num = iter_num + 1

            end_time = time.time()
            diff_time = end_time - start_Train_time
            net.plot.RuntimeworkerIter.append(diff_time)
            net.plot.Iter_num.append(iter_num)
            net.plot.Loss.append(loss_)
            net.plot.Accuracy.append(Acc_)
            print ("iter {} Batch RunTime: {:.3f} Acc: {:0.3f} Elbo_loss: {:0.3f} ".format(iter_num , diff_time,Acc_,loss_))
                    
            
            
            if (not mon_sess.should_stop()) == True:
                
                if iter_num % 500 == 0:
                    qm_vals, qs_vals = mon_sess.run((qmeans, qstds),feed_dict={image: image_, label: label_})
                        

        # Train Time End
        end_Train_time = time.time()
        diff_trainSess = end_Train_time-start_Train_time
        net.Totalruntimeworker.append(diff_trainSess)

        fname1 = os.path.join(dirmake,("SavedWeight_Mean_STD_DATA_"+str(iter_num)) )    
        
        with open(fname1, "wb") as out:
            pickle.dump([names,qm_vals, qs_vals], out)
        
  
        # Dumps results for 
        fname = os.path.join(dirmake,("PlotRunTimeIteration_" + str(iter_num)) )
        with open(fname, "wb") as out:
            pickle.dump([net.plot.RuntimeworkerIter,
                        net.plot.Loss,
                        net.plot.Accuracy,
                        net.plot.Iter_num,
                        net.Totalruntimeworker], out)
    


if __name__ == "__main__":
    tf.app.run()


