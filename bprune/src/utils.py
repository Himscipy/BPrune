import numpy as np
import tensorflow as tf
from absl import flags
import pickle
import os
import re
import argparse
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions


class Initialize:
    def __init__(self):
        super().__init__()
        

    def ArgParser(self):
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@ArgFile.txt', description="Bayesian neural network using tensorflow_probability")
        
        parser.add_argument("--data_dir",type=str, default=os.path.join('./results','/BNN_Run/data/'))
        parser.add_argument("--model_dir",type=str,default='/home/hsharma/WORK/Project_BNN/Theta_data/bayesian_VGG_Data_8rank/train_logs/')
        parser.add_argument("--data_path",type=str,default='/home/hsharma/WORK/Project_BNN/bnn_horovod/TFP_CIFAR10/cifar-10-batches-py')
        parser.add_argument("--model_ckpt",type=str,default='model.ckpt-1562.meta')
        parser.add_argument("--batch_size",type=int,default=64)
        parser.add_argument("--num_monte_carlo",type=int,default=200)
        parser.add_argument("--thres_val",type=float,default=10.0)
        parser.add_argument("--subtract_pixel_mean",action='store_true')
        parser.add_argument("--cluster",action='store_true')
        parser.add_argument("--verbose",action='store_true')
        parser.add_argument("--plotting",action='store_true')
        parser.add_argument("--inference",action='store_true')
        parser.add_argument("--prune",action='store_true')
        parser.add_argument("--cifar10",action='store_true')
        parser.add_argument("--name_scope",action='store_true')
        parser.add_argument("--horovod_used",action='store_true')
        parser.add_argument("--kmp_blocktime",type=int,default=0,required=False)
        parser.add_argument("--kmp_affinity",type=str,default='granularity=fine,verbose,compact,1,0',required=False)
        parser.add_argument("--num_intra",type=int,default=128,required=False)
        parser.add_argument("--num_inter",type=int,default=1,required=False)
        

        return parser.parse_args()
    

class PreProcess:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    def create_config_proto(self):

        '''HS: TF config setup'''
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = self.FLAGS.num_intra
        config.inter_op_parallelism_threads = self.FLAGS.num_inter
        config.allow_soft_placement         = True
        if self.FLAGS.cluster:
            os.environ['KMP_BLOCKTIME'] = str(self.FLAGS.kmp_blocktime)
            os.environ['KMP_AFFINITY'] = self.FLAGS.kmp_affinity
        return config

    def Setup_Seed(self):
        seed_adjustment= 700 # worked 690
        np.random.seed(6118 + seed_adjustment)
        tf.set_random_seed(1234 + seed_adjustment)
        original_seed = 1092 + seed_adjustment
        seed = tfp.util.SeedStream(original_seed, salt="random_beta")
        #print('Seed Initialized')
        #print("Original_seed:",original_seed,"Seed",seed())
        return seed
    

class DataAPI:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
    

    def load_MNIST_data(self):
        ## Loads only npz data format for MNIST

        path = self.FLAGS.data_path
        with np.load( os.path.join(path, 'mnist.npz')) as f:
                x_train,y_train = f['x_train'],f['y_train']
                x_test,y_test = f['x_test'],f['y_test']

        x_train = np.reshape(x_train, (-1, 784)) / 255.0
        x_test = np.reshape(x_test, (-1, 784)) / 255.0

        # x_test.reshape(*IMAGE_SHAPE)
        # x_train.reshape(*IMAGE_SHAPE)
        return (x_train,y_train), (x_test,y_test)

    def dataset_gen(self):
        '''
        Description: 
        To load the Tf record dataset
        '''
        
        batch_size = self.FLAGS.batch_size
        data_path = self.FLAGS.data_path

        feature_set_in =  {


            'height':  tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True), 
            'width':  tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True), 
            'depth':  tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True), 
            'label':  tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True), 
            'image_raw':  tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)      
                        }
        
        # List all the Files
        # tf.io.gfile.glob(args.data_path)
        data_loc = (os.path.join(data_path , "Test_Rank_*.tfrecords"))
        # print (data_loc)
        filenames_glob = tf.io.gfile.glob(data_loc) 
        
        # print (filenames_glob)
        
        data_len = sum([1 for fn in filenames_glob for record in tf.python_io.tf_record_iterator(fn)])
        

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, feature_set_in)
            image = tf.decode_raw(parsed_features['image_raw'],tf.uint8)
            image = tf.reshape(image, [28,28,1])
            scaled_image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
        
            class_label = tf.cast(parsed_features['label'], tf.int64)
            class_label =  tf.squeeze(class_label)
            
            return scaled_image, class_label

        dset = tf.data.TFRecordDataset(filenames_glob)
           
        dset = dset.map(_parse_function)
        dset = dset.batch(batch_size)
        
        iterator = dset.make_one_shot_iterator()

        next_element = iterator.get_next()
        
        return iterator,next_element,data_len,dset

    # Load the Model
    def load_CIFAR10_data(self):
        """Loads CIFAR10 dataset.
        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        
        #path = '/projects/datascience/hsharma/bnn_horovod/TFP_CIFAR10/RunScript/cifar-10-batches-py'
        path = self.FLAGS.data_path
        #path = '/home/hsharma/WORK/Project_BNN/bnn_horovod/TFP_CIFAR10/cifar-10-batches-py'
        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
            y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train /= 255
        x_test /= 255

        if self.FLAGS.subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean
            
        # y_train = y_train.flatten()
        # y_test = y_test.flatten()
        y_train= np.int32(y_train)
        y_test= np.int32(y_test)

        return (x_train, y_train), (x_test, y_test)

    def train_input_generator(self,x_train, y_train):
        assert len(x_train) == len(y_train)
        batch_size = self.FLAGS.batch_size
        while True:
            p = np.random.permutation(len(x_train))
            x_train, y_train = x_train[p], y_train[p]
            index = 0
            while index <= len(x_train) - batch_size:
                yield x_train[index:index + batch_size], np.reshape(y_train[index:index + batch_size], -1),
                index += batch_size


class Graph_Info_Writer:
    def __init__(self,CaseDir):
        super().__init__()
        self.Case_dir = CaseDir
        self.Graph_Info_Writer()
    
    def Graph_Info_Writer(self):
        # Dump graph Trainable operations and variables.
        with open( os.path.join(self.Case_dir ,"LayerNames.txt"), 'w') as _out:
            total_parameters = 0
            for variable in tf.trainable_variables():
                this_variable_parameters = np.prod([s for s in variable.shape])
                total_parameters += this_variable_parameters
                _out.write("{}\n".format(variable.name))
            
            _out.close()
                
        # Writing the Name of other operations from the Graph.
        F_write = open(os.path.join(self.Case_dir,'Ops_name_BNN.txt'),'w')
        
        for op in tf.get_default_graph().get_operations():
            F_write.write(str(op.name)+'\n')
        F_write.close()
        
        return

class PostProcess:
    def __init__(self,Flags):
        self.FLAGS = Flags
        self.Data = self.Load_Data()

    def Load_Data(self):
        fnames = self.File_path()
        assert (len(fnames) == 1), "No Output File found...!!"

        if len(fnames) == 1:
            with open(fnames[0],'rb') as load:
                Data = pickle.load(load)  
        else:
            Data = []
            for fname in fnames:
                with open(fnames,'rb') as load:
                    data = pickle.load(load)  
                    Data.append(data)

        return Data

    def File_path(self):
        path_dir = self.FLAGS.data_dir
        fnames = []    
        for Root,dir_,files in os.walk(path_dir):
            for file_ in files:
                print (file_)
                if re.search('Run_InferenceMode_\w',file_):
                    fnames.append(os.path.join(Root,file_))
        return fnames
    
    def Save_Prune_Model(self):
        

        return

