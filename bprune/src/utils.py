import numpy as np
import tensorflow as tf
from absl import flags
import os
from tensorflow.python.keras.datasets.cifar import load_batch
import tensorflow_probability as tfp
tfd = tfp.distributions

class Initialize:
    def __init__(self):
        super().__init__()
    
    def FlagParser(self):
        flags.DEFINE_string("data_dir",
                    default=os.path.join('./results',
                                         "bayesian_neural_network/data/"),
                    help="Directory where data is stored (if using real data).")
        flags.DEFINE_string("model_dir",
            default='/projects/datascience/hsharma/bnn_horovod/TFP_CIFAR10/RunScript/BNN_5BL_Scaling/results/bayesian_VGG_Data_8rank',
            help="Directory to put the model's fit.")
        flags.DEFINE_integer("num_monte_carlo", default=200,help="Network draws to compute predictive probabilities.")
        flags.DEFINE_integer("batch_size", default=64,help="Define the batch size for test data")
        flags.DEFINE_boolean("subtract_pixel_mean",default=True,help="Boolean for normalizing the images (used for CIFAR10)")
        flags.DEFINE_bool("fake_data",default=None,help="If true, uses fake data. Defaults to real data.")
        flags.DEFINE_string("data_path",default="/projects/datascience/hsharma/bnn_horovod/TFP_CIFAR10/RunScript/cifar-10-batches-py",help="Path to load data  from directory")
        flags.DEFINE_integer('num_intra', default=128,help='Intra thread num_intra')
        flags.DEFINE_integer('num_inter', default=1,help='num_inter')
        flags.DEFINE_bool('cluster',default=False,help='Define the run in the')
        flags.DEFINE_integer('kmp_blocktime', default=0,help='KMP BLOCKTIME')
        flags.DEFINE_string('kmp_affinity', default='granularity=fine,verbose,compact,1,0',help='KMP AFFINITY')
        flags.DEFINE_bool('Verbose',default=False,help='Verbosre to print extra details about the model')
        flags.DEFINE_bool('Modify_Weights',default=False,help='Turn on/off the weight modification method Currently supports only second last Layer of VGG-16')
        flags.DEFINE_bool('Plotting',default=False,help='Turn on/off the plotting')
        flags.DEFINE_string('model_ckpt',default='model.ckpt-0.meta',help='Checkpoint step to restore the weights')
        flags.DEFINE_integer('iterations', default=1,help=' Number of iterations to perform')
        flags.DEFINE_float('Thres_val',default=10.0, help='Value of Threshold to use at restart')
        flags.DEFINE_bool('Per_Prune',default=False,help='Turn on/off the Percent_Prune')
        flags.DEFINE_bool('CIFAR10',default=False,help='Data flag to use CIFAR10')
        flags.DEFINE_bool('MNIST_BIGDATA',default=False,help='Data flag to use MNIST transform data')


        FLAGS = flags.FLAGS

        return FLAGS


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
        seed = tfd.SeedStream(original_seed, salt="random_beta")
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

