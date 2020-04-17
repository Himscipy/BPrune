import numpy as np
import tensorflow as tf
import bprune.src.utils as UT
import bprune.src.Prune_BNN as BP

# from absl import flags
import argparse

# FLAGS = flags.FLAGS

# import bprune.src.utils as UT
# from  bprune.src.Prune_BNN import *
# from ...src.Prune_BNN import Prune_Model



def main(args):
 
    # PreObj = UT.PreProcess(args)

    # config = PreObj.create_config_proto()
    
    # Seed_set = PreObj.Setup_Seed()

    # # Data loading
    # dataObj = UT.DataAPI(args)
    
    # (_, _), (x_test, y_test) = dataObj.load_MNIST_data()

    # test_batch_gen = dataObj.train_input_generator(x_test,y_test)    
    
    # Images, Labels = next(test_batch_gen)

    # Sess = tf.Session(config=config)


    # prune_obj = BP.Prune_Model(args,Sess,Images,Labels)

    # ####################################
    # ## Run Prunning....!!
    # ####################################
    # prune_obj.RunPrune()


    PostObj = UT.PostProcess(args)
    

    return
    
if __name__ == "__main__":
    init_obj = UT.Initialize()
    args = init_obj.ArgParser()
    print ((args))
    main(args)
   