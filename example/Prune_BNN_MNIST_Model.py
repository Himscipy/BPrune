import numpy as np
import tensorflow as tf
from bprune.src.utils import Initialize
# from absl import flags
import argparse

# FLAGS = flags.FLAGS

# import bprune.src.utils as UT
# from  bprune.src.Prune_BNN import *
# from ...src.Prune_BNN import Prune_Model



def main(args):

    print(args)
    
    # PreObj = UT.PreProcess(args)

    # config = PreObj.create_config_proto()
    
    # Seed_set = PreObj.Setup_Seed()

    # sess = tf.Session(config=config)

    # # prune_obj = Prune_Model(args,sess,Images,Labels)
    return
    
if __name__ == "__main__":
    init_obj = Initialize()
    
    # args = init_obj.FlagParser()
    args = init_obj.ArgParser()
    
    print (args.parse_args())
    # print(args.flag_values_dict())
    # tf.compat.v1.app.run(main=main(args))
