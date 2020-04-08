import numpy as np
import tensorflow as tf
from src.utils import *
# import bprune.src.utils as UT
# from  bprune.src.Prune_BNN import *
# from ...src.Prune_BNN import Prune_Model



def main(args):

    print(args.flag_values_dict())
    
    PreObj = PreProcess(args)

    config = PreObj.create_config_proto()
    
    Seed_set = PreObj.Setup_Seed()

    sess = tf.Session(config=config)

    # prune_obj = Prune_Model(args,sess,Images,Labels)

    


if __name__ == "__main__":
    init_obj = Initialize()
    args = init_obj.FlagParser()
    tf.app.run(main=main(args))