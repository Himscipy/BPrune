import  numpy as np
import tensorflow as tf
import src.utils as UT
import src.Prune_BNN as BP


def main(args):
    preProc = UT.PreProcess(args)

    seed = preProc.Setup_Seed()
    config = preProc.create_config_proto()

    # Data loading
    dataObj = UT.DataAPI(args)
    
    (_, _), (x_test, y_test) = dataObj.load_CIFAR10_data()

    test_batch_gen = dataObj.train_input_generator(x_test,y_test)    
    
    Images, Labels = next(test_batch_gen)

    Sess = tf.Session(config=config)

    # Initiate Prune Object
    # Prune_Model = BP.Prune_Model(args,Sess,Images,Labels)






if __name__ =='__main__':
    init = UT.Initialize()
    args = init.FlagParser()
    tf.app.run(main=main(args))