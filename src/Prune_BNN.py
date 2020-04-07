
import os
import sys
import re
import tensorflow as tf


class Prune_Model:
    def __init__(self, FLAGS, Sess):
        self.FLAGS = FLAGS
        self.sess = Sess
        self.Op_Dicts = Read_OpsFile()
        self.Layer_Name = Read_LayerFile()
    
    def Read_LayerFile(self):
        Layer_Name = []
        
        filepath = os.path.join(os.path.dirname(os.path.dirname(self.FLAGS.model_dir)),'LayerNames.txt')

        if not os.path.exists(filepath):
            print ('Layer File not exists.. at {}'.format(filepath))
            sys.exit()
        else:
            with open(filepath,'r') as file_read:
                content = file_read.read().splitlines()
            for line in content:
                #   print (line)
                tmp = line.split('/')[-1]
                if re.search("bias_posterior_loc", tmp):
                    pass
                elif re.search("batch_normalization",tmp):
                    pass
                elif re.search("Variable:0",tmp):
                    pass
                else:
                    Layer_Name.append(line)

        Global_List = []
        for i in range(int(len(Layer_Name)/2)):
            tmp = []
            tmp.append(Layer_Name[i*2])
            tmp.append(Layer_Name[1 + i*2])
            Global_List.append(tmp)
        #   print (Global_List)
        return Global_List

        
    def Read_OpsFile(self):

        Name = []
        label_dist = []
        acc_val = []
        acc_up = []
        filepath = os.path.join(os.path.dirname(os.path.dirname(self.FLAGS.model_dir)),'Ops_name_BNN.txt')

        if not os.path.exists(filepath):
            print ('Ops File not exists..at {}'.format(filepath))
            sys.exit()
        else:
            with open(filepath,'r') as file_read:
                content = file_read.read().splitlines()
            count =  0

            # Loop over file contents
            for line in content:
                # print (line)
                
                if count < 2: # First two lines are the input (x) and label (y)
                    Name.append(line)

                if re.search('\w/log_prob/mul$',line):
                    label_dist.append(line)
                

                if self.FLAGS.NameScope: # Some models will use tf.name_scope to define these operations.
                    if re.search('\w/accuracy/value$',line):
                        acc_val.append(line) 

                    if re.search('\w/accuracy/update_op$',line):
                        acc_up.append(line)
                
                else:
                    if re.search('accuracy/value',line):
                        acc_val.append(line) 

                    if re.search('accuracy/update_op',line):
                        acc_up.append(line)
                
                count += 1
                    
        Op_Dicts = {'x': Name[0] , 'y': Name[1] , 'label_dist': label_dist[0] , 'accuracy_val': acc_val[0] , 'accuracy_up':acc_up [0] }
        return Op_Dicts


    def Load_Model(self):
        metafile = os.path.join(self.FLAGS.model_dir, self.FLAGS.model_ckpt )
        new_saver = tf.compat.v1.train.import_meta_graph(metafile)
        init_l = tf.local_variables_initializer()
        # Local variable initialization for running the tf.meteric.accuracy
        self.sess.run(init_l)
        
        # Restore the session from the checkpoint
        new_saver.restore(self.sess, tf.train.latest_checkpoint(self.FLAGS.model_dir))
        
        # Set the Model graph
        print ('Model loaded...!!!')
        graph = tf.compat.v1.get_default_graph()
        print ('Graph set loaded...!!!')
        
        return 

    def Calc_Global_SNR(self):
        
        return
    
