
import os
import sys
import re
import tensorflow as tf
import numpy as np
from src.Viz_Plotting import *


class Prune_Model:
    def __init__(self, FLAGS, Sess):
        self.FLAGS = FLAGS
        self.sess = Sess
        self.Op_Dicts = self.Read_OpsFile()
        self.Layer_Name = self.Read_LayerFile()
        self.graph = self.Load_Model()
        self.Viz = Viz_Plotting(self.FLAGS) 
    
    
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
        
        return graph

    
    def Load_Inference_Variable(self):
        Var_Dict = self.Op_Dicts

        x = graph.get_tensor_by_name( Var_Dict['x']+":0" )
        y = graph.get_tensor_by_name( Var_Dict['y']+":0" )
  
        labels_distribution = tf.get_default_graph().get_tensor_by_name( Var_Dict ['label_dist'] +":0") 

        Accuracy = graph.get_tensor_by_name( Var_Dict['accuracy_val'] + ":0")
        update_ops = graph.get_tensor_by_name( Var_Dict['accuracy_up'] + ":0")

        return x,y,labels_distribution,Accuracy,update_ops


    def Calc_Ratio_mu_sigma(self,layer):
        mean_val = layer[0] #'dense_flipout'+'/kernel_posterior_loc:0'
        Std_val = layer[1] #'dense_flipout'+'/kernel_posterior_untransformed_scale:0'
        
        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        for var, val in zip(tvars,tvars_vals):
            if var.name == mean_val:
                Conv_Last_mean = val
            elif var.name == Std_val:
                Conv_Last_std_softplus = val
            else:
                pass

        std_soft = self.Eval_SoftPlus(Conv_Last_std_softplus.flatten())
        Ratio_mean_STD = abs(Conv_Last_mean.flatten())/ std_soft 
        
        return Conv_Last_mean, std_soft, Ratio_mean_STD

    def Calc_Global_SNR(self):
        Ratio_All = []
        for layer in self.Layer_Name:
            _,_,Ratio_mean_STD = self.Calc_Ratio_mu_sigma(layer)
            
            if self.FLAGS.Plotting:
                self.Viz.Plot_dist(Ratio_mean_STD,layer)

            Ratio_All.append(Ratio_mean_STD)

        return Ratio_All

    
    def Eval_SoftPlus(self, x): 
        # Softplus
        val = np.log( np.exp(x) + 1 ) 
        return val

    
    def Return_Mean_Sigma_All(self,sess,graph,LayerName_All,FLAGS):
        All_Layer_Mean_Std = []
        
        for layer in self.Layer_Name:
            Layer_Mean_Std = []
            mean, std, _ = self.Calc_Ratio_mu_sigma(layer)
            Layer_Mean_Std.append(mean); Layer_Mean_Std.append(std)

            if FLAGS.Plotting:
                self.Viz.Plot_mean_std(mean,std,layer)

            All_Layer_Mean_Std.append(Layer_Mean_Std)
        
        return All_Layer_Mean_Std
    
    def Calc_NonZeros(self):
        '''
        Useful when ratios for all the layers are not available
        '''
        N = len(self.Layer_Name) # 11 Layers in VGG
        i = 0
        count_nonzeros = 0
        Layer_nonZero = []

        for layer  in (self.Layer_Name):
            Ratio_modify = self.Calc_Ratio_mu_sigma(layer)
            # Factor of 2 multiplied since each layer has 2 parameter (mu,sigma)
            zero_count = np.count_nonzero(Ratio_modify)*2.0
            Layer_nonZero.append( zero_count )
            count_nonzeros += np.count_nonzero(zero_count)

        return count_nonzeros*2.0, Layer_nonZero


    def Calc_NonZeros_V2(self,Layer_Name,All_Ratio_mean_Std):
        '''
        This is useful when all the individual ratios are avalilabel for each layer. 
        '''

        N = len(Layer_Name) # 11 Layers in VGG
        i = 0
        count_nonzeros = 0
        Layer_nonZero = []
        
        for name,val in zip(Layer_Name,All_Ratio_mean_Std):
            # Ratio_modify = self.Calc_Ratio_mu_sigma(name)
            zero_count = np.count_nonzero(val) * 2.0
            Layer_nonZero.append( zero_count )
            count_nonzeros += zero_count

        return count_nonzeros, Layer_nonZero

        
    def Call_Weight_Writer(self,layer):
        mean_val = layer[0] # 'dense_flipout'+'/kernel_posterior_loc:0'
        std_val = layer[1] # 'dense_flipout'+'/kernel_posterior_untransformed_scale:0'

        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        Conv_mean = []
        Conv_std = []   
        for var, val in zip(tvars, tvars_vals):
            if var.name == mean_val:
                Conv_mean.append(val)
            elif var.name == std_val:
                Conv_std.append(val)
            else:
                pass
            
        return np.array(Conv_mean),np.array(Conv_std)
    
    
    def Replace_Weights_Network(self, Sess,layer,New_weights_Mean,New_weights_Std,CHECK=False):
        Mean_vals = layer[0] #'dense_flipout'+'/kernel_posterior_loc:0' #layer + '/kernel_posterior_loc:0'
        Std_vals =  layer[1] #'dense_flipout'+'/kernel_posterior_untransformed_scale:0' # layer + '/kernel_posterior_untransformed_scale:0'

        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        for var, val in zip(tvars,tvars_vals):
            if var.name == Mean_vals:
                assign_op = var.assign(New_weights_Mean)
            elif var.name == Std_vals:
                assign_op_2 = var.assign(New_weights_Std)
            else:
                pass
        
        # Run the assingment operation
        self.sess.run([assign_op,assign_op_2])
        
        if CHECK == True: # Check to see if the assign operation worked

            Conv_mean = []
            Conv_std = []
            tvars_vals_2 = Sess.run(tvars)
            for var, val in zip(tvars,tvars_vals_2):
                if var.name == Mean_vals:
                    Conv_mean.append(val)
                elif var.name == Std_vals:
                    Conv_std.append(val)
                else:
                    pass
            try:
                assert ((Conv_mean  == New_weights_Mean).all() )
                assert ((Conv_std  == New_weights_Std).all() )
            except AssertionError:
                print ('The modification of weights failed...!')

        return



    def Make_Conv_Sparse(self,Ratio_mean_STD,Weights_Mean,Weights_STD,Thres_val):
        '''
        Description:
        Based on the Ratio of mean and standard deviation for each layer
        the learning paramerter "kernel_loc" and "kernel_untransformed_loc" is 
        modified. The Threshold value is used to find index in the ratio_mean_std array.

        Inputs:
        
        Ratio_mean_STD: The ratio of mean and standard deviation of each layer.
        Weight_Mean: The "kernel_loc" which is needed to be prunned.
        Weight_STD: The "kernel_untransformed_scale" need to be prunned.
        Thres_val: The threshold value used for the finding prune index locations. 
        '''
        
        if Thres_val == 0.0:
            tmp = Ratio_mean_STD <= 1e-44 
        else:
            tmp = Ratio_mean_STD <= Thres_val

        rep_val = np.float32(0) 

        Replace_Conv_mean = np.where( tmp == True, rep_val,Weights_Mean.flatten())
        Replace_Conv_std = np.where( tmp == True, rep_val,Weights_STD.flatten())
        shape = Weights_Mean.shape 
        
        
        return tmp,Replace_Conv_mean.reshape(shape[1:]),Replace_Conv_std.reshape(shape[1:])

    def Calc_Ratio_mu_sigma_V2(self,Conv_Last_mean,Conv_Last_std):
        std_soft = self.Eval_SoftPlus(Conv_Last_std.flatten())
        Ratio_mean_STD = abs(Conv_Last_mean.flatten())/ std_soft 
        return Ratio_mean_STD
    





    def RunPrune(self):
        if self.FLAGS.Infer:
            # No modification simiply run the 
            # Heldout Calculations
            pass

        else:
            # Modify the weight and run 
            # Heldout Calculations
            init_Thres = self.FLAGS.Thres_val
            Index_All = []
            Modified_Ratio_All = []
            
            for i, layer in enumerate(self.Layer_Name):
                
                # Load the training parameters of last convolution layer
                Conv_Last_mean,Conv_Last_std = self.Call_Weight_Writer(layer)
                

                Ratio_mean_STD = self.Calc_Ratio_mu_sigma_V2(Conv_Last_mean,Conv_Last_std)
               
                
                # Prune the weights and make them more sparse...   
                Index, Sparse_Conv_mean,Sparse_Conv_std = self.Make_Conv_Sparse(Ratio_mean_STD,Conv_Last_mean,Conv_Last_std,init_Thres)

                Index_All.append(Index)

                # Change the weights in the network....!
                Replace_Weights_Network(sess,layer,Sparse_Conv_mean,Sparse_Conv_std)

                # Ratio after modifing Weights....
                Modified_Ratio_All.append( self.Calc_Ratio_mu_sigma(layer) )
            
            Non_zeros,Layer_nonZero = self.Calc_NonZeros_V2(self.Layer_Name,Modified_Ratio_All)
