
import os
import sys
import re
import tensorflow as tf
import numpy as np
import time
import pickle 
from bprune.src.Viz_Plotting import Plot_Viz
tf.contrib.resampler 

class Prune_Model:
    def __init__(self, FLAGS, Sess, Images, Labels):
        self.FLAGS = FLAGS
        self.sess = Sess
        self.Images = Images
        self.Labels = Labels
        self.Op_Dicts = self.Read_OpsFile()
        self.Layer_Name = self.Read_LayerFile()

        if not os.path.exists(self.FLAGS.data_dir):
            os.makedirs(self.FLAGS.data_dir)
            


        #Initialize the class if horovod is used.
        if self.FLAGS.horovod_used:
            import horovod.tensorflow as hvd
            hvd.init()
        else:
            pass

        self.graph = self.Load_Model()
        self.Viz = Plot_Viz(self.FLAGS)
         
    
    
    def Read_LayerFile(self):
        Layer_Name = []

        tmp = self.FLAGS.model_dir.split('/')[-1]    
        if tmp =='/':
            filepath = os.path.join(os.path.dirname(os.path.dirname(self.FLAGS.model_dir)),'LayerNames.txt')
        else:
            dir_name = self.FLAGS.model_dir+'/'
            filepath = os.path.join(os.path.dirname(os.path.dirname(dir_name)),'LayerNames.txt')

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
                elif re.search("gamma:0$",tmp):
                    pass
                elif re.search("beta:0$",tmp):
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
        
        tmp = self.FLAGS.model_dir.split('/')[-1]    
        if tmp =='/':
            filepath = os.path.join(os.path.dirname(os.path.dirname(self.FLAGS.model_dir)),'Ops_name_BNN.txt')
        else:
            dir_name = self.FLAGS.model_dir+'/'
            filepath = os.path.join(os.path.dirname(os.path.dirname(dir_name)),'Ops_name_BNN.txt')

        
        # filepath = os.path.join(os.path.dirname(os.path.dirname(self.FLAGS.model_dir)),'Ops_name_BNN.txt')

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
                

                if self.FLAGS.name_scope: # Some models will use tf.name_scope to define these operations.
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

        x = self.graph.get_tensor_by_name( Var_Dict['x']+":0" )
        y = self.graph.get_tensor_by_name( Var_Dict['y']+":0" )

        # tf.get_default_graph()
        labels_distribution = self.graph.get_tensor_by_name( Var_Dict ['label_dist'] +":0") 

        Accuracy = self.graph.get_tensor_by_name( Var_Dict['accuracy_val'] + ":0")
        update_ops = self.graph.get_tensor_by_name( Var_Dict['accuracy_up'] + ":0")

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

    
    def Return_Mean_Sigma_All(self):
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
            _ , _ , Ratio_modify = self.Calc_Ratio_mu_sigma(layer)
            # Factor of 2 multiplied since each layer has 2 parameter (mu,sigma)
            zero_count = np.count_nonzero(Ratio_modify)*2.0
            Layer_nonZero.append( zero_count )
            count_nonzeros += (zero_count)

        return count_nonzeros, Layer_nonZero


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
    
    
    def Replace_Weights_Network(self,layer,New_weights_Mean,New_weights_Std,CHECK=False):
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
    
    def Calc_HeldOutLP_Acc(self,x,y,update_ops,Accuracy,labels_distribution,Non_zeros,Layer_nonZero):
        
        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: self.Images, y: self.Labels}

        #######################################################################################
        # Accuracy Calculations
        #######################################################################################
        for i in range(50):
            _ = self.sess.run(update_ops,feed_dict=feed_dict_testing)

        Acc = self.sess.run(Accuracy,feed_dict=feed_dict_testing)
        
        
        ######################################################################################
        ## Heldout-set probability calculation
        ######################################################################################
        start_time_infer = time.time()
        ## Define the p(y* |X,Y,\theta)
        probs = np.asarray([self.sess.run(tf.nn.softmax(labels_distribution),feed_dict=feed_dict_testing)
                                                for _ in range(self.FLAGS.num_monte_carlo)])
        end_time_infer = time.time()

        Infer_RunTime = (end_time_infer - start_time_infer)
        print ("{},{},{},{}\n".format(self.FLAGS.thres_val,Acc,Non_zeros,Infer_RunTime ))

        mean_probs = np.mean(probs, axis=0)
        heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]), self.Labels.flatten()]))
        
        print (" ... Held-out nats: {:.3f}".format(heldout_lp)+"\n")

        
        ########################################################################################
        # Save Calculations results
        #########################################################################################
        fname = os.path.join(self.FLAGS.data_dir, ("Run_InferenceMode_{}_Threshold_{}_MC_{}".format(self.FLAGS.inference,self.FLAGS.thres_val,self.FLAGS.num_monte_carlo) ) )
        with open(fname, "wb") as out:
            pickle.dump([self.Images,self.Labels,probs,heldout_lp,Acc,self.FLAGS.thres_val,Infer_RunTime,Non_zeros,Layer_nonZero], out)

        return





    def RunPrune(self):
        if self.FLAGS.inference:
            # No modification simiply run the 
            # Heldout Calculations
            try:
                assert(self.FLAGS.thres_val == 0.0)
                x,y,labels_distribution,Accuracy,update_ops = self.Load_Inference_Variable()
                Non_zeros_beforePrune,Layer_nonZero_beforePrune = self.Calc_NonZeros()
                self.Calc_HeldOutLP_Acc(x,y,update_ops,Accuracy,labels_distribution,Non_zeros_beforePrune,Layer_nonZero_beforePrune)
            except AssertionError:
                print ("Running in Inference Mode...Threshold value should be 0.0 given {}".format(self.FLAGS.thres_val))
                
                  
        elif self.FLAGS.prune:
            
            #
            # Pruning Run...... #
            #     

            x,y,labels_distribution,Accuracy,update_ops = self.Load_Inference_Variable()
            Non_zeros_beforePrune,Layer_nonZero_beforePrune = self.Calc_NonZeros()
    
            # Modify the weight and run 
            # Heldout Calculations
            init_Thres = self.FLAGS.thres_val
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
                self.Replace_Weights_Network(layer,Sparse_Conv_mean,Sparse_Conv_std)

                # Ratio after modifing Weights....
                _,_,Modify_Ratio = self.Calc_Ratio_mu_sigma(layer)
                Modified_Ratio_All.append( Modify_Ratio )
            
            Non_zeros,Layer_nonZero = self.Calc_NonZeros_V2(self.Layer_Name,Modified_Ratio_All)
            print ('% Prune {}'.format( 100.0 * (1 - (Non_zeros / Non_zeros_beforePrune) ) ) )
            
            self.Calc_HeldOutLP_Acc(x,y,update_ops,Accuracy,labels_distribution,Non_zeros,Layer_nonZero)
        
        else:
            print ('Method not defined...check the input arguments..!')
            sys.exit()
