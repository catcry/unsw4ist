# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:12:06 2019
____________________________________________________________
This is Convolutional Neural Network Implementation (LeNet5)
It can be run after prep_X_files where the train, validation and Test sets are 
defined.
> Hyperparameters have to be set manually
___________________________________________________________

@author: catcry
"""
Xtrain = Xtrain_orig
Xvalid = Xvalid_orig
Xtest = Xtest_orig

num_classes = 1
#==============================================================================
#==========================>   CONV Implementation  <==========================
#==============================================================================
nn_type = '1D Convulotional'
every_item_repeat = 1
#num_filters = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200])
num_filters = np.array ([64])
num_filters = np.tile(num_filters,every_item_repeat)
repeat_no = len(num_filters)

kern_size = 3
step_size = 1
dilation_size = 1

padd = 'same'
data_form = 'channels_last' # (batch, steps, channels)
act_funct = 'relu'#keras.layers.LeakyReLU(alpha=0.3)#'relu'
auxi_act_funct = 'relu' #keras.layers.LeakyReLU(alpha=0.3)  #for other dense layers
output_act = 'sigmoid' #keras.layers.LeakyReLU(alpha=0.3)#'softmax'

loss_funct = 'binary_crossentropy'#'mse'#
batch = 20240
num_epochs = 300
cls_weights = None
#cls_weights = {0:1,1:5, 2:1,3:640,4:1290} #for NSL-KDD Multi class

#cls_weights = compute_class_weight('balanced',np.unique(Ytrain),Ytrain)

#-------------------------------------------
#Optimizer

opt_type = 'Adam'
learning_rate = 0.002
decay_rate = learning_rate/num_epochs
moment = 0.8

#from tensorflow.compat.v1.keras.callbacks import CSVLogger
#globals()['csv_logger_U'+str(num_layers[0])] = CSVLogger('log.csv', append=True, separator=';')

# >> Optimizer Type
sgd = optimizers.SGD(lr=learning_rate,\
                     decay = decay_rate, \
                     momentum = moment, \
                     nesterov=True)
#------------------------------------------------------------------------------
rms = optimizers.RMSprop(lr = learning_rate, \
                         rho= 0.9, \
                         epsilon = None,\
                         decay = decay_rate)
#------------------------------------------------------------------------------
adagrad = optimizers.Adagrad (lr = learning_rate , \
                              epsilon = None , \
                              decay = decay_rate)

#------------------------------------------------------------------------------
adadelta = optimizers.Adadelta(lr = learning_rate, \
                         rho=0.95 , \
                         epsilon = None,\
                         decay = decay_rate)

#------------------------------------------------------------------------------
adam = optimizers.Adam(lr = learning_rate, \
                         beta_1 = 0.9 , \
                         beta_2 = 0.999 , \
                         epsilon = None,\
                         decay = decay_rate,\
                         amsgrad = True )

nadam = optimizers.Nadam(lr = learning_rate, \
                         beta_1 = 0.9, \
                         beta_2 = 0.999, \
                         epsilon = None, \
                         schedule_decay = 0.004)
#_________________________>>   End of  Optimizers    <<________________________

t_start_c = np.zeros(repeat_no)
t_end_c = np.zeros(repeat_no)

ac_train_c  = np.zeros(repeat_no)
ac_train_c_mean_intime = np.zeros(repeat_no+1)

ac_valid_c = np.zeros(repeat_no)
ac_valid_c_mean_intime = np.zeros(repeat_no+1)

ac_test_c = np.zeros(repeat_no)
ac_test_c_mean_intime = np.zeros(repeat_no+1)

ac_test_21_c = np.zeros(repeat_no)
ac_test_21_c_mean_intime = np.zeros(repeat_no+1)
#------------------------------------------------------------------------------

i=0 
while (i < repeat_no):
    
#++++++++++++++++++++++++>>>     Model Archi       <<<+++++++++++++++++++++++++
    
        t_start_c[i] = time.time()
        model_c = Sequential()

        # model_c.add(Dense(5,activation = auxi_act_funct,input_shape = (None,Xtrain.shape[2])))
#>>>>>  Layer 1 : Conv Layer---------------------------------------------------       
        model_c.add(Conv1D(filters = num_filters[i], \
                           kernel_size = kern_size,\
                           strides = step_size,\
                           padding = padd,\
                           data_format = data_form,\
                           dilation_rate = 1,\
                           activation = act_funct,\
                           use_bias = True,\
#                           kernel_initializer = 'glorot_uniform',\
#                           bias_initializer = 'ones',\
                          kernel_regularizer = keras.regularizers.l2(0.),\
                          bias_regularizer = None,\
                          activity_regularizer = None,\
#                           kernel_constraint = None,\
#                           bias_constraint = None,\
                           input_shape = (Xtrain_orig.shape[1],\
                                          Xtrain_orig.shape[2])\
                           ))
#>>>>>   Layer 2 : Dense Layer-------------------------------------------------
##
#        model_c.add(Dense(100,activation = auxi_act_funct))       
##        
        model_c.add (MaxPooling1D(pool_size = 3, \
                                  strides=None, \
                                  padding='same', \
                                  data_format=data_form,\
                                 ))
##>>>>>   Layer 3 : Conv Layer2 ------------------------------------------------      
        
        
#        
#         model_c.add(Conv1D(filters = 50, \
#                           kernel_size = kern_size,\
#                           strides = step_size,\
#                           padding = padd,\
#                           data_format = data_form,\
#                           dilation_rate = 1,\
#                           activation = act_funct,\
#                           use_bias = True,\
#                           kernel_initializer = 'glorot_uniform',\
# #                           bias_initializer = 'ones',\
# #                           kernel_regularizer = keras.regularizers.l2(0.),\
# #                           bias_regularizer = None,\
# #                           activity_regularizer = None,\
# #                           kernel_constraint = None,\
#                           bias_constraint = None,\
#                           ))
        
# # #>>>>>   Layer 4 : Aveg Layer2 ------------------------------------------------       
        
#         model_c.add (MaxPooling1D(pool_size = 3 , \
#                                   strides=None, \
#                                   padding='same', \
#                                   data_format = data_form,\
#                                   ))
#>>>>>   Layer 5 : Dense Layer2 -----------------------------------------------        
#        model_c.add(Dense(30,activation = auxi_act_funct))
#        model_c.add (MaxPooling1D(pool_size = 3, \
#                                  strides=None, \
#                                  padding='valid', \
#                                  data_format='channels_first',\
#                                 ))
        # model_c.add(Dense(120,activation = auxi_act_funct))
        # model_c.add(Dense(84,activation = auxi_act_funct))
#         model_c.add(Conv1D(filters = num_filters[i], \
#                            kernel_size = kern_size,\
#                            strides = step_size,\
#                            padding = padd,\
#                            data_format = data_form,\
#                            dilation_rate = 1,\
#                            activation = act_funct,\
#                            use_bias = True,\
# #                           kernel_initializer = 'glorot_uniform',\
#                            bias_initializer = 'ones',\
#                            # kernel_regularizer = keras.regularizers.l2(0.),\
#                            bias_regularizer = None,\
#                            activity_regularizer = None,\
#                            kernel_constraint = None,\
#                            bias_constraint = None,\
                                                      # ))
        model_c.add (MaxPooling1D(pool_size = 3 , \
                                  strides=None, \
                                  padding='same', \
                                  data_format = data_form,\
                                  ))
#        model_c.add (MaxPooling1D(pool_size = 3 , \
#                              strides=None, \
#                              padding='same', \
#                              data_format = data_form,\
#                             ))
        # model_c.add(Flatten())
        model_c.add(Dense(80,activation = auxi_act_funct))
        model_c.add(Dense(64,activation = auxi_act_funct))
#>>>>>   Layer 4 : output layer -----------------------------------------------
        model_c.add(Dense(num_classes, activation = output_act))
        
        
        
        model_c.compile (loss=loss_funct , optimizer = adam , metrics=['mae', 'acc'])
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',\
                                                        min_delta = 0 ,\
                                                        patience = 3,\
                                                        mode  = 'min')
        history_c = model_c.fit (Xtrain_orig,Ytrain_1hot, \
                     epochs = num_epochs, \
                     batch_size = batch, \
                     class_weight = cls_weights,\
                     validation_split = 0.0,\
                     validation_data = (Xvalid_orig,Yvalid_1hot) ,\
                     verbose = 1, \
                     shuffle = True
#                     callbacks=[#globals()['csv_logger_U'+str(num_filters[0])],\
#                             callback]
                    )
                             
        #----------------------------------------------------------------------
        t_end_c[i] = time.time()
        
        
        #++++++++++++++++++++.>>>       weight Get    <<<.+++++++++++++++++++++
        
        def weight_get(model,i):
                    layer = model[i]
                    return layer.get_weights()
                
#        globals()['dense100_layer_weights_'+str(i)] = weight_get(model_c.layers,0)
#        globals()['dense80_layer_weights_'+str(i)] = weight_get(model_c.layers,1)
#        globals()['dense150_layer_weights_'+str(i)] = weight_get(model_c.layers,2)
        globals()['conv1st_layer_weights_'+str(i)] = weight_get(model_c.layers,0)
        globals()['maxpool1_layer_weights_'+str(i)] = weight_get(model_c.layers,1)
        globals()['maxpool2_layer_weights_'+str(i)] = weight_get(model_c.layers,2)
#        globals()['avgpool_layer_weights_'+str(i)] = weight_get(model_c.layers,3)
#        globals()['dense100_layer_weights_'+str(i)] = weight_get(model_c.layers,1)
        globals()['dense84_layer_weights_'+str(i)] = weight_get(model_c.layers,3)
        globals()['output_layer_weights_'+str(i)] = weight_get(model_c.layers,4)
#        
            
            
        
                        
         
        
        
        #____________________________>   FOR 2-Class Mode    <_________________________
        #==============================================================================
        #==============================Getting Results Ready===========================
        #==============================================================================
                     
        #ts_result = time.time()
        yh_Xtrain_c = model_c.predict(Xtrain_orig)
        
        if num_classes == 1 :
            yh_Xtrain_c = np.around(yh_Xtrain_c,decimals=0).astype(np.int)
        else: 
            yh_Xtrain_c = yh_Xtrain_c.argmax(axis = 1)
           
        yh_Xtrain_c = yh_Xtrain_c.reshape(len(yh_Xtrain_c))
           
        train_conf_matrix_c = confusion_matrix (Ytrain,yh_Xtrain_c)
        ac_train_c[i] = accuracy_score(Ytrain_1hot,yh_Xtrain_c)
        
        # ac_train_c_mean_intime[i+1] = (sum(ac_train_c_mean_intime) + ac_train_c[i] ) / (i+1)        
        # #______________________________________________________________________ 
        
        yh_valid_c = model_c.predict(Xvalid_orig)
        
        if num_classes == 1:
            yh_valid_c = np.around(yh_valid_c,decimals=0).astype(np.int)
        else : 
            yh_test_c = yh_valid_c.argmax(axis = 1)
            
        yh_valid_c = yh_valid_c.reshape(len(yh_valid_c))   
        
        valid_con_matrix_c = confusion_matrix (Yvalid , yh_valid_c)
        ac_valid_c[i] = accuracy_score(Yvalid , yh_valid_c)
        
        # ac_test_c_mean_intime[i+1] = (sum(ac_valid_c_mean_intime) + ac_valid_c[i]) / (i+1)
           #______________________________________________________________________ 
        yh_test_c = model_c.predict(Xtest_orig)
        
        if num_classes == 1:
            yh_test_c = np.around(yh_test_c,decimals=0).astype(np.int)
        else : 
            yh_test_c = yh_Xtrain_c.argmax(axis = 1)
            
        yh_test_c = yh_test_c.reshape(len(yh_test_c)) 
        
        #________________>>      Generating Error Vector    <<_________________
        # globals()['yerror_testU'+str(num_filters[0])+'it_no'+str(i)] = Ytest - yh_test_c
        
           
        test_con_matrix_c = confusion_matrix (Ytest , yh_test_c)
        ac_test_c[i] = accuracy_score(Ytest , yh_test_c)
        
        # ac_test_c_mean_intime[i+1] = (sum(ac_test_c_mean_intime) + ac_test_c[i]) / (i+1)
        
        #______________________________________________________________________
        
        
        
        #yh_hard = model_g.predict(Xhard)
        #yh_hard = np.around(yh_hard,decimals=0).astype(np.int)
        #hardtest_con_matrix =confusion_matrix (Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
        #ac_hard_test = accuracy_score(Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
        
          
        
        # yh_test_21_c = model_c.predict(Xtest_21)
        
        # if num_classes == 1:
        #     yh_test_21_c  = np.around(yh_test_21_c,decimals=0).astype(np.int)
        # else:
        #     yh_test_21_c = yh_test_21_c.argmax(axis=1)
            
        # yh_test_21_c = yh_test_21_c.reshape(len(yh_test_21_c)) 
        # ac_test_21_c[i] = accuracy_score(Ytest_21 , yh_test_21_c)
        
        #________________>>      Generating Error Vector    <<_________________
        # globals()['yerror_test_21'+str(num_filters[0])+str(i)] = Ytest_21 - yh_test_21_c
        
        
        # test_21_con_matrix_g =confusion_matrix (Ytest_21 , yh_test_21_c)
        # ac_test_21_c[i] = accuracy_score(Ytest_21 , yh_test_21_c)
        
        # ac_test_21_c_mean_intime[i+1] = ( sum(ac_test_21_c_mean_intime) + ac_test_21_c[i] )/(i+1)
        #______________________________________________________________________
           
        #----------------------------------------------------------------------
        
        
        print('------------------------------------------------------------------------')
        
        print("The Specification of Classifier is:")
        print('------------------------------------------------------------------------')
        print ("Layers Config : ")
        print ("Neural Network Type : ", nn_type)
        print ("No. 1st layer filters: ",num_filters [i], \
#               '|| No. of 2nd Layer  filters:', num_2nd_filter,\
               "|| No. of Classes: ",num_classes)
        print ("Kernel Size = ",kern_size, \
               "|| stride = ", step_size,\
               "|| Padding : ", padd)
        print('------------------------------------------------------------------------')
        print ("Activtion Function: ",act_funct)
        print ("Output Layer Activtion Function: ",output_act)
        #print('-----------------------------------------------------------------------')
        print ("Loss Function : " , loss_funct)
        print ("No. of Epochs : " , num_epochs)
        print('------------------------------------------------------------------------')
        print("Optimizer Config : ")
        print("Type :", opt_type ," || Learning Rate: ",learning_rate,"||  Decay : " , decay_rate )
        print('------------------------------------------------------------------------')
        print("Learning Duration : ", round((t_end_c[i] - t_start_c[i]),2),' (secs)',"~ = ",\
              round((t_end_c[i] - t_start_c[i])/60),"(mins)")
        print('------------------------------------------------------------------------')
        #print("Getting Results  : %0.2f" % round(te_result - ts_result,2),' (sec)')
        print() 
        #----------------------------------------------------------------------
        print('----------------->>>     ', i, '     <<<-----------------------')
        
        
        print('========================================================================')
        print("Results:")   
        print('------------------------------------------------------------------------')
        print ('TrainSet Acc = ',round(ac_train_c[i]*100,3),'%')
        print('')
        print('-----------------------') 
        print ('ValidSet Acc = ',round(ac_valid_c[i]*100,3),'%')
        print('')
        print('-----------------------') 
        print('TestSet Acc = ',round(ac_test_c[i]*100,3),'%')
        print('')
        print('-----------------------')
        # print('Test-21 Set Acc = ',round(ac_test_21_c[i]*100,3),'%')
        # print('')
        # print('-----------------------')
        # print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
        # print('')
        
        # #----------------
        # if (ac_test_21_c[i] > 0.62):
        #     i = len(num_filters)
            
            
        # plt.figure()
        # plt.grid(b=True,which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
        # plt.minorticks_on()
        # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.8)
        # plt.plot(history_c.history['acc'])
        # plt.plot(history_c.history['val_acc'])

        # plt.figure()
        # plt.grid(which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
        # plt.minorticks_on()
        # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.8)
        # plt.plot(history_c.history['mae'])
        # plt.plot(history_c.history['val_mae'])    
        
        get_layer_output = k.function([model_c.layers[0].input],\
                                              [model_c.layers[0].output])
                
        Xtrain_new = get_layer_output(Xtrain_orig)
        Xvalid_new = get_layer_output(Xvalid_orig)
        Xtest_new = get_layer_output(Xtest_orig)
        
        globals()['Xtrain'+str(i)] = Xtrain_new[0]
        globals()['Xvalid'+str(i)] = Xvalid_new[0]
        globals()['Xtest'+str(i)] = Xtest_new[0]
        
        i+=1
        #----------------
        #___________________>>     End of result printing      <<______________
        
        

 
#------------------------------------------------------------------
#+++++++++++++++++++>>       Mean & STD       <<+++++++++++++++++++
#------------------------------------------------------------------
# globals()['ac_train_mean_U'+str(num_filters[0])] = np.mean(ac_train_c*100)
# globals()['ac_train_std_U'+str(num_filters[0])] = np.std(ac_train_c*100)
# globals()['ac_train_min_U'+str(num_filters[0])] = np.min(ac_train_c*100)
# globals()['ac_train_max_U'+str(num_filters[0])] = np.max(ac_train_c*100)

# globals()['ac_valid_mean_U'+str(num_filters[0])] = np.mean(ac_valid_c*100)
# globals()['ac_valid_std_U'+str(num_filters[0])] = np.std(ac_valid_c*100)
# globals()['ac_valid_min_U'+str(num_filters[0])] = np.min(ac_valid_c*100)
# globals()['ac_valid_max_U'+str(num_filters[0])] = np.max(ac_valid_c*100)

# globals()['ac_test_mean_U'+str(num_filters[0])] = np.mean(ac_test_c*100)
# globals()['ac_test_std_U'+str(num_filters[0])] = np.std(ac_test_c*100)
# globals()['ac_test_min_U'+str(num_filters[0])] = np.min(ac_test_c*100)
# globals()['ac_test_max_U'+str(num_filters[0])] = np.max(ac_test_c*100)

# globals()['ac_test_21_mean_U'+str(num_filters[0])] = np.mean(ac_test_21_c*100)
# globals()['ac_test_21_std_U'+str(num_filters[0])] = np.std(ac_test_21_c*100)
# globals()['ac_test_21_min_U'+str(num_filters[0])] = np.min(ac_test_21_c*100)
# globals()['ac_test_21_max_U'+str(num_filters[0])] = np.max(ac_test_21_c*100)

#-------------------------------------------------------------------
#--------------------->>    Print Mean and Var ---------------------

# print('---------------->>   Mean And Variance   <<-----------------')

# print('Train Set Mean   = ',\
#       round(globals()['ac_train_mean_U'+str(num_filters[0])] ,3),\
#       ' ||  Train Set Std Dev.   = ', \
#       round(globals()['ac_train_std_U'+str(num_filters[0])],3 ))
# print()
# print('Valid Set Mean   = ',\
#       round( globals()['ac_valid_mean_U'+str(num_filters[0])] ,3),\
#       ' ||  Valid Set Std.Dev.   = ', \
#       round(globals()['ac_valid_std_U'+str(num_filters[0])],3) )
# print()
# print('Test Set Mean    = ',\
#       round( globals()['ac_test_mean_U'+str(num_filters[0])] ,3),\
#       ' ||  Test Set Std. Dev.    = ', \
#       round(globals()['ac_test_std_U'+str(num_filters[0])],3) )
# print()
# print('Test-21 Set Mean = ', \
#       round(globals()['ac_test_21_mean_U'+str(num_filters[0])],3),\
#       '   ||  Test-21 Set Std. Dev. = ',\
#       round(globals()['ac_test_21_std_U'+str(num_filters[0])],3) )
# print('Test-21 Set: Min = ', round(np.min(ac_test_21_c*100),3)\
#       ,'  ||  Max = ', round(np.max(ac_test_21_c*100),3))
#______________________________________________________________________________
#___________________>>   End of GRU Implementation      <<_____________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
 
 
 


 
#==============================================================================
#=========================>>>    Plot per Epoch    <<<=========================
#==============================================================================
#plt.plot(history.history['val_acc'])
 
 
 
 
 
 
 

