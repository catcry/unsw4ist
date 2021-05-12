# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:13:37 2019

@author: catcry

_________________________________________________________________________
                >>   RNN-based Decoder   <<   
_________________________________________________________________________
                
"""
Xtrain_compact = Xtrain0
Xvalid_compact = Xvalid0
Xtest_compact = Xtest0
#==============================================================================
#==========================>   Windowing Data   <==============================
#==============================================================================
#def data_win(data,time_step,time_win):
#    x = data[0:time_win,:]
#    i = time_step
#    while i<= len(data)-time_win:
#        x = np.concatenate((x,data[i:i+time_win,:]),axis=0)
#        i = i + time_step
#    x=x.reshape(i,time_win,data.shape[1])
#    return (x,i)
#
# def data_windowing (data,labels,time_step,time_win):
#     i=time_step
#     while i<time_win:
#       globals()['data_shift'+str(i)] = data[i:,:]
#       i=i+time_step
    
#     data = data[0:-time_win+1]
#     n=time_step
#     while n<i-time_step:
#         globals()['data_shift'+str(n)] = globals()['data_shift'+str(n)][0:-time_win+n+1]
#         data = np.dstack((data,globals()['data_shift'+str(n)]))
#         n=n+time_step
#     data=np.dstack((data,globals()['data_shift'+str(n)]))
#     data=data.transpose(0,2,1)
#     labels = labels[time_win-1:]
#     return data,labels
#
#time_step = 1
#time_win = 100
#a=time.time()
#Xtrain = data_windowing(data = Xtrain, time_step = time_step, time_win = time_win)
#b=time.time()
#Xvalid = data_windowing(data = Xvalid, time_step = time_step, time_win = time_win)
#a=time.time()
#Xtest=   data_windowing(data = Xtest, time_step = time_step, time_win = time_win)
#b=time.time()


#==============================================================================
#==========================>   GRU Implementation  <===========================
#==============================================================================


#+++++++++++++++++++++++>>    Log the loss Values     <<+++++++++++++++++++++++ 
        
#class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#        self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        

num_classes = 42
#++++++++++++++++++++++++>  RNN Learning Parameters    <+++++++++++++++++++++++nn_type = 'GRU'
nn_type = 'GRU'
allover_repeat = 1
#num_layers = np.array([1,2,3,4,5,61,7,8,9,10,20,30,40,50,60,70,80,90,100,200])
num_layers = np.array([300])
num_layers = np.repeat(num_layers,allover_repeat)
num_2nd_layer = 80
#-----------------------------------------
act_funct = 'elu' #keras.layers.LeakyReLU(alpha=0.4)#'relu' # #keras.activations.exponential# k
second_act_funct = 'elu'#'relu'
output_act_funct = 'sigmoid'
loss_funct = 'mse' 
num_epochs = 200
#-------------------------------------------
batch =20480
#cls_weights = {0:4500,1:1, 2:1,3:20,4:25600}
#cls_weights = {0:1,1:5, 2:1,3:640,4:1290} #for NSL-KDD Multi class

#cls_weights = compute_class_weight('balanced',np.unique(Ytrain),Ytrain)
#cls_weights = None

#-------------------------------------------
#Optimizer

opt_type = 'adam'

learning_rate = 0.0001
#decay_steps = 0#10000000
decay_rate = (learning_rate)/num_epochs

#learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(ini_learning_rate,\
#        decay_steps = decay_steps,decay_rate =decay_rate)

moment = 0.8

from tensorflow.compat.v1.keras.callbacks import CSVLogger
globals()['csv_logger_U'+str(num_layers[0])] = CSVLogger('log.csv', append=True, separator=';')

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



#+++++++++++++++++>>  Running GRU Training for Several Times    <<+++++++++++++ 
repeat_no = len(num_layers)

t_start_g = np.zeros(repeat_no)
t_end_g = np.zeros(repeat_no)

ac_train_g  = np.zeros(repeat_no)
ac_train_g_mean_intime = np.zeros(repeat_no+1)

ac_valid_g = np.zeros(repeat_no)
ac_valid_g_mean_intime = np.zeros(repeat_no+1)

ac_test_g = np.zeros(repeat_no)
dr_test_g = np.zeros(repeat_no)
pr_test_g = np.zeros(repeat_no)
fpr_test_g = np.zeros(repeat_no)
ac_test_g_mean_intime = np.zeros(repeat_no+1)

#ac_test_21_g = np.zeros(repeat_no)
#ac_test_21_g_mean_intime = np.zeros(repeat_no+1)

i = 0
     
while i < repeat_no :
                
            #+++++++++++++++++++++++>    RNN Model Making    <+++++++++++++++++++++++++++++
            
            t_start_g[i] = time.time()
            model_g= Sequential()
            # model_g.add(Dense(5,activation = 'elu',\
            #                   input_shape = (Xtrain.shape[1],Xtrain.shape[2])))
            # model_g.add(Embedding(input_dim = Xtrain.shape[0],\
            #                     output_dim = 100,\
            #                     input_length = 40))
            model_g.add(GRU(num_layers[i], \
                            recurrent_activation = None, \
                            use_bias = True ,\
                            activation = act_funct,\
                            unroll= True,\
                            implementation = 1,\
##                           kernel_initializer = 'he_uniform',#keras.initializers.Ones(),\
##                           recurrent_initializer = keras.initializers.Ones(),\
##                           bias_initializer = keras.initializers.Ones(),\
                            return_sequences = True,\
##                           kernel_regularizer=regularizers.l2(0.1),\
##                           activity_regularizer=regularizers.l2(0.1),\
##                           recurrent_regularizer=regularizers.l2(0.1), \
##                           bias_regularizer=regularizers.l2(0.1),\
                            input_shape = (Xtrain_compact.shape[1], \
                                           Xtrain_compact.shape[2]),\
                            ))
            #
            
            
         
            
#            model_g.add(CuDNNLSTM(num_layers[i], \
#                           kernel_initializer='glorot_uniform',\
#                           recurrent_initializer='orthogonal',\
#                           bias_initializer='zeros',\
#                           unit_forget_bias=True, \
#                           kernel_constraint=None, \
#                           recurrent_constraint=None, \
#                           bias_constraint=None, \
#                           return_sequences=False,\
#                           return_state=False, \
#                           stateful=False,\
##                           use_bias = True ,\
##                           activation = act_funct,\
#                           unroll= True,\
#                           implementation = 1,\
##                           kernel_initializer = keras.initializers.Ones(),\
##                           recurrent_initializer = keras.initializers.Ones(),\
##                           bias_initializer = keras.initializers.Ones(),\
#                           kernel_regularizer=regularizers.l2(0.1),\
#                           activity_regularizer=regularizers.l2(0.1),\
#                           recurrent_regularizer=regularizers.l2(0.1), \
#                           bias_regularizer=regularizers.l2(0.1),\
#                           input_shape = (Xtrain.shape[1], Xtrain.shape[2])
#                           ))
##            #
            
            
            
            # # #
            # model_g.add(GRU(80, \
            #               recurrent_activation= None, \
            #               use_bias = True ,\
            #               activation = act_funct,\
            #               unroll= True,\
            #               implementation=1,\
            #               return_sequences=True,\
            #               ))
#            
            #
            #
            #
#            model_g.add(LSTM(300, \
#                           recurrent_activation= None, \
#                           use_bias = True ,\
#                           activation = act_funct,\
#                           unroll= True,\
#                           implementation=1,\
#                           ))
#            
            
            #model.add(LSTM(num_layers,return_sequences=True, \
            #               activation='tanh',\
            #               unroll=True,\
            #               input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
            
            #model.add(LSTM(num_layers, return_sequences=True,\
            #               activation='tanh',\
            #               unroll=True))
            ###
#            model_g.add(Dense(300,activation = second_act_funct))
#            model_g.add(Dense(200,activation = second_act_funct))
#            model_g.add(Dense(200,activation = second_act_funct, kernel_initializer = 'he_uniform'))
#            model_g.add(Dense(100,activation = second_act_funct, kernel_initializer = 'he_uniform'))
#            model_g.add(Dense(200,activation = act_funct))
#            model_g.add(Dense(120,activation = act_funct))
##            model.add(Dense(300,activation = act_funct))
            model_g.add(Dense(100,activation = second_act_funct))
            model_g.add(Dense(90,activation = second_act_funct))
            model_g.add(Dense(80,activation = second_act_funct)) #, #kernel_initializer = 'he_uniform'
            model_g.add(Dense(70,activation = second_act_funct))
            model_g.add(Dense(60,activation = second_act_funct))
            model_g.add(Dense(55,activation = second_act_funct))
            model_g.add(Dense(50,activation = second_act_funct))
            # model_g.add(Dense(10,activation = second_act_funct))
#            model_g.add(Dense(num_2nd_layer,activation = second_act_funct))
#            model_g.add(Dense(num_2nd_layer,activation = second_act_funct))
#            model_g.add(Dense(num_2nd_layer,activation = second_act_funct))
#          
#            model_g.add(Dense(num_2nd_layer,activation = act_funct))
#            
          
            model_g.add(Dense(num_classes, activation = output_act_funct ))
            
            model_g.compile(loss=loss_funct , optimizer = adam , \
            metrics=['mse', 'acc'])
           
            
            #loss_Spec = "model Loss = mae , Optimizer = sgd"
            
            
            #+++++++++++++++++>       RNN Training and Validation    <+++++++++++++++++++++
            
#            history_g = LossHistory()
            
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',\
                                                        min_delta = 0 ,\
                                                        patience = 3,\
                                                        mode  = 'min')
            history_g = model_g.fit (Xtrain_compact,Xtrain_orig, \
                                 epochs = num_epochs, \
                                 batch_size = batch, \
#                                 class_weight = cls_weights,\
                                 validation_split = 0.0,\
                                 validation_data = (Xvalid_compact,Xvalid_orig) ,\
#                                 validation_data = (Xtest_21,Ytest_21_1hot),\
                                 verbose = 1, \
                                 shuffle = True,\
                                 # callbacks=[globals()['csv_logger_U'+str(num_layers[0])],\
                                 #            callback]\
                                            )
            
             #(Xvalid,Yvalid_1hot),\
             
             
            t_end_g[i] = time.time()
            
            
            #------------------------------------------------------------------
            #+++++++++++++++++++>>     Getting Weights     <<++++++++++++++++++
            #------------------------------------------------------------------
#            def weight_get(model,i):
#                    layer = model[i]
#                    return layer.get_weights()
#                
#            globals()['gru_layer_weights_'+str(i)] = weight_get(model_g.layers,0)
#            globals()['dense1_layer_weights_'+str(i)] = weight_get(model_g.layers,1)
#            globals()['dense1_layer_weights_'+str(i)] = weight_get(model_g.layers,2)
#            globals()['dense1_layer_weights_'+str(i)] = weight_get(model_g.layers,3)
#            globals()['dense1_layer_weights_'+str(i)] = weight_get(model_g.layers,4)
#            globals()['output_layer_weights_'+str(i)] = weight_get(model_g.layers,5)
#            
            
            
            
            #--------------------------------------------------------------------------------------
            
            
            
            
                        
#             #%%
#             #____________________________>   FOR 2-Class Mode    <_____________
#             #==================================================================
#             #==============================Getting Results Ready===============
#             #==================================================================
            
#             #ts_result = time.time()
            yh_Xtrain_g = model_g.predict(Xtrain_compact)
#             # yh_Xtrain_g = yh_Xtrain_g.reshape(len(yh_Xtrain_g))
            
#             # if num_classes == 1 :
#             #     yh_Xtrain_g = np.around(yh_Xtrain_g,decimals=0).astype(np.int)
#             # else: 
#             #     yh_Xtrain_g = yh_Xtrain_g.argmax(axis = 1)
               
               
#             train_conf_matrix_g = confusion_matrix (Xtrain_orig,yh_Xtrain_g)
            
#             ac_train_g[i] = accuracy_score(Xtrain_orig,yh_Xtrain_g)
#             ac_train_g_mean_intime[i+1] = (sum(ac_train_g_mean_intime) + ac_train_g[i] ) / (i+1)        
#             #______________________________________________________________________ 
            
            yh_valid_g = model_g.predict(Xvalid_compact)
#             # yh_valid_g = yh_valid_g.reshape(len(yh_valid_g))
            
#             # if num_classes == 1:
#             #     yh_valid_g = np.around(yh_valid_g,decimals=0).astype(np.int)
#             # else : 
#             #     yh_valid_g = yh_valid_g.argmax(axis = 1)
                
                
#             valid_conf_matrix_g = confusion_matrix (Xvalid_orig , yh_valid_g)
#             ac_valid_g[i] = accuracy_score(Xvalid_orig , yh_valid_g)
            
#             ac_test_g_mean_intime[i+1] = (sum(ac_valid_g_mean_intime) + ac_valid_g[i]) / (i+1)
#            #______________________________________________________________________ 
            yh_test_g = model_g.predict(Xtest_compact)
#             # yh_test_g = yh_test_g.reshape(len(yh_test_g))
            
#             # if num_classes == 1:
#             #     yh_test_g = np.around(yh_test_g,decimals=0).astype(np.int)
#             # else : 
#             #     yh_test_g = yh_test_g.argmax(axis = 1)
                
                
#             test_conf_matrix_g = confusion_matrix (Xtest_orig , yh_test_g)
#             dr_test_g[i] = test_conf_matrix_g[1,1]/(test_conf_matrix_g[1,1]+\
#                                                    test_conf_matrix_g[1,0])
#             pr_test_g[i] = test_conf_matrix_g[1,1]/(test_conf_matrix_g[1,1]+\
#                                                    test_conf_matrix_g[0,1])
#             fpr_test_g[i] = test_conf_matrix_g[0,1]/(test_conf_matrix_g[0,0]+\
#                                              test_conf_matrix_g[0,1])
#             ac_test_g[i] = accuracy_score(Xvalid_orig , yh_test_g)
            
            
#             ac_test_g_mean_intime[i+1] = (sum(ac_test_g_mean_intime) + ac_test_g[i]) / (i+1)
#             #______________________________________________________________________
            
            
            
#             #yh_hard = model_g.predict(Xhard)
#             #yh_hard = np.around(yh_hard,decimals=0).astype(np.int)
#             #hardtest_con_matrix =confusion_matrix (Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
#             #ac_hard_test = accuracy_score(Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
            
          
            
# #            yh_test_21_g = model_g.predict(Xtest_21)
# #            
# #            if num_classes == 1:
# #                yh_test_21_g  = np.around(yh_test_21_g,decimals=0).astype(np.int)
# #            else:
# #                yh_test_21_g = yh_test_21_g.argmax(axis=1)
# #                
# #            test_21_con_matrix_g =confusion_matrix (Ytest_21 , yh_test_21_g)
# #            ac_test_21_g[i] = accuracy_score(Ytest_21 , yh_test_21_g)
# #            
# #            ac_test_21_g_mean_intime[i+1] = ( sum(ac_test_21_g_mean_intime) + ac_test_21_g[i] )/(i+1)
#             #______________________________________________________________________
           
#             #----------------------------------------------------------------------
            #%%
            
#             print('------------------------------------------------------------------------')
            
#             print("The Specification of Classifier is:")
#             print('------------------------------------------------------------------------')
#             print ("Layers Config : ")
#             print ("Neural Network Type : ", nn_type)
#             print ("No. of 1st Layer: ",num_layers[i], \
#                    '|| No. of 2nd Layer :', num_2nd_layer,\
#                    "|| No. of Classes: ",num_classes)
#             print('------------------------------------------------------------------------')
#             print ("Activtion Function: ",act_funct)
#             print ("Output Layer Activtion Function: ",output_act_funct)
#             #print('-----------------------------------------------------------------------')
#             print ("Loss Function : " , loss_funct)
#             print ("No. of Epochs : " , num_epochs)
#             print('------------------------------------------------------------------------')
#             print("Optimizer Config : ")
#             print("Type :", opt_type ," || Learning Rate: ",learning_rate,"||  Decay : " , decay_rate )
#             print('------------------------------------------------------------------------')
#             print("Learning Duration : ", round((t_end_g[i] - t_start_g[i]),2),' (secs)',"~ = ",\
#                   round((t_end_g[i] - t_start_g[i])/60),"(mins)")
#             print('------------------------------------------------------------------------')
#             #print("Getting Results  : %0.2f" % round(te_result - ts_result,2),' (sec)')
#             print()
            
            
            
#             print('========================================================================')
#             print("Results:")   
#             print('------------------------------------------------------------------------')
#             print ('TrainSet Acc = ',round(ac_train_g[i]*100,3),'%')
#             print('')
#             print('-----------------------') 
#             print ('ValidSet Acc = ',round(ac_valid_g[i]*100,3),'%')
#             print('')
#             print('-----------------------') 
#             print('TestSet Acc = ',round(ac_test_g[i]*100,3),'%')
#             print('')
#             print('TestSet DR = ',round(dr_test_g[i]*100,3),'%',\
#                   '|| TestSet Precision = ', round(pr_test_g[i]*100,3),'%')
#             print('')
#             print('TestSet F-Measure= ',round(((2*dr_test_g[i]*pr_test_g[i])\
#                                               /(dr_test_g[i]+pr_test_g[i]))*100,3),\
#                    ' || TestSet FPR = ', round(fpr_test_g[i]*100,3),'%')
#             print('-----------------------')
#             print('-----------------------')
# #            print('Test-21 Set Acc = ',round(ac_test_21_g[i]*100,3),'%')
# #            print('')
#             #print('-----------------------')
#             #print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#             #print('')
            
#             print('--------------------->    i=',i , '   <-----------------------')
#             print('------------------------------------------------------------------------')
           
# #           
# #            if (ac_test_21_g[i]*100) > 64:
# #                i= repeat_no + 0.5
            
            # plt.figure()
            # plt.grid(b=True,which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
            # plt.minorticks_on()
            # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            # plt.plot(history_g.history['acc'])
            # plt.plot(history_g.history['val_acc'])

            # plt.figure()
            # plt.grid(which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
            # plt.minorticks_on()
            # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            # plt.plot(history_g.history['categorical_crossentropy'])
            # plt.plot(history_g.history['val_categorical_crossentropy'])
            
            
                
            i+=1
            
            
            #------------------------------------------------------------------
            #+++++++++++++++++++>>       Mean & VAR       <<+++++++++++++++++++
            #------------------------------------------------------------------
#             globals()['ac_train_mean_U'+str(num_layers[0])] = np.mean(ac_train_g*100)
#             globals()['ac_train_std_U'+str(num_layers[0])] = np.std(ac_train_g*100)
            
#             globals()['ac_test_mean_U'+str(num_layers[0])] = np.mean(ac_test_g*100)
#             globals()['ac_test_std_U'+str(num_layers[0])] = np.std(ac_test_g*100)
            
# #            globals()['ac_test_21_mean_U'+str(num_layers[0])] = np.mean(ac_test_21_g*100)
# #            globals()['ac_test_21_std_U'+str(num_layers[0])] = np.std(ac_test_21_g*100)
            
#             #-------------------------------------------------------------------
#             #--------------------->>    Print Mean and Var ---------------------
            
#             print('---------------->>   Mean And Variance   <<-----------------')
#             print('Train Set Mean   = ',\
#                   round(globals()['ac_train_mean_U'+str(num_layers[0])] ,3),\
#                   ' ||  Train Set STD   = ', \
#                   round(globals()['ac_train_std_U'+str(num_layers[0])],3 ))
#             print()
#             print('Test Set Mean  = ',\
#                   round( globals()['ac_test_mean_U'+str(num_layers[0])] ,3),\
#                   ' ||  Test Set STD    = ', \
#                   round(globals()['ac_test_std_U'+str(num_layers[0])],3) )
# #            print()
# #            print('Test-21 Set Mean = ', \
# #                  round(globals()['ac_test_21_mean_U'+str(num_layers[0])],3),\
# #                  ' ||  Test-21 Set STD = ',\
# #                  round(globals()['ac_test_21_std_U'+str(num_layers[0])],3) )
#             #______________________________________________________________________________
            #___________________>>   End of GRU Implementation      <<_____________________
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    

E = yh_Xtrain_g -Xtrain_orig
e_train=np.multiply(E,E)
error_train = np.sum(e_train)

print('')
print('----------------------------------')
print ('Xtrain Total MSE  = ',error_train)
print('Xtrain Average MSE  = ',error_train/Xtrain_orig.shape[0])
print('MSE per feature = ', \
      np.sqrt([error_train/(Xtrain_orig.shape[0]*Xtrain_orig.shape[2])]))



Xtrain=yh_Xtrain_g
Xvalid=yh_valid_g
Xtest=yh_test_g





            