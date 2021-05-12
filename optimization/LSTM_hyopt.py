# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:15:00 2020

@author: catcry

______________________________________________________________________________
LSTM-based Classifier -  Baysian  HyperParameter  Optimization
    > plotting Accuracy and Binary cross entropy for every opt run
______________________________________________________________________________
"""

from hyperopt import pyll, hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

num_classes =1
space = {
    'num_units' : hp.quniform('num_units',10,200,10),\
    'num_2nd_units' : hp.quniform('num_2nd_units',10,200,10),\
    'act_funct' : hp.choice('act_funct',['elu','relu','tanh','sigmoid']),\
    'num_epochs' : hp.quniform('num_epochs', 100,1000,50),\
    'learning_rate' : hp.uniform('learning_rate',0.00001,0.01),\
    'latent' : hp.choice('latent',[2,3,4])
    }
    
    
    
    
    
    
    
    
    
cls_weights = compute_class_weight('balanced',np.unique(Ytrain),Ytrain)


    
    
def objective(space):
    #%%--------->>   Some Predfinitions
    learning_rate = space['learning_rate']
    num_epochs = int(space['num_epochs'])
    decay_rate = learning_rate/num_epochs
    adam = optimizers.Adam(lr = learning_rate , \
                         beta_1 = 0.9 , \
                         beta_2 = 0.999 , \
                         epsilon = None,\
                         decay = decay_rate,\
                         amsgrad = True )
    
    
    #%%---------->>    Model Structture
    t_start_g = time.time()
    model_g= Sequential()
    model_g.add(Dense(space['latent'],activation = space['act_funct'],\
                              input_shape = (Xtrain.shape[1],Xtrain.shape[2])))
    model_g.add(LSTM(units = int(space['num_units']), \
                            # recurrent_activation = 'elu', \
                            use_bias = True ,\
                            activation = space['act_funct'],\
                            unroll= True,\
                            implementation = 1,\
##                           kernel_initializer = 'he_uniform',#keras.initializers.Ones(),\
##                           recurrent_initializer = keras.initializers.Ones(),\
##                           bias_initializer = keras.initializers.Ones(),\
                            return_sequences = True,\
                           # kernel_regularizer=regularizers.l2(0.1),\
                          # activity_regularizer=regularizers.l2(0.1),\
                           # recurrent_regularizer=regularizers.l2(0.1), \
                           # bias_regularizer=regularizers.l2(0.1),\
                            # input_shape = (Xtrain.shape[1], Xtrain.shape[2]),\
                            ))
    model_g.add(LSTM(units = int(space['num_units']), \
                          recurrent_activation= space['act_funct'], \
                          use_bias = True ,\
                          activation = space['act_funct'],\
                          unroll= True,\
                          implementation=1,\
                          return_sequences=False,\
                          ))
    model_g.add(Dense(80,activation = space['act_funct']))
    model_g.add(Dense(60,activation = space['act_funct']))
    model_g.add(Dense(40,activation = space['act_funct']))
    model_g.add(Dense(20,activation = space['act_funct']))
    model_g.add(Dense(10,activation = space['act_funct']))
    model_g.add(Dense(num_classes, activation = 'sigmoid' ))
    
    #%%---------->>   Model Fit
    model_g.compile(loss=loss_funct , optimizer = adam , \
                            metrics=['binary_crossentropy', 'acc'])
    
    history_g = model_g.fit (Xtrain,Ytrain_1hot, \
                                 epochs = num_epochs , \
                                 batch_size = batch, \
                                 class_weight = cls_weights,\
                                 # validation_split = 0.1,\
                                  validation_data = (Xvalid,Yvalid_1hot) ,\
                                # validation_data = (Xtest,Ytest_1hot),\
                                 verbose = 1, \
                                 shuffle = True,\
                                 #callbacks=[globals()['csv_logger_U'+str(num_layers[0])],\
#                                            callback]\
                                            )
    t_end_g = time.time()
    learn_time = t_end_g - t_start_g
    
    #%%---------->>     Plotting
    plt.figure()
    plt.grid(b=True,which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(history_g.history['acc'])
    plt.plot(history_g.history['val_acc'])

    plt.figure()
    plt.grid(which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(history_g.history['binary_crossentropy'])
    plt.plot(history_g.history['val_binary_crossentropy'])
    
    return {'loss' : history_g.history['val_binary_crossentropy'][num_epochs-1],\
            'status': STATUS_OK}

trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 40,
            trials= trials)

# {'act_funct': 2,
#  'latent': 4,
#  'learning_rate': 0.009227814537638688,
#  'num_2nd_units': 110.0,
#  'num_epochs': 600.0,
#  'num_units': 180.0}