# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:41:20 2020


_________________________________________________________________________
Bayesian Hyperparameter Optimization for a DNN Encoder 
_________________________________________________________________________


@author: catcry
"""


from hyperopt import pyll, hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


num_classes =1
space = {
    'layers' : hp.choice('layers',[[10,20,30,40,50,60,70,80,70,60,55],\
                                   [10,20,30,40,50],\
                                   [70,80,70,60,55,50]  ,\
                                   [10,20,30,40,50,60,70,80,55]]),\
    'act_funct' : hp.choice('act_funct',['elu','relu','tanh','sigmoid']),\
    'num_epochs' : hp.quniform('num_epochs', 1000,4000,500),\
    'learning_rate' : hp.uniform('learning_rate',0.0001,0.01)
    # 'latent' : hp.choice('latent',[2,3,4])
    }
    
    
def create_autoen(data,layers, activation, optimizer):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, activation = activation,\
                            input_shape = (data.shape[1], data.shape[2])))
        else:
            model.add(Dense(nodes,activation = activation))
    model.add(Dense(42, activation = 'sigmoid'))
    model.compile (loss='mse' , optimizer = optimizer , \
                         metrics=['mse','binary_crossentropy'])
    return model

#------------------------------------------------------------------------------

def best_model (model,data,vdata,tdata,y,yt,yv,no_epochs, batch_size):
    history = model.fit(data,y,\
                            epochs=no_epochs,\
                            batch_size=batch_size, \
                            validation_data=(vdata,yv),\
                            verbose=1,\
                            shuffle=True) 
    data_h = model.predict(data)
    # train_conf = confusion_matrix (data,data_h)
    # train_ac = accuracy_score (data,data_h)
    
    vdata_h = model.predict(vdata)
    # valid_conf = confusion_matrix (vdata,vdata_h)
    # valid_ac = accuracy_score (vdata,vdata_h)
    
    tdata_h = model.predict(tdata)
    return (data_h,vdata_h,tdata_h)
      
    
    
    
    
    
batch_size = 2000000   
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
    model = create_autoen(data = Xtrain_compact, layers = space['layers'],\
                      activation = space['act_funct'], optimizer = adam)    
        
    history = model.fit(Xtrain_compact,Xtrain_orig,\
                            epochs=num_epochs,\
                            batch_size=batch_size, \
                            validation_data=(Xvalid_compact,Xvalid_orig),\
                            verbose=1,\
                            shuffle=True) 
    data_h = model.predict(Xtrain_compact)
    # train_conf = confusion_matrix (data,data_h)
    # train_ac = accuracy_score (data,data_h)
    
    vdata_h = model.predict(Xvalid_compact)
    # valid_conf = confusion_matrix (vdata,vdata_h)
    # valid_ac = accuracy_score (vdata,vdata_h)
    
    tdata_h = model.predict(Xtest_compact)    

    t_end_g = time.time()
    learn_time = t_end_g - t_start_g
    return {'loss' : history.history['val_mse'][num_epochs-1],\
            'status': STATUS_OK}
    
    #%%---------->>     Plotting
  
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