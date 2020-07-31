# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:48:59 2020

@author: catcry

                        >>>      AutoEncoder       <<<                 
"""
Xtrain_compact = Xtrain0
Xvalid_compact = Xvalid0
Xtest_compact = Xtest0


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#from bestnncomb import create_model

num_classes = 1

loss_funct = 'mse'#'binary_crossentropy'# 
batch = 20240 #[256,1024,10240]
no_epochs = 100 #[20,50,80]
# layers = [[5],[30,10],[5,30,10],[35,25,15,5],[5,35,25,15,5],\
#           [37,29,21,13,5],[5,37,29,21,13,5],\
#           [40,35,30,25,20,15,5],[5,40,35,30,25,20,15,5]]




layers = [[10,20,30,40,50,60,70,80,70,60,55]]
# layers = [[80,70,60,50]]
# layers = [[40,30,20,10,5,10,20,30,40]]
act_func = 'relu'
# cls_weights = compute_class_weight('balanced',np.unique(Ytrain),Ytrain)

opt_type = 'Adam'
learning_rate = 0.009
decay_rate = learning_rate/no_epochs
moment = 0.8

adam = optimizers.Adam(lr = learning_rate, \
                         beta_1 = 0.9 , \
                         beta_2 = 0.999 , \
                         epsilon = None,\
                         decay = decay_rate,\
                         amsgrad = True )

adadelta = optimizers.Adadelta(lr = learning_rate, \
                         rho=0.95 , \
                         epsilon = None,\
                         decay = decay_rate)
#______________________>>    Deep Model Creator   <<___________________________
# def create_model(layers, activation, optimizer):
#     model = Sequential()
#     for i, nodes in enumerate(layers):
#         if i==0:
#             model.add(Dense(nodes, activation = activation,\
#                             input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
#         else:
#             model.add(Dense(nodes,activation = activation))
#     model.add(Dense(1,activation = 'sigmoid'))        
#     model.compile (loss='binary_crossentropy' , optimizer = optimizer , \
#                          metrics=['acc','binary_crossentropy'])
#     return model
#_-----------------------------------------------------------------------------

#__________________________>>   AutoEncoder create func   <<___________________
def create_autoen(data,layers, activation, optimizer):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, activation = activation,\
                            input_shape = (data.shape[1], data.shape[2])))
        else:
            model.add(Dense(nodes,activation = activation))
    model.add(Dense(42, activation = 'sigmoid'))
    model.compile (loss=loss_funct , optimizer = optimizer , \
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
    plt.figure()
    plt.grid(b=True,which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.figure()
    plt.grid(which ='both', axis = 'both' ,color='r', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mse'])
    
    data_h = model.predict(data)
    # train_conf = confusion_matrix (data,data_h)
    # train_ac = accuracy_score (data,data_h)
    
    vdata_h = model.predict(vdata)
    # valid_conf = confusion_matrix (vdata,vdata_h)
    # valid_ac = accuracy_score (vdata,vdata_h)
    
    tdata_h = model.predict(tdata)
    return (data_h,vdata_h,tdata_h)
    
model = create_autoen(data = Xtrain_compact, layers = layers[0],\
                      activation = act_func, optimizer = adam)    
[Xtrain,Xvalid,Xtest] = best_model(model=model, \
                                   no_epochs = no_epochs,\
                                   batch_size = batch,\
                                   data = Xtrain_compact,y=Xtrain_orig,\
                                   tdata = Xtest_compact,yt=Xtest_orig,\
                                   vdata = Xvalid_compact,yv=Xvalid_orig)        
    
E_train = (Xtrain-Xtrain_orig)
e_train=np.multiply(E_train,E_train)
error_train = np.sum(e_train)


print('')
print('----------------------------------')
print ('Xtrain Total MSE  = ',error_train)
print('Xtrain Average MSE  = ',error_train/Xtrain.shape[0])
print('MSE per feature = ', \
      np.sqrt([error_train/(Xtrain_compact.shape[0]*Xtrain_compact.shape[2])]))

E_test = Xtest -Xtest_orig
e_test=np.multiply(E_test,E_test)
error_test = np.sum(e_test)

print('')
print('----------------------------------')
print ('Xtest Total MSE  = ',error_test)
print('Xtest Average MSE  = ',error_test/Xtest_orig.shape[0])
print('MSE per feature = ', \
      np.sqrt([error_test/(Xtest_orig.shape[0]*Xtest_orig.shape[2])]))
