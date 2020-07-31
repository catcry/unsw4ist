# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:41:35 2020

_________________________________________________________________________
 Main File to load and prepai(Pre-process) the "TON_IoT20" dataset:

> Loads every needed library
> Load iot20_prep_func
> Seperates Train, Validation and Test sets by given propotion
> The no. of classes can be set and 1hot labels will be produced
___________________________________________________________________________

@author: catcry
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as k

#Keras Lib Essens:
#import tensorflow.python.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM ,GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D,  AveragePooling1D, Flatten
from tensorflow.keras.layers import Embedding
from keras.layers import CuDNNLSTM
from tensorflow.keras.activations import sigmoid, elu
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU, ThresholdedReLU, ReLU
from tensorflow.keras import losses , optimizers, regularizers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model


##Keras with out Tensorflow
#from keras.models import Sequential
#from keras.layers import Activation, Dense, LSTM ,GRU, Bidirectional
#from keras.layers import Conv1D, MaxPooling1D,  AveragePooling1D
#from keras.layers import CuDNNLSTM
#from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU, ThresholdedReLU, ReLU
#from keras import losses , optimizers, regularizers
#from keras.utils.np_utils import to_categorical
#from keras.models import load_model

#Ski-Learn Essentials
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#ski-learn ML
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans



from iot20_prep_func import iot20_prep

#=================
num_classes =  1 #||
#=================

faddr = r'F:\Git\unsw4ist\IoT20\Network\Train_Test_Network.csv'
seperator = ','


[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest] = \
            iot20_prep (faddr = faddr, seperator = seperator,\
                       vald = 1, vald_percentage = 10,\
                       test = 1, test_percentage = 10,\
                       binary_class = True)
            
# Definin appropriate array shapes for Keras Neural Nets
Xtrain = Xtrain.reshape((len(Xtrain),1,Xtrain.shape[1]))
Xtrain_orig = np.copy(Xtrain)
Xvalid = Xvalid.reshape((len(Xvalid),1,Xvalid.shape[1]))
Xvalid_orig = np.copy(Xvalid)
Xtest  = Xtest.reshape((len(Xtest),1,Xtest.shape[1]))
Xtest_orig = np.copy(Xtest)

Ytrain_1hot = np.zeros([len(Ytrain),num_classes])
Yvalid_1hot = np.zeros([len(Yvalid),num_classes])
Ytest_1hot = np.zeros([len(Ytest),num_classes])

if num_classes != 1:
    for i in range(len(Ytrain)):
            Ytrain_1hot[i,Ytrain[i]] = 1
    
    for i in range(len(Yvalid)):
            Yvalid_1hot[i,Yvalid[i]] = 1
        
    for i in range(len(Ytest)):
            Ytest_1hot[i,Ytest[i]] = 1  
    
 
else:
      Ytrain_1hot = Ytrain
      Yvalid_1hot = Yvalid
      Ytest_1hot = Ytest
