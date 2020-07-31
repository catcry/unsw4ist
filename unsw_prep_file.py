# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:14:26 2020

@author: catcry

        
_________________________________________________________________________
>>   Data Preparation File For "UNSW-NB15" Dataset   << 
Main File to load and prepai(Pre-process) the "UNSW-NB15" dataset:

> Loads every needed library
> Load unsw_prep_func
> Seperates Train, Validation and Test sets by given propotion
> The no. of classes can be set and 1hot labels will be produced
___________________________________________________________________________

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



# from unsw_prep_func import unsw_prep
from unsw_prep_func_predefined import unsw_prep
from funz import Metrix

#=================
num_classes =  1 #||
#=================

#__________________>    Combined Manual UNSW-NB15 Dataset (ALL)    <___________
faddr = r'F:\#DataSets\UNSW-NB15\CSVs\all_lbld.csv'

#__________________>    Predefined UNSW-NB15 Dataset (ALL)    <___________
faddr_train = r'F:\#DataSets\UNSW-NB15\CSVs\UNSW_NB15_training-set.csv'
faddr_test = r'F:\#DataSets\UNSW-NB15\CSVs\UNSW_NB15_testing-set.csv'

sep = ','

[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest] = \
            unsw_prep (faddr_data = faddr_train, \
                       faddr_unique = faddr_test,\
                       seperator = sep,\
                       vald = 1, vald_percentage = 10,\
                       test = 0, test_percentage = 0,\
                       binary_class = True)
                       
[Xtest,Ytest,a,b,c,d] =   \
            unsw_prep (faddr_data = faddr_test, \
                       faddr_unique = faddr_test,\
                           seperator = sep,\
                       vald = 0, vald_percentage = 0,\
                       test = 0, test_percentage = 0,\
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

