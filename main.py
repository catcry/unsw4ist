# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:52:42 2021

@author: catcry

            Main Run function for unsw pack
            
"""

from unsw4ist.preprocess.iot_prep_file import *
from unsw4ist.classifier import *
from unsw4ist.autoencoders import * 
from unsw4ist.optimization import *


def main(dataset, ):
    
faddr = r'F:\Git\unsw4ist\IoT20\Network\Train_Test_Network.csv'
num_classes = 1

[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest] = \
    iot20_sets (faddr, num_classes = num_classes)
    