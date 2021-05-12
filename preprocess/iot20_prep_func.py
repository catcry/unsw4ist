# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:15:30 2020

@author: catcry

____________________________________________________________________________
                        >>>  UNSW - TON-IoT20   <<<
                        Dataset Preperation Function
    > Loading
    > Handling missing data
    > Defined a class to numerize nominal features
    > Function parameters: 
                            File address
                            seperator
                            validition set generation and its percentage
                            Test set generation and its percentage
                            binary or multiclass labels
____________________________________________________________________________

"""

    


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Numifi():
    import pandas as pd
    def __init__(self,data,name):
        self.name = name
        self.data = data
        self.vectors = self.data[self.name]
#        self.uniq = data[self.name].unique()
#        self.len = len()

    def convtonum(self):
        i=0
        while i<len(self.name):
            j=0
            uniq = self.data[self.name[i]].unique()
            while j<len(uniq):
                self.data = self.data.replace({self.name[i] : uniq[j]}, j)
                j+=1
            i+=1
        return self.data


def iot20_prep (faddr,seperator,vald,vald_percentage,test,test_percentage,binary_class):
    
    x = pd.read_csv(faddr,sep=seperator)  #Reading csv File
   
    
    #==========================================================================
    #-------------------->>>       cleaning dataset       <<<------------------
    #==========================================================================
    
    x = x.drop(columns = ['src_ip','dst_ip','ts','dns_query'])  #Drop Source and Dest IPs
    
    
    
    #___________________>    Converting hex in string to int   <_______________
    
    
    #__________________________________________________________________________
    
    #____________________>    Handling Missing Data     <______________________
    
#    for i in list(x):
#        globals()['isna_'+i] = x[x[i].isna()]
#        globals()['isna_index_'+i] = x.index.values.astype(int)[x[i].isna()]
        
#    for i in list(data):
#        globals()['missin_dash_'+i] = data[data[i]=='-']
#        globals()['missin_dash_index_'+i] = data.index.values.astype(int)[x[i]=='-']
        
   
#    'http_trans_depth' feature is a numerical feature but it has the string 
#    data type, it is changed to numerical manualy because the numifi class 
#    may treat its values differently which its meaning would be gone.
 
    x= x.replace({'http_trans_depth':'-'}, 0)
    x['http_trans_depth'] = x['http_trans_depth'].astype('int64')
    
    
    #--------------------------------------------------------------------------
    #_________________>   Getting the list of non num features   <_____________
    ls_strings =[]

    for column in list(x.columns):
        if x[column].dtypes == 'O':
            ls_strings.append(column)
    
    
    #__________________________________________________________________________
    #____________________________>>>      Numifi     <<<_______________________
    
    #Defining "do" as an Object of Nomifi class which Numericalize Nominal Features
    do = Numifi(x,ls_strings) 
    data = do.convtonum()
    
    
    #    Making all entries astype(float)
    for i in list(data):
        data[i] = data[i].astype(float)
    # Attack_category and Labels as Y Arrays and Dropping them from feature set
    y_cat = np.array(data['type'])
    y_lbl = np.array(data['label'])
    data = data.drop(columns = ['type','label'])
       
    #__________________________________________________________________________
    #_____________________>>>      Normalization     <<<_______________________
    
    
    scaler = MinMaxScaler()
    list_features = list(data.columns)
    
    
    data [list_features] = scaler.fit_transform(data [list_features])                    
    
    X = data.to_numpy()
    #--------------------------------------------------------------------------
    
    if binary_class:
        Y=y_lbl
    else:
        Y=y_cat
    
    if \
    vald == 1 and \
    test == 1 and \
    vald_percentage != 100 and \
    test_percentage != 100 :
         X,Xtest,Y,Ytest = train_test_split(X,Y, stratify=Y,test_size=(test_percentage/100))
         Xtrain,Xvalid,Ytrain,Yvalid = train_test_split(X,Y, stratify=Y,test_size=(vald_percentage/100))
         
    elif vald == 0 and test == 1 and vald_percentage !=100 and test_percentage != 100:
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y, stratify=Y,test_size=(test_percentage/100))
        Xvalid = 0
        Yvalid = 0
        
    elif vald ==1 and test == 0 and vald_percentage !=100 and test_percentage != 100:
        Xtrain,Xvalid,Ytrain,Yvalid = train_test_split(X,Y, stratify=Y,test_size=(vald_percentage/100))
        Xtest = 0
        Ytest = 0
                
    elif vald ==0 and test == 0 and vald_percentage !=100 and test_percentage != 100 :
            Xtrain = X
            Ytrain = Y            
            Xvalid = 0
            Xtest = 0
            Ytest = 0
            Yvalid = 0
            
    elif vald ==1 and vald_percentage ==100 or test == 1  and test_percentage == 100 :
            Xtrain = 0
            Ytrain = 0            
            Xvalid = 0
            Yvalid = 0
            Xtest = X
            Ytest = Y
            
    return Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest  
#_____________________________________________________________________________#
#=============================================================================#
            
    
#______________________________________________________________________________
#_________________>>    Some Useful  Data Analyse  Pieces   <<_________________
#______________________________________________________________________________            
#list_strings = ['proto', 'service', 'conn_state', 'dns_query', 'ssl_version',\
#                'ssl_cipher', 'ssl_subject', 'ssl_issuer', 'http_method',\
#                'http_uri', 'http_version', 'http_user_agent',\
#                'http_orig_mime_types','http_resp_mime_types', 'type', \
#                'weird_name', 'weird_addl']
#
#list_bools = ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed',\
#              'ssl_established', 'weird_notice']
#
#
#ls_strings =[]
#
#for column in list(x.columns):
#    if x[column].dtypes == 'O':
#        ls_strings.append(column)
#        
            
            
            
            
            
            
            
            
            
            
            
            
            