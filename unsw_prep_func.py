    
# faddr = r'F:\#DataSets\UNSW-NB15\CSVs\UNSW_NB15_testing-set.csv'

seperator = ','


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


def unsw_prep (faddr,seperator,vald,vald_percentage,test,test_percentage,binary_class):
    
    x = pd.read_csv(faddr,sep=seperator)  #Reading csv File
   
    
    #==========================================================================
    #-------------------->>>       cleaning dataset       <<<------------------
    #==========================================================================
    
    x = x.drop(columns = ['Source IP','Destination IP'])  #Drop Source and Dest IPs
    x['attack_cat'] = x['attack_cat'].fillna('normal')  #Fill 'NaN' with 'Normal'
    
    
    #___________________>    Converting hex in string to int   <_______________
    
    sport_dtype = x['Source Port'].dtype
    dport_dtype = x['Destination Port'].dtype
    
    if sport_dtype == 'O':
        x['Source Port'][x['Source Port'].str[1]==('x')]= \
        x['Source Port'][x['Source Port'].str[1]==('x')].apply(lambda x: int(x, 16))
    
    if dport_dtype == 'O':
        x['Destination Port'][x['Destination Port'].str[1]==('x')]=\
        x['Destination Port'][x['Destination Port'].str[1]==('x')].apply(lambda x: int(x, 16))
    #__________________________________________________________________________
    
    #____________________>    Handling Missing Data     <______________________
    
    for i in list(x):
        globals()['isna_'+i] = x[x[i].isna()]
        globals()['isna_index_'+i] = x.index.values.astype(int)[x[i].isna()]
        
#    for i in list(data):
#        globals()['missin_dash_'+i] = data[data[i]=='-']
#        globals()['missin_dash_index_'+i] = data.index.values.astype(int)[x[i]=='-']
        
        
    #   ICMP protocol (Internet Control Message Protocol) is a protocol that manages\
    #   Error messages, does not use port, so searching the whole dataset with 
    #   "f=x[x['Transaction Protocol']=='icmp']" showed that the dashed('-') "Destination Ports"
    #   are all using "icmp" protocol and their next entry are also a "icmp" message
    #   with the same "Source Port" and "0" filled "Destination Port", So all of 
    #   dashed('-') entries in this column filled by "0".
    x= x.replace({'Destination Port':'-'}, 0)
    x= x.replace({'Source Port':'-'}, 0)
    
    
    #'ct_flw_http_mthd' : No. of flows that has methods such as Get and Post in http service.
    #"1348145" records have 'NaN' in this feature, they all are set to '0'      
    x['ct_flw_http_mthd'] = x['ct_flw_http_mthd'].fillna(0)
    
    #   'ct_ftp_cmd' is the no. of flows that has a command in a ftp session
    #   the records with ' ' (space char) entries are put to "0" commands.
    x= x.replace({'ct_ftp_cmd':' '}, 0)
    
    
    x['is_ftp_login'] = x['is_ftp_login'].fillna(0)
    
    
    #__________________________________________________________________________
    #____________________________>>>      Numifi     <<<_______________________
    
    #Defining "do" as an Object of Nomifi class which Numericalize Nominal Features
    do = Numifi(x,['Transaction Protocol','State','Service','attack_cat']) 
    data = do.convtonum()
    
    
    #    Making all entries astype(float)
    for i in list(data):
        data[i] = data[i].astype(float)
    # Attack_category and Labels as Y Arrays and Dropping them from feature set
    y_cat = np.array(data['attack_cat'])
    y_lbl = np.array(data['Label'])
    data = data.drop(columns = ['attack_cat','Label'])
       
    #========================>   Normalization   <=============================
    
    scaler = MinMaxScaler()
    list_features = ['Source Port','Destination Port', 'Transaction Protocol',\
           'State', 'Duration', 'Sbytes','Dbytes','Sttl','Sttl2','Sloss','Dloss',\
           'Service', 'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',\
           'smeansz','dmeansz','trans_depth', 'res_bdy_len', 'Sjit', 'Djit',\
           'Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',\
           'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',\
           'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_Itm','ct_src_Itm',\
           'ct_src_dport_Itm','ct_dst_sport_Itm','ct_dst_src_Itm']
    
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
            
