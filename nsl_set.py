# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:15:00 2020

@author: catcry

____________________________________________________________________________
This file can be used to import saved and pre-processed Train, Valid and Test
Numpy arrays:
    
    >NSL-KDD manual and standard 
    >NSL-KDD windowed (100 Records)
    
    >TON_IoT20 with DNS feature
    >TON_IoT20 without DNS
    >TON_IoT20 with DNS feature windowed
    
    >Decision Combine : OR and Majority
____________________________________________________________________________

"""



#_____________________>>    Load Saved NSL-KDD data     <<_____________________
Xtrain = np.load(r'f:\git\nsl4conf\nsl_Xtrain_manual.npy')
Ytrain = np.load(r'f:\git\nsl4conf\nsl_Ytrain_manual.npy')

Xtest = np.load(r'f:\git\nsl4conf\nsl_Xtest_manual.npy')
Ytest = np.load(r'f:\git\nsl4conf\nsl_Ytest_manual.npy')

Xvalid = np.load(r'f:\git\nsl4conf\nsl_Xvalid_manual.npy')
Yvalid = np.load(r'f:\git\nsl4conf\nsl_Yvalid_manual.npy')

Ytrain_1hot = Ytrain
Ytest_1hot =Ytest
Yvalid_1hot = Yvalid
Xtrain_orig = Xtrain
Xvalid_orig = Xvalid
Xtest_orig = Xtest

np.save('nsl_Xtrain_manual',Xtrain)
np.save('nsl_Ytrain_manual',Ytrain)

np.save('nsl_Xtest_manual',Xtest)
np.save('nsl_Ytest_manual',Ytest)

np.save('nsl_Xvalid_manual',Xvalid)
np.save('nsl_Yvalid_manual',Yvalid)
#______________________________________________________________________________
#_______________>>    Load Save NSL-KDD data Winded     <<_____________________

Xtrain = np.load(r'F:/Git/unsw4ist/Codez/nsl_Xtrain_winded.npy')
Ytrain = np.load(r'F:/Git/unsw4ist/Codez/nsl_Ytrain_winded.npy')


Xtest = np.load(r'F:/Git/unsw4ist/Codez/nsl_Xtest_winded.npy')
Ytest = np.load(r'F:/Git/unsw4ist/Codez/nsl_Ytest_winded.npy')


Xvalid = np.load(r'F:/Git/unsw4ist/Codez/nsl_Xvalid_winded.npy')
Yvalid = np.load(r'F:/Git/unsw4ist/Codez/nsl_Yvalid_winded.npy')


Ytrain_1hot = Ytrain
Ytest_1hot =Ytest
Yvalid_1hot = Yvalid


#________________>>    Load Saved IoT20 data (with DNS)    <<__________________
Xtrain = np.load(r'F:\Git\unsw4ist\Codez\iot20_Xtrain_dns.npy')
Ytrain = np.load(r'F:\Git\unsw4ist\Codez\iot20_Ytrain_dns.npy')

Xtest = np.load(r'F:\Git\unsw4ist\Codez\iot20_Xtest_dns.npy')
Ytest = np.load(r'F:\Git\unsw4ist\Codez\iot20_Ytest_dns.npy')

Xvalid = np.load(r'F:\Git\unsw4ist\Codez\iot20_Xvalid_dns.npy')
Yvalid = np.load(r'F:\Git\unsw4ist\Codez\iot20_Yvalid_dns.npy')

Ytrain_1hot = Ytrain
Ytest_1hot =Ytest
Yvalid_1hot = Yvalid
Xtrain_orig = Xtrain
Xvalid_orig = Xvalid
Xtest_orig = Xtest

Xtrain_orig=Xtrain_orig.reshape(Xtrain_orig.shape[0],Xtrain_orig.shape[2])
Xtest_orig=Xtest_orig.reshape(Xtest_orig.shape[0],Xtest_orig.shape[2])
Xvalid_orig=Xvalid_orig.reshape(Xvalid_orig.shape[0],Xvalid_orig.shape[2])

Xtrain_orig=Xtrain_orig.reshape(Xtrain_orig.shape[0],1,Xtrain_orig.shape[1])
Xtest_orig=Xtest_orig.reshape(Xtest_orig.shape[0],1,Xtest_orig.shape[1])
Xvalid_orig=Xvalid_orig.reshape(Xvalid_orig.shape[0],1,Xvalid_orig.shape[1])

Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[2])
Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[2])
Xvalid = Xvalid.reshape(Xvalid.shape[0],Xvalid.shape[2])

Xtrain = Xtrain.reshape(Xtrain.shape[0],1,Xtrain.shape[1])
Xvalid = Xvalid.reshape(Xvalid.shape[0],1,Xvalid.shape[1])
Xtest = Xtest.reshape(Xtest.shape[0],1,Xtest.shape[1])

a=time.time()
[y1,n1] = data_win(x1,1,100)
b=time.time()
#______________________________________________________________________________

#_____________>>    Load Saved IoT20 data (without DNS)     <<_________________
Xtrain = np.load(r'F:\Git\unsw4ist\Codez\iot_Xtrain_no_dns.npy')
Ytrain = np.load(r'F:\Git\unsw4ist\Codez\iot_Ytrain_no_dns.npy')

Xtest = np.load(r'F:\Git\unsw4ist\Codez\iot_Xtest_no_dns.npy')
Ytest = np.load(r'F:\Git\unsw4ist\Codez\iot_Ytest_no_dns.npy')

Xvalid = np.load(r'F:\Git\unsw4ist\Codez\iot_Xvalid_no_dns.npy')
Yvalid = np.load(r'F:\Git\unsw4ist\Codez\iot_Yvalid_no_dns.npy')

Ytrain_1hot = Ytrain
Ytest_1hot =Ytest
Yvalid_1hot = Yvalid
Xtrain_orig = Xtrain
Xvalid_orig = Xvalid
Xtest_orig = Xtest

#_______________>>    Load Save IoT20 data Winded     <<_______________________

Xtrain = np.load(r'F:\Git\unsw4ist\Codez\Xtrain_iot20_winded.npy')
Ytrain = np.load(r'F:\Git\unsw4ist\Codez\iot20_Ytrain_dns.npy')
Ytrain = Ytrain[99:]

Xtest = np.load(r'F:\Git\unsw4ist\Codez\Xtest_iot20_winded.npy')
Ytest = np.load(r'F:\Git\unsw4ist\Codez\iot20_Ytest_dns.npy')
Ytest = Ytest[99:]

Xvalid = np.load(r'F:\Git\unsw4ist\Codez\Xvalid_iot20_winded.npy')
Yvalid = np.load(r'F:\Git\unsw4ist\Codez\iot20_Yvalid_dns.npy')
Yvalid = Yvalid[99:]

Ytrain_1hot = Ytrain
Ytest_1hot =Ytest
Yvalid_1hot = Yvalid
#______________________________________________________________________________

ac_train_dc_gru  = np.zeros(len(layers))
conf_train_dc_gru = np.zeros([len(layers),2,2])

ac_valid_dc_gru = np.zeros(len(layers))
conf_valid_dc_gru = np.zeros([len(layers),2,2])

ac_test_dc_gru = np.zeros(len(layers))
dr_test_dc_gru = np.zeros(len(layers))
conf_test_dc_gru = np.zeros([len(layers),2,2])
pr_test_dc_gru = np.zeros(len(layers))
fpr_test_dc_gru = np.zeros(len(layers))

for k in range (11):
    Xtrain = globals()['Xtrain'+str(k)]
    Xvalid = globals()['Xvalid'+str(k)]
    Xtest = globals()['Xtest'+str(k)]
    runfile('F:/Git/unsw4ist/Codez/GRU.py', wdir='F:/Git/unsw4ist/Codez', \
            current_namespace=True)
    ac_train_dc_gru[k] =  ac_train_g[0]   
    conf_train_dc_gru[k] = train_conf_matrix_g
    ac_valid_dc_gru[k] = ac_valid_g[0]
    conf_valid_dc_gru[k] = valid_conf_matrix_g
    ac_test_dc_gru[k]  = ac_test_g[0] 
    conf_test_dc_gru[k] = test_conf_matrix_g
#______________________________________________________________________________
    
ac_train_dc_rf  = np.zeros(len(layers))
conf_train_dc_rf = np.zeros([len(layers),2,2])

ac_valid_dc_rf = np.zeros(len(layers))
conf_valid_dc_rf = np.zeros([len(layers),2,2])

ac_test_dc_rf = np.zeros(len(layers))
dr_test_dc_rf = np.zeros(len(layers))
conf_test_dc_rf = np.zeros([len(layers),2,2])
pr_test_dc_rf = np.zeros(len(layers))
fpr_test_dc_rf = np.zeros(len(layers))

for k in range (11):
    Xtrain = globals()['Xtrain'+str(k)]
    Xvalid = globals()['Xvalid'+str(k)]
    Xtest = globals()['Xtest'+str(k)]
    runfile('F:/Git/unsw4ist/Codez/rf.py', wdir='F:/Git/unsw4ist/Codez', \
            current_namespace=True)
    ac_train_dc_rf[k] =  ac_train_rf[0]   
    conf_train_dc_rf[k] = train_con_matrix_rf
    ac_valid_dc_rf[k] = ac_valid_rf[0]
    conf_valid_dc_rf[k] = valid_con_matrix_rf
    ac_test_dc_rf[k]  = ac_test_rf[0] 
    conf_test_dc_rf[k] = test_con_matrix_rf
    
    
    
yh_Xtrain_c = yh_Xtrain_c.reshape(len(yh_Xtrain_c))
yh_valid_c = yh_valid_c.reshape(len(yh_valid_c))
yh_test_c = yh_test_c.reshape(len(yh_test_c))

#++++++++++++++++++++>>           OR           <<+++++++++++++++++++++++++++++                                        
yh_Xtrain_or = yh_Xtrain_g | yh_Xtrain_c | yh_train_rf
ac_train_or = accuracy_score(Ytrain,yh_Xtrain_or)
ac_train_or_confusion = confusion_matrix (Ytrain,yh_Xtrain_or)

yh_valid_or = yh_valid_g | yh_valid_c | yh_valid_rf
ac_valid_or = accuracy_score(Yvalid,yh_valid_or)
ac_valid_or_confusion = confusion_matrix (Yvalid,yh_valid_or)

yh_test_or = yh_test_g | yh_test_rf | yh_test_c
ac_test_or = accuracy_score(Ytest,yh_test_or)
ac_test_or_confusion = confusion_matrix (Ytest,yh_test_or)


#++++++++++++++++++++>>   Majority Opinion      <<+++++++++++++++++++++++++++++
yh_Xtrain_major_sum = sum(np.array([yh_Xtrain_c,\
                            yh_Xtrain_g,\
                            yh_train_rf]))
yh_Xtrain_major = np.zeros(len(yh_Xtrain_major_sum))
    
for j in range (len(yh_Xtrain_major_sum)):
    if yh_Xtrain_major_sum[j] > 1 :
        yh_Xtrain_major[j] = 1

ac_train_cmbnd = accuracy_score(Ytrain_1hot,yh_Xtrain_major)

#-----------------------------------------------------------------
yh_test_major_sum = sum(np.array([yh_test_c,\
                            yh_test_g,\
                            yh_test_rf]))
    
yh_test_major = np.zeros(len(yh_test_major_sum))   

for j in range (len(yh_test_major_sum)):
    if yh_test_major_sum[j] > 1:
        yh_test_major[j] = 1      
        
ac_test_cmbnd = accuracy_score(Ytest,yh_test_major)
ac_test_cmbnd_confusion = confusion_matrix (Ytest,yh_test_major)