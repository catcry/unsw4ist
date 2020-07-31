# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:49:26 2020

@author: catcry
_________________________________________________________________________
        >>     Result Combine    <<
_________________________________________________________________________

"""


#==============================================================================
#=====================>  Combing the Results    <==============================
#==============================================================================


#++++++++++++++++++++>>   OR Logic combing      <<+++++++++++++++++++++++++++++
yh_Xtrain_g = yh_Xtrain_g.reshape(len(yh_Xtrain_g))
yh_test_g = yh_test_g.reshape(len(yh_test_g))

# RF Results to '(int)'
yh_test_rf = yh_test_rf.astype(int)

yh_Xtrain_or = yh_Xtrain_g | yh_Xtrain_c | yh_train_rf
ac_train_or = accuracy_score(Ytrain,yh_Xtrain_or)
ac_train_or_confusion = confusion_matrix (Ytrain,yh_Xtrain_or)

yh_test_or = yh_test_g | yh_test_rf #| yh_test_c
ac_test_or = accuracy_score(Ytest,yh_test_or)
ac_test_or_confusion = confusion_matrix (Ytest,yh_test_or)

yh_test_21_or = yh_test_21_c | yh_test_21_g |  yh_test_21_rf
ac_test_21_or = accuracy_score(Ytest_21,yh_test_21_or)
ac_tesr_21_or_confusion = confusion_matrix (Ytest_21,yh_test_21_or)
#______________________________________________________________________
________
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
yh_test_major_sum = sum(np.array([#yh_test_c,\
                            yh_test_g,\
                            yh_test_rf]))
    
yh_test_major = np.zeros(len(yh_test_major_sum))   

for j in range (len(yh_test_major_sum)):
    if yh_test_major_sum[j] > 1:
        yh_test_major[j] = 1      
        
ac_test_cmbnd = accuracy_score(Ytest,yh_test_major)
ac_test_cmbnd_confusion = confusion_matrix (Ytest,yh_test_major)
#------------------------------------------------------------------
yh_test_21_major_sum = sum(np.array([yh_test_21_c,\
                            yh_test_21_g,\
                            yh_test_21_rf]))
    
yh_test_21_major = np.zeros(len(yh_test_21_major_sum))   

for j in range (len(yh_test_21_major_sum)):
    if yh_test_21_major_sum[j] > 1 :
        yh_test_21_major[j] = 1      
        
ac_test_21_cmbnd = accuracy_score(Ytest_21,yh_test_21_major) 
ac_tesr_21_or_confusion = confusion_matrix (Ytest_21,yh_test_21_or)     
#_______________________>   End of getting cmnd results     <__________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


confc = Metrix(valid_con_matrix_c)
confg = Metrix(valid_conf_matrix_g)
confr = Metrix(valid_conf_matrix_rf)

pc = confc.dr/confc.fr
pg = confg.dr/confg.fr
pr = confr.dr/confr.fr

wc = pc/ (pc+pg+pr)
wg = pg/ (pc+pg+pr)
wr = pr/ (pc+pg+pr)


print (wc,wg,wr)


yh_test_pro = wc*yh_test_c + wg * yh_test_g + wr*yh_test_rf

i=0
while i<len(yh_test_pro):
    if yh_test_pro[i]> 0.2:
       yh_test_pro[i]=1
    else:
       yh_test_pro[i] =0
    i+=1
    
test_conf_pro = confusion_matrix(Ytest,yh_test_pro)

#==============================================================================
#====================>>          Printing Results        <<====================
#==============================================================================

print('========================================================================')
print("Results for OR Mode:")
print('----------------------')
print ('TrainSet Acc = ',round(ac_train_or*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_or*100,3),'%')
print('')
#print('-----------------------')
#print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#print('')

print('========================================================================')
print("Results for Combined Mode:")
print('-----------------------------')
print ('TrainSet Acc = ',round(ac_train_cmbnd*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_cmbnd*100,3),'%')
print('-----------------------')
#print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#print('')

print('------------------------------------------------------------------------')



