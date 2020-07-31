# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:18:26 2020

____________________________________________________________________________
Two Function for windowing datasets
> best solution need less time and it is feasible 
____________________________________________________________________________

@author: catcry
"""

from keras.preprocessing import sequence
x = np.arange(1,101)
x = x.reshape(20,5)
y = np.arange(1,21)
ts= 5

p = sequence.pad_sequences(x,maxlen =   ts)

model = Sequential()
model.add(Embedding(x.shape[0],x.shape[1],input_length=4))   
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x, y, epochs=3, batch_size=64)



time_step = 1
time_win = 4
i = time_step
x1 = x[0:time_win,:]
while i<= len(x)-time_win:
    temp = x[i:i+time_win,:]
    x1 = np.concatenate((x1,temp),axis=0)
    i = i + time_step

def data_win(data,time_step,time_win):
    x = data[0:time_win,:]
    i = time_step
    while i<= len(data)-time_win:
        x = np.concatenate((x,data[i:i+time_win,:]),axis=0)
        i = i + time_step
    x=x.reshape(i,time_win,data.shape[1])
    return (x,i)
    
#____________________>>>  Windowning Best Solution    <<<______________________    
def data_windowing (data,labels,time_step,time_win):
    i=time_step
    while i<time_win:
      globals()['data_shift'+str(i)] = data[i:,:]
      i=i+time_step
    
    data = data[0:-time_win+1]
    n=time_step
    while n<i-time_step:
        globals()['data_shift'+str(n)] = globals()['data_shift'+str(n)][0:-time_win+n+1]
        data = np.dstack((data,globals()['data_shift'+str(n)]))
        n=n+time_step
    data=np.dstack((data,globals()['data_shift'+str(n)]))
    data=data.transpose(0,2,1)
    labels = labels[time_win-1:]
    return data,labels
    
    