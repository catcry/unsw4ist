# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:42:30 2020

@author: catcry
"""

model=Sequential()
model.add(GRU(3,unroll= True, use_bias=False, input_shape=(x.shape[1],3)))
model.add(Dense(10,use_bias=False))
model.compile
model.summary()
