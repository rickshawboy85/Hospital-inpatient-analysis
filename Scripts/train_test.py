#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:07:05 2020

@author: samthomas
"""


from sklearn.model_selection import train_test_split

def train_test(x, y):
   
    x_train, x_test, y_train, y_test  = train_test_split(x, y, train_size=0.8, random_state=15)
    
    print('x_train dimensions: ', x_train.shape)
    print('x_test dimensions: ', x_test.shape)
    print('y_train dimensions: ', y_train.shape)
    print('y_test dimensions: ', y_test.shape)
    
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    train_test(x, y)