#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:58:20 2020

@author: samthomas
"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

def linear_regressor(x_train, x_test, y_train, y_test):
    
    print('Fitting the model.\n')
    reg_model = linear_model.LinearRegression().fit(x_train, y_train)
    
    print('Making predictions.\n')
    Y_pred_train = reg_model.predict(x_train)

    Y_pred_test = reg_model.predict(x_test)
    
    print('Calculating train and test error.\n')
    RMSE_train = sqrt(mean_squared_error(y_train, Y_pred_train))
    
    RMSE_test = sqrt(mean_squared_error(y_test, Y_pred_test))
    
    print('Train error: ', RMSE_train)
    print('Test error: ', RMSE_test)
    
    return RMSE_train, RMSE_test, Y_pred_train, Y_pred_test

if __name__ == '__main__':
    linear_regressor(x_train, x_test, y_train, y_test)
    