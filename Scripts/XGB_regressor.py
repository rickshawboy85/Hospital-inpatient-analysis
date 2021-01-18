#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:57:34 2020

@author: samthomas
"""
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

XGmodel = XGBRegressor()
XGmodel.get_params()

def XGB_regressor(x_train, x_test, y_train, y_test):
    
    XGmodel = XGBRegressor()
    
    params = {'booster':['gbtree'], 
              'colsample_bytree':[0.8], 
              'gamma':[0], 
              'learning_rate':[0.1], 
              'max_depth':[3], 
              'min_child_weight':[1], 
              'n_estimators':[100], 
              'objective':['reg:linear'], 
              'reg_alpha':[0.0], 
              'silent':[True], 
              'subsample':[1]}

    grid_solver = GridSearchCV(estimator = XGmodel,
                               param_grid = params, 
                               scoring = 'neg_mean_squared_error',
                               cv = 5,
                               n_jobs = -1,
                               refit = 'neg_mean_squared_error',
                               verbose = 1)


    model_result_xgboost = grid_solver.fit(x_train, y_train)
    
    y_pred_train = model_result_xgboost.predict(x_train)
    y_pred_test = model_result_xgboost.predict(x_test)
    
    error_train = sqrt(mean_squared_error(y_train, y_pred_train))
    error_test = sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("Train RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_pred_train)))
    print("Test RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_pred_test)))

    return error_train, error_test, y_pred_train, y_pred_test

if __name__ == '__main__':
    XGB_regressor(x_train, x_test, y_train, y_test)

