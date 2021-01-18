#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:36:26 2020

@author: samthomas
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from math import sqrt


def GBM_regressor(x_train, x_test, y_train, y_test):

    GBMlight = LGBMRegressor()

    params = {
              'boosting_type':['gbdt'],
              'colsample_bytree':[0.8],
              'learning_rate':[0.1],
              'n_estimators':[1000],
              'reg_lambda':[0.01],
              'subsample':[0.7],
              'min_child_samples':[30],
              'max_depth':[-1, 3]
             }    
    

   
    grid_solver = GridSearchCV(estimator = GBMlight,
                               param_grid = params, 
                               scoring = 'neg_mean_squared_error',
                               cv = 5,
                               n_jobs = -1,
                               refit = 'neg_mean_squared_error',
                               verbose = 1)


    model_result_GBMlight = grid_solver.fit(x_train, y_train)
    

    y_pred_train = model_result_GBMlight.predict(x_train)
    y_pred_test = model_result_GBMlight.predict(x_test)
    
    error_train = sqrt(mean_squared_error(y_train, y_pred_train))
    error_test = sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("Train RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_pred_train)))
    print("Test RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_pred_test)))


    return error_train, error_train, y_pred_train, y_pred_test

if __name__ == '__main__':
    GBM_regressor(x_train, x_test, y_train, y_test)



def GBM_regressor2(x_train, x_test, y_train, y_test):
    
    GBMlight2 = LGBMRegressor()
    
    GBMlight2.fit(x_train, y_train)
    
    yhat_train = GBMlight2.predict(x_train)
    yhat_test = GBMlight2.predict(x_test)
    
    error_train = sqrt(mean_squared_error(y_train, yhat_train))
    error_test = sqrt(mean_squared_error(y_test, yhat_test))
    
    print("Train RMSE: %.2f" % sqrt(mean_squared_error(y_train, yhat_train)))
    print("Test RMSE: %.2f" % sqrt(mean_squared_error(y_test, yhat_test)))
    
    return error_train, error_test, yhat_train, yhat_test

if __name__ == '__main__':
    GBM_regressor2(x_train, x_test, y_train, y_test)