#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:19:23 2020

@author: samthomas
"""

import os
path = '/Users/samthomas/Desktop/NEOLAND/Project'
os.chdir(path + '/Scripts/')

import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from read_csv import read_data
from data_prep import prep_dataset
from train_test import train_test
from EDA_plots import plots
import results_analysis as pa
from LinearRegressor import linear_regressor
from XGB_regressor import XGB_regressor
from xgboost import XGBRegressor
from Light_GBM_regressor import GBM_regressor, GBM_regressor2
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


##=====================================================================================
## load data set
##=====================================================================================

dt = read_data(path)
df = dt.copy()


##=====================================================================================
## Clean data set (impute missings etc.)
##=====================================================================================

datarame = prep_dataset(df)


##=====================================================================================
## Data insppection
##=====================================================================================

# separate target 
X = datarame.drop('Length of Stay', axis=1)
Y = datarame['Length of Stay']

# plot each variable against the target
plots = plots(X, Y)


##=====================================================================================
## Train the models
##=====================================================================================

# convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)


# train test split
x_train, x_test, y_train, y_test = train_test(X, Y)


# linear regression model
RMSE_train, RMSE_test, Y_pred_train, Y_pred_test = linear_regressor(x_train, 
                                                                    x_test, 
                                                                    y_train, 
                                                                    y_test)


# xgb model
RMSE_train2, RMSE_test2, Y_pred_train2, Y_pred_test2 = XGB_regressor(x_train, 
                                                                     x_test, 
                                                                     y_train, 
                                                                     y_test)


# gbm model1
RMSE_train3, RMSE_test3, Y_pred_train3, Y_pred_test3 = GBM_regressor(x_train, 
                                                                     x_test, 
                                                                     y_train, 
                                                                     y_test)


# gbm model2 (no params)
RMSE_train4, RMSE_test4, Y_pred_train4, Y_pred_test4 = GBM_regressor2(x_train, 
                                                                      x_test, 
                                                                      y_train, 
                                                                      y_test)


##=====================================================================================
## Analysis of results
##=====================================================================================


data1 = pa.plot_prep(x_train.copy(), y_train.copy(), Y_pred_train.copy())
pa.RMSE_plot(data1)
pa.countplot(data1)


data2 = pa.plot_prep(x_train.copy(), y_train.copy(), Y_pred_train2.copy())
pa.RMSE_plot(data2)
pa.countplot(data2)


data3 = pa.plot_prep(x_train.copy(), y_train.copy(), Y_pred_train3.copy())
pa.RMSE_plot(data3)
pa.countplot(data3)


data4 = pa.plot_prep(x_train.copy(), y_train.copy(), Y_pred_train4.copy())
pa.RMSE_plot(data4)
pa.countplot(data4)

pa.pieplot(data4)

