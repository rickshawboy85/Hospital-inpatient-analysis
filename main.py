#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:19:23 2020

@author: samthomas
"""

import Functions as fn
import glob
from math import sqrt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
path = '/Users/samthomas/Desktop/LoS'
os.chdir(path + '/Scripts/')


# ==================PARAMS==========================

target = 'Length of Stay'
ordinals = ['Age Group', 'APR Severity of Illness Description',
            'APR Risk of Mortality']
dummies = ['APR MDC Description', 'Type of Admission', 'Gender', 'CCS Procedure Description',
           'APR DRG Description', 'CCS Diagnosis Code', 'Facility Name', 'Emergency Department Indicator']


# ==================FUNCTIONS==========================

# load and clean data
data = fn.LoadData(path)
fn.plots(path, data, target, ordinals, dummies)

# transformations done in CreateMasterTable
data2 = fn.CreateMasterTable(data, ordinals, dummies, target, path)
x, y = fn.CreateXY(data2, target)

# deploy lgbm regressors (untuned params and tuned params)
RMSE_tr, RMSE_te, x_test, y_test, y_pred_test = fn.simple_model(x, y, path)
RMSE_tr_tuned, RMSE_te_tuned, x_test_tuned, y_test_tuned, y_pred_test_tuned = fn.TunedModel(
    x, y)

# splits predictions into groups and give rmse per group
fn.resultsAnalysis(path, x_test, y_test, y_pred_test)


# I am adding a change
