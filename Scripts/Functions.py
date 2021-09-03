#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from math import sqrt
import pickle
import glob


## ==================FUNCTIONS========================== 

def LoadData(path):
    files = sorted(glob.glob(path + '/Data/*.csv'))
    read_files = []
    for file in files:
        temp = pd.read_csv(file)
        read_files.append(temp)
    
    dt = pd.concat(read_files, ignore_index=True)
    dt = dt.drop('Unnamed: 0', axis=1)
    
    dt.loc[dt['Length of Stay'] == '120 +', 'Length of Stay'] = '120'
    dt['Length of Stay'] = dt['Length of Stay'].astype(int)
    
    dt['APR Risk of Mortality'] = dt.loc[dt['APR Risk of Mortality'].notnull(), :] 
    
    dt['Type of Admission'].replace('Urgent', 'Emergency', inplace=True)
    dt = dt.loc[dt['Type of Admission'] != 'Not Available', :]
    
    dt.drop_duplicates(keep='first', inplace=True)
    dt.to_csv(path + '/Outputs/Inpatient_Data_Clean.csv')
    return dt


def plots(path, dt, target, ordinals, dummies):
    sns.distplot(dt[target]) 
    plt.savefig(path + '/Plots/LoS_distribution.png')
    plt.show()
    cols = ordinals + dummies   
    for col in dt[cols]:
        sns.barplot(dt[col], dt[target])
        plt.xticks(rotation=90)
        plt.ylabel('Mean Length of Stay')
        plt.savefig(path + '/Plots/' + col + '.png')
        plt.show()
    

def ordEncoding(dt, column):
    dt = dt.copy()
    if column == 'Age Group':
        cats = sorted(list(dt[column].unique()))
    else:
        cats = list(dt[column].unique())
    number = 0
    map_dict = {}
    for cat in cats:
        map_dict[cat] = number
        number += 1
    return dt[column].map(map_dict)    


def CreateMasterTable(dt, ordinals, dummies, target, path):
    df = dt.copy()
    for col in ordinals:
        df[col] = ordEncoding(df, col)
    cols = ordinals + dummies + [target] 
    df = df.loc[:,cols]
    df = pd.get_dummies(df, columns=dummies, drop_first=True)
    df.to_csv(path + '/Outputs/MasterTable.csv') 
    return df
        

def CreateXY(data2, target):
    x = data2.loc[:, :].drop(target, axis=1)
    y = data2.loc[:, target]
    y = np.log(data2.loc[:, target])
    return x, y


def simple_model(x, y, path):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15, test_size=0.2)
    GBM = LGBMRegressor()
    GBM.fit(x_train, y_train)
    y_pred_train = GBM.predict(x_train)
    y_pred_test = GBM.predict(x_test)
    error_train = np.exp(sqrt(mean_squared_error(y_train, y_pred_train)))
    error_test = np.exp(sqrt(mean_squared_error(y_test, y_pred_test)))
    sns.distplot(np.exp(y_pred_test))
    sns.distplot(np.exp(y_test))
    plt.legend(['yhat', 'y'])
    plt.savefig(path + '/Plots/simple_predictions_distplot.png')
    file = open(path + '/Models/LGBM_simple', 'wb')
    pickle.dump(GBM, file)
    return error_train, error_test, x_test, y_test, y_pred_test


def TunedModel(x, y, path):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15, test_size=0.2)
    
    GBM = LGBMRegressor()
    
    params = {'boosting_type':['gbdt'], 
              'class_weight':[None], 
              'colsample_bytree':[0.8],
              'importance_type':['split'], 
              'learning_rate':[0.1, 0.05], 
              'max_depth':[5, 8],
              'min_child_samples':[20], 
              'min_child_weight':[0.001], 
              'min_split_gain':[0.0],
              'n_estimators':[100, 500, 800], 
              'n_jobs':[-1], 
              'num_leaves':[70, 80], 
              'objective':[None],
              'random_state':[None], 
              'reg_alpha':[0.0], 
              'reg_lambda':[0.0], 
              'silent':[True],
              'subsample':[1.0], 
              'subsample_freq':[0]}
    
    grid_solver = GridSearchCV(estimator = GBM,
                               param_grid = params, 
                               scoring = 'neg_mean_squared_error',
                               cv = 5,
                               n_jobs = -1,
                               refit = 'neg_mean_squared_error',
                               verbose = 1)


    model = grid_solver.fit(x_train, y_train)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    error_train = np.exp(sqrt(mean_squared_error(y_train, y_pred_train)))
    error_test = np.exp(sqrt(mean_squared_error(y_test, y_pred_test)))
    
    sns.distplot(np.exp(y_pred_test))
    sns.distplot(np.exp(y_test))
    plt.legend(['yhat', 'y'])
    plt.savefig(path + '/Plots/tuned_predictions_distplot.png')
    
    file = open(path + '/Models/LGBM_tuned', 'wb')
    pickle.dump(GBM, file)
    return error_train, error_test, x_test, y_test, y_pred_test

    
# def resultsAnalysis(path, x_test, y_test, y_pred_test):
#     data = x_test.copy()
#     data['LoS'] = [np.exp(x).round() for x in y_test]  
#     data['LoS_pred'] = [np.exp(x).round() for x in y_pred_test] 
#     data['LoS_error'] = ((data['LoS'] - data['LoS_pred'])**2)**(1/2)
#     data['stay_length_group'] = pd.cut(data['LoS'], bins=[-1,5,10,15,20,120,], labels=['0-5', '6-10', '11-15', '16-20', '>20'])
#     data['group_RMSE'] = 0
#     for stay in data['stay_length_group'].unique():
#         group = data.loc[data['stay_length_group'] == stay, :].copy()
#         data.loc[data['stay_length_group'] == stay, 'group_RMSE'] = round(sqrt(mean_squared_error(group['LoS'], group['LoS_pred'])), 3)
#     sns.barplot(data['stay_length_group'].unique(), data['group_RMSE'].unique())
#     plt.xlabel('Length of Stay')
#     plt.ylabel('RMSE')
#     rmse_list = sorted(list(data['group_RMSE'].astype(float).unique()))
#     for i, v in enumerate(rmse_list):
#         plt.text(i - 0.2, v + 0.2, str(v))
#     plt.savefig(path + '/Outputs/RMSE_by_LoS_group.png')
#     plt.show()
    
def resultsAnalysis(path, x_test, y_test, y_pred_test):
    data = x_test.copy()
    data['LoS'] = np.exp(y_test)
    data['LoS_pred'] = np.exp(y_pred_test)
    data['LoS_error'] = ((data['LoS'] - data['LoS_pred'])**2)**(1/2)
    data['stay_length_group'] = pd.cut(data['LoS'], bins=[-1,1,2,5,10,15,20,120], labels=['1', '2', '3-5', '6-10', '11-15', '16-20', '>20'])
    data['group_RMSE'] = 0
    for stay in data['stay_length_group'].unique():
        group = data.loc[data['stay_length_group'] == stay, :].copy()
        data.loc[data['stay_length_group'] == stay, 'group_RMSE'] = round(sqrt(mean_squared_error(group['LoS'], group['LoS_pred'])), 3)
    sns.barplot(data['stay_length_group'].unique(), data['group_RMSE'].unique())
    plt.xlabel('Length of Stay')
    plt.ylabel('RMSE')
    rmse_list = sorted(list(data['group_RMSE'].astype(float).unique()))
    for i, v in enumerate(rmse_list):
        plt.text(i - 0.2, v + 0.2, str(v))
    plt.savefig(path + '/Plots/RMSE_by_LoS_group.png')
    plt.show()

if __name__ == '__main__':
    LoadData(path)
    plots(path, dt, target, ordinals, dummies)
    ordEncoding(dt, column)
    CreateMasterTable(dt, ordinals, dummies, target, path)
    CreateXY(data2, target)
    simple_model(x, y, path)
    TunedModel(x, y, path)
    resultsAnalysis(x_test, y_test, y_pred_test)


