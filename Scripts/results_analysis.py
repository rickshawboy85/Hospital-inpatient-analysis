#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:35:54 2021

@author: samthomas
"""

import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error 
import seaborn as sns
import matplotlib.pyplot as plt


##==================================================================================
##==================================================================================

def plot_prep(x_train, y_train, yhat_train):
    
    data = x_train
    
    data['length_of_stay'] = y_train  
    
    data['predicted_LoS'] = yhat_train
    
    data['LoS_error'] = ((data['length_of_stay'] - data['predicted_LoS'])**2)**(1/2)
    
    # create groups according their length of stay
    data['stay_length_group'] = pd.cut(data['length_of_stay'], bins=[-1,5,10,15,20,120,], labels=['0-5', '6-10', '11-15', '16-20', '>20'])
    
    ### create dataframes for all the stay length groups
    stay_0_5 = data[data['stay_length_group'] == '0-5']
    stay_6_10 = data[data['stay_length_group'] == '6-10']
    stay_11_15 = data[data['stay_length_group'] == '11-15']
    stay_16_20 = data[data['stay_length_group'] == '16-20']
    stay_21_ = data[data['stay_length_group'] == '>20']
    
    
    ## create a column with RMSE values according to length of stay group
    data['RMSE_per_LoS_group'] = np.where(data['stay_length_group'] == '0-5', sqrt(mean_squared_error(stay_0_5['length_of_stay'], stay_0_5['predicted_LoS'])), '')
    
    data['RMSE_per_LoS_group'] = np.where(data['stay_length_group'] == '6-10', sqrt(mean_squared_error(stay_6_10['length_of_stay'], stay_6_10['predicted_LoS'])), data['RMSE_per_LoS_group'])
    
    data['RMSE_per_LoS_group'] = np.where(data['stay_length_group'] == '11-15', sqrt(mean_squared_error(stay_11_15['length_of_stay'], stay_11_15['predicted_LoS'])), data['RMSE_per_LoS_group'])
    
    data['RMSE_per_LoS_group'] = np.where(data['stay_length_group'] == '16-20', sqrt(mean_squared_error(stay_16_20['length_of_stay'], stay_16_20['predicted_LoS'])), data['RMSE_per_LoS_group'])
    
    data['RMSE_per_LoS_group'] = np.where(data['stay_length_group'] == '>20', sqrt(mean_squared_error(stay_21_['length_of_stay'], stay_21_['predicted_LoS'])), data['RMSE_per_LoS_group'])
    
    data['RMSE_per_LoS_group'] = data['RMSE_per_LoS_group'].astype(float).round(3)
    
    return data

if __name__ == '__main__':
    plot_prep(x_train, y_train, yhat_train)
    

##==================================================================================    
##==================================================================================


def RMSE_plot(data):
    
                            
    sns.barplot(data['stay_length_group'].unique(), 
            data['RMSE_per_LoS_group'].unique())
    
    plt.xlabel('LoS Group')
    plt.ylabel('RMSE')
        
    rmse_list = list(data['RMSE_per_LoS_group'].astype(float).unique())
    
    temp  = rmse_list[2]
    rmse_list.pop(2)
    rmse_list.append(temp)
    
    for i, v in enumerate(rmse_list):
        plt.text(i - 0.2, v + 0.2, str(v))
    
    plt.show()
    
if __name__ == '__main__':
    RMSE_plot(data)
    

##==================================================================================
##==================================================================================


def countplot(data):
    
    
    counts = list(data['stay_length_group'].value_counts())
    temp = counts[3]
    counts.pop(3)
    counts.append(temp)
    
    sns.countplot(data['stay_length_group'])
    plt.xlabel('LoS Group')
    plt.ylabel('Number of Patients')
    
    for i, v in enumerate(counts):
        plt.text(i - 0.3, v + 10000, str(v))
    
    plt.show()
    
if __name__ == '__main__':
    countplot(data)   
    


##==================================================================================
##==================================================================================

def pieplot(data):
    
    counts = list(data['stay_length_group'].value_counts())
    temp = counts[3]
    counts.pop(3)
    counts.append(temp)
    
    labels = list(data['stay_length_group'].unique())
    temp2 = labels[2]
    labels.pop(2)
    labels.append(temp2)
    
    
    plt.pie(counts, autopct='%1.1f%%')
    plt.title('Group Proportions')
    plt.legend([(str(x) + ' days') for x in labels],
               title = "Stay length group",
               loc = "upper left",
               bbox_to_anchor = (1, 0, 0.5, 1))
    plt.show()
    
if __name__ == '__main__':
    pieplot(data)   