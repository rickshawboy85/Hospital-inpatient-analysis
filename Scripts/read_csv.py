#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:38:44 2020

@author: samthomas
"""

import pandas as pd
import glob


def read_data(path):
    
    print('\nReading file...\n')
    
    files = glob.glob(path + '/Data/*.csv')
    
    read_files = []

    for file in files:
    
        small_frame = pd.read_csv(file)
        read_files.append(small_frame)
    
    dataframe = pd.concat(read_files)
    
    dataframe = dataframe.drop('Unnamed: 0', axis=1)
    
    print('\nReading file complete.')
    
    return dataframe

if __name__ == '__main__':
    read_data(path)
    
    
