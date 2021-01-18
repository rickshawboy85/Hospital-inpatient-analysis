#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:38:44 2020

@author: samthomas
"""

import pandas as pd

def read_data(path):
    
    print('Reading file...\n')
    
    hosp_data = pd.read_csv(path + 'Hospital_Inpatient_Discharges.csv')
    
    print('\nReading file complete.')
    
    return hosp_data

if __name__ == '__main__':
    read_data(path)
