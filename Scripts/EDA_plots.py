#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:00:42 2021

@author: samthomas
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt

##==================================================================================
##==================================================================================


def plots(X, Y):
    
    os.chdir('/Users/samthomas/Desktop/NEOLAND/Project/Plots/')
    
    for variable in X:
        
        
        sns.barplot(X[variable], Y)
        plt.xticks(rotation=90)
        plt.ylabel('Mean Length of Stay')
        
        plt.savefig(variable)
        
        plt.show()
        

if __name__ == '__main__':
    plots(X, Y)
