# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:31:00 2020

@author: Olga
"""
import os
import csv
import pandas as pd

'''
Return dataset X - the matrix of features, Y - the vector of class labels
inputFile = 'Lymphoma_500.txt'

'''
class Data:
    
    def __init__(self, inputFile):
        
        self.inputFile = inputFile
        
    def readData(self):
        
        outputFile = os.path.splitext(self.inputFile)[0] + '.csv'
        
        with open(self.inputFile, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split('\t') for line in stripped if line)
            with open(outputFile, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)
                
                dataset = pd.read_csv(outputFile)
                #X - features, Y - labels 
                    
            df = dataset.iloc[1::, 1::].values
            
            #Data
            X = df.T
            y = dataset.iloc[0:1, 1:(len(X)+1)].values
            y = y.T
            
            return X, y