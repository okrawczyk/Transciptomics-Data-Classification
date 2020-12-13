# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:32:56 2020

@author: Olga
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
"""
Encode labels and fit the scaler to the feature data

"""

class DataPreprocessing:
    
    def __init__(self, features, labels):
        
        self.features = features
        self.labels = labels 

        
    def getStandarizedData(self):   
        
        #Encoding data
        self.y = LabelEncoder().fit_transform(self.labels)
    
        #Create a scaler object
        sc = StandardScaler()
        #Fit the scaler to the feature data and transform
        self.X_std = sc.fit_transform(self.features)
        
        return self.X_std, self.y

    def makePCA(self):

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(self.X_std)
        print('pca',pca.n_components_)

        return X_pca    

    def plotHeatMap(self):
        
        X=(self.X_std).T
        #Heatmap plot
        plt.figure(figsize = (8,8))
        sns.heatmap(X, square=True)
        plt.title('Heatmap')
        plt.show()
    
