# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:25:34 2020

@author: Olga
"""
import warnings
from microarrayClassifiaction.Data import Data
from microarrayClassifiaction.AllClassifiers import AllClassifiers
from microarrayClassifiaction.ConfigurationParams import ConfigurationParams
from microarrayClassifiaction.DataPreprocessing import DataPreprocessing
warnings.filterwarnings("ignore") 

def main():
    
    cfg_params = ConfigurationParams('config.cfg')
    features, labels = Data(cfg_params.inputFile).readData()
    
    data = DataPreprocessing(features, labels)
    X, y = data.getStandarizedData()
    data.plotHeatMap()
    
    def performModels(PCA):
        
        if PCA:
            
            #Train all of the classifiers on data after PCA and save results
            X_pca = data.makePCA()
            AllClassifiers(cfg_params).performAllModels(X_pca, features, y, 
                          gridSearch=eval(cfg_params.grid), randomSearch=eval(cfg_params.random))
            
        else:
            #Train all of the classifiers and save results
            AllClassifiers(cfg_params).performAllModels(X, features, y, 
                          gridSearch=eval(cfg_params.grid), randomSearch=eval(cfg_params.random))
            
    performModels(PCA = eval(cfg_params.ifPCA))

if __name__ == "__main__":
    main()