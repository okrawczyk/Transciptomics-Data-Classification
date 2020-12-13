# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:39:54 2020

@author: Olga
"""
import os
import scipy
import numpy as np
import configparser
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
Hold all parameters from configuration file 

"""

class ConfigurationParams():
    
    def __init__(self, filename):
        
        self.cfg = configparser.ConfigParser()
        self.cfg.read(filename)
        
        inputFile = self.cfg['FILES']['dataset']
        cv_folds = int(self.cfg['CV']['folds'])
        n_iter = int(self.cfg['Parameters Tuning']['n_iter'])
        outputFile = self.cfg['FILES']['GS_results']
        outputFile2 = self.cfg['FILES']['GS_Validation']
        outputFile3 = self.cfg['FILES']['RS_results']
        outputFile4 = self.cfg['FILES']['RS_Validation']
        model_name = self.cfg['Models']['models'].split(',')
        self.folder_name = self.cfg['FILES']['folder_name']
        ifPCA = self.cfg['Preprocessing']['PCA']
        microROC = self.cfg['ROC']['micro']
        macroROC = self.cfg['ROC']['macro']
        classesROC = self.cfg['ROC']['classes']
        grid = self.cfg['Parameters Tuning']['grid']
        random = self.cfg['Parameters Tuning']['random']


        self.folder_path = os.getcwd()    
        
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.outputFile2 = outputFile2
        self.outputFile3 = outputFile3
        self.outputFile4 = outputFile4
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.model_name = model_name
        self.ifPCA = ifPCA
        self.microROC = microROC
        self.macroROC = macroROC
        self.classesROC = classesROC
        self.grid = grid
        self.random = random

    def getDict(self, n_classifier):
        
        #Return parameters for a specific classifier 
        if n_classifier == "LinearSVM": 
            self.d_param, self.d_rsparam = self._getLSVC()      
        elif n_classifier == "SVM": 
            self.d_param, self.d_rsparam = self._getSVC()
        elif n_classifier == "KNN": 
            self.d_param, self.d_rsparam = self._getKNN()
        elif n_classifier == "DecisionTree": 
            self.d_param, self.d_rsparam = self._getDecisionTree()
        elif n_classifier == "LDA": 
            self.d_param, self.d_rsparam = self._getLDA()
        elif n_classifier == "RandomForest": 
            self.d_param, self.d_rsparam = self._getRandomForest()
        else:
            self.d_param = None 
            self.params_rs = None
                
        return self.d_param, self.d_rsparam
    
    def _getLSVC(self):
        
        #Parameters to be tunned for a Linear SVM 
        params = {'estimator__kernel': self.cfg['SVM']['kernel_2'].split(',')
            ,'estimator__C': np.fromstring(self.cfg['SVM']['c_grid'], dtype=int, sep=',').tolist()
            ,'estimator__probability': [True]}
        
        #Parameters to be tunned for a Linear SVM with Random Search
        params_rs = {'estimator__kernel': self.cfg['SVM']['kernel_2'].split(',')
            ,'estimator__C': scipy.stats.uniform(scale=max(np.fromstring(self.cfg['SVM']['c_grid'], 
                                                                         dtype=int, sep=',').tolist()))
            ,'estimator__probability': [True]}        
  
        return params, params_rs     
    
    def _getSVC(self):
        
        #Parameters to be tunned for a SVM 
            
        params = {'estimator__kernel': self.cfg['SVM']['kernel_1'].split(',')
            ,'estimator__gamma': np.fromstring(self.cfg['SVM']['gamma_grid'], dtype=float, sep=',').tolist()
            ,'estimator__C': np.fromstring(self.cfg['SVM']['c_grid'], dtype=int, sep=',').tolist()
            ,'estimator__probability': [True]}
         
        #Parameters to be tunned for a SVM with Random Search
        params_rs = {'estimator__C': scipy.stats.uniform(scale=max(np.fromstring(self.cfg['SVM']['c_grid'], 
                                                                                 dtype=int, sep=',').tolist()))
            ,'estimator__gamma': scipy.stats.uniform(scale=max(np.fromstring(self.cfg['SVM']['gamma_grid'], 
                                                                             dtype=float, sep=',').tolist()))
            ,'estimator__kernel': self.cfg['SVM']['kernel_1'].split(',')
            ,'estimator__probability': [True]}
             
        return params, params_rs
    

    def _getKNN(self):
        
        #Parameters to be tunned for a KNeighbors
       params = {'n_neighbors': np.fromstring(self.cfg['KNN']['n_neighbors'], dtype=int ,sep=',').tolist()}  
       #Parameters to be tunned for a KNeighbors with Random Search
       params_rs = {'n_neighbors': scipy.stats.randint(min(np.fromstring(self.cfg['KNN']['n_neighbors'], 
                                                                         dtype=int ,sep=',').tolist()),
                                                       max(np.fromstring(self.cfg['KNN']['n_neighbors'], 
                                                                         dtype=int ,sep=',').tolist()))}       
       
       return params, params_rs 
   


    def _getDecisionTree(self):
        
        #Parameters to be tunned for a Decision Tree 
        params = {'criterion': self.cfg['Decision Tree']['criterion_gird'].split(',')
           ,'max_depth': np.fromstring(self.cfg['Decision Tree']['max_depth_grid'], 
                                       dtype=int, sep=',').tolist()
           ,'min_samples_leaf': np.fromstring(self.cfg['Decision Tree']['min_samples_leaf'], 
                                       dtype=int, sep=',').tolist()
           ,'min_samples_split': np.fromstring(self.cfg['Decision Tree']['min_samples_split'], 
                                       dtype=int, sep=',').tolist()}
        
        #Parameters to be tunned for a Decision Tree with Random Search
        params_rs = {'criterion': self.cfg['Decision Tree']['criterion_gird'].split(',')
           ,'max_depth': scipy.stats.randint(min(np.fromstring(self.cfg['Decision Tree']['max_depth_grid'], 
                                                               dtype=int, sep=',').tolist()),
                                             max(np.fromstring(self.cfg['Decision Tree']['max_depth_grid'], 
                                                               dtype=int, sep=',').tolist()))
            ,'min_samples_split': scipy.stats.randint(min(np.fromstring(self.cfg['Decision Tree']['min_samples_split'], 
                                                               dtype=int, sep=',').tolist()),
                                             max(np.fromstring(self.cfg['Decision Tree']['min_samples_split'], 
                                                               dtype=int, sep=',').tolist()))
            ,'min_samples_leaf': scipy.stats.randint(min(np.fromstring(self.cfg['Decision Tree']['min_samples_leaf'], 
                                                               dtype=int, sep=',').tolist()),
                                             max(np.fromstring(self.cfg['Decision Tree']['min_samples_leaf'], 
                                                               dtype=int, sep=',').tolist()))}  


        return params, params_rs
    
    
    def _getLDA(self):
        
        #Parameters to be tunned for a LDA
        params = {}
        params_rs = {}

        return params,params_rs

    def _getRandomForest(self):
        #params={}
        #Parameters to be tunned for a RandomForest 
        params = {'n_estimators': np.fromstring(self.cfg['Random Forest']['n_grid'], dtype=int, sep=',').tolist()
            ,'max_features': self.cfg['Random Forest']['max_features'].split(',')
            ,'criterion': self.cfg['Random Forest']['criterion_grid'].split(',')
            ,'min_samples_leaf': np.fromstring(self.cfg['Random Forest']['min_samples_leaf'], 
                                       dtype=int, sep=',').tolist()
           ,'min_samples_split': np.fromstring(self.cfg['Random Forest']['min_samples_split'], 
                                       dtype=int, sep=',').tolist()}
                
        #Parameters to be tunned for a RandomForest 
        params_rs = {'n_estimators': scipy.stats.randint(min(np.fromstring(self.cfg['Random Forest']['n_grid'], dtype=int, sep=',').tolist()) ,max(np.fromstring(self.cfg['Random Forest']['n_grid'], dtype=int, sep=',').tolist()))
            ,'max_features': self.cfg['Random Forest']['max_features'].split(',')
            #,'max_depth': scipy.stats.randint(min(np.fromstring(self.cfg['Random Forest']['max_depth_grid'], dtype=int, sep=',').tolist()),
            #                                  max(np.fromstring(self.cfg['Random Forest']['max_depth_grid'], dtype=int, sep=',').tolist()))
            ,'criterion': self.cfg['Random Forest']['criterion_grid'].split(',')
            ,'min_samples_leaf': np.fromstring(self.cfg['Random Forest']['min_samples_leaf'], 
                                       dtype=int, sep=',').tolist()
           ,'min_samples_split': np.fromstring(self.cfg['Random Forest']['min_samples_split'], 
                                       dtype=int, sep=',').tolist()}

        return params, params_rs
    

    def getModel(self, model_name):
        
        #Return type of classifier to be perfomed 
        if model_name == "LinearSVM": 
            self.model = OneVsRestClassifier(SVC())
        elif model_name == "SVM": 
            self.model = OneVsRestClassifier(SVC())
        elif model_name == "KNN": 
            self.model = KNeighborsClassifier()
        elif model_name == "DecisionTree": 
            self.model = DecisionTreeClassifier()
        elif model_name == "LDA": 
            self.model = LinearDiscriminantAnalysis()
        elif model_name == "RandomForest": 
            self.model = RandomForestClassifier()
        else:
            self.model = None      
                
        return self.model
        
  
        