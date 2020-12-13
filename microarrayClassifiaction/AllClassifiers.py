 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:05:34 2020

@author: Olga
"""
import os
from microarrayClassifiaction.Classifier import Classifier
'''

Train all of the classifiers and save results

'''

class AllClassifiers():
    
    def __init__(self, cfg_params): 
        
        self.cfg_params = cfg_params
        self.GS_filename = cfg_params.outputFile
        self.CV_filename = cfg_params.outputFile2
        self.RS_filename = cfg_params.outputFile3
        self.CV2_filename = cfg_params.outputFile4
        self.model_name = cfg_params.model_name
        self.folder_path = cfg_params.folder_path    
        self.folder_name = cfg_params.folder_name
        self.microROC = cfg_params.microROC
        self.macroROC = cfg_params.macroROC
        self.classesROC = cfg_params.classesROC
        self.grid = cfg_params.grid
        self.random = cfg_params.random

        
    def createFile(self, filename):
        
        #Create files to save results
        fullpath = os.path.join(str(self.folder_path), self.folder_name, filename)
        out_file = open(fullpath, 'w')

        return out_file
    
    def trainGridSearch(self, X_std,features, y, el, save=True):
            
        #Training
        self.best_params, self.means, self.stds, self.params, best_estimator = self.clf.gridSearch(X_std, y, self.param)

        #Cross-Validation Testing
        y_pred, auc = self.clf.crossValidation(features, y, best_estimator)
           
        #Plot and save ROC Curve
        roc_auc = self.clf.plotMulticlassRoc(y, y_pred, 'GS', micro=eval(self.microROC), macro=eval(self.macroROC), each_class=eval(self.classesROC))
            
        if save:
            self.saveResult(self.out_file, self.out_file2, el, roc_auc)
    
    def trainRandomSearch(self, X_std, features, y, el, save=True):
            
        #Training
        self.best_params, self.means, self.stds, self.params, best_estimator = self.clf.randomSearch(X_std, y, self.rs_param)
            
        #Cross-Validation Testing
        y_pred, auc = self.clf.crossValidation(features, y, best_estimator)

        #Plot and save ROC Curve
        roc_auc = self.clf.plotMulticlassRoc(y, y_pred, 'RS', micro=eval(self.microROC), macro=eval(self.macroROC), each_class=eval(self.classesROC))
    
        if save:
            self.saveResult(self.out_file3, self.out_file4, el, roc_auc)
            
          
    def performAllModels(self, X_std, features, y, gridSearch, randomSearch):
        
        #Perform all classifiers, save results ang figures
        
        os.mkdir(os.path.join(str(self.folder_path), self.folder_name))
        
        self.out_file = self.createFile(self.GS_filename)
        self.out_file2 = self.createFile(self.CV_filename)
        self.out_file3 = self.createFile(self.RS_filename)
        self.out_file4 = self.createFile(self.CV2_filename)
        
        
        for el in range(len(self.model_name)):
            
            model = self.cfg_params.getModel(self.model_name[el])
            self.param, self.rs_param = self.cfg_params.getDict(self.model_name[el])
            self.clf = Classifier(model, self.model_name[el], self.cfg_params)
            
            #Train GS, Cross-Validation, Plot and save ROC Curve
            if gridSearch:
                self.trainGridSearch(X_std, features, y, el, save = True)
             
            #Train RS, Cross-Validation, Plot and save ROC Curve
            if randomSearch:
                self.trainRandomSearch(X_std, features, y, el, save = True)
        
        self.clf.plotPCA(X_std, y)
        
        self.out_file.close()
        self.out_file2.close()        
        self.out_file3.close()
        self.out_file4.close()
        
        
    def saveResult(self, out_file, out_file2, el, roc_auc):
          
             
        out_file.write("%s Classifier\n" %self.model_name[el] + 
                       "Best parameters found on training set:\n%s\n" %str(self.best_params)) 
     
        out_file.write('\nTested parameters:\n\nMean\tStd (+/-)\tParameters\n' )

        self.means.astype(str)
        self.stds.astype(str)
            
        c = [self.means, self.stds*2, self.params]
            
        for i in zip(*c):
            out_file.write("{0:.3f}\t{1:.3f}\t{2}\n".format(*i))
        out_file.write('\n')
        #print(roc_auc)  
        
        
        
        
        out_file2.write("%s Classifier\n" %self.model_name[el] + "\nBest parameters: %s" %str(self.best_params) + 
                    ('\nAvarage AUC score: %0.3f (+/- %0.3f)\n\n' % (roc_auc.mean(), roc_auc.std() * 2) ))

        

                
  