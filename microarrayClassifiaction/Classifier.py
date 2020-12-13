from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
import os
"""
Train and test classifier 

"""

class Classifier:

    def __init__(self, model, model_name, cfg_params):
        
        self.clf = model
        self.model_name = model_name
        self.folder_path = cfg_params.folder_path    
        self.folder_name = cfg_params.folder_name
        self.n_iter=cfg_params.n_iter
        self.cv_folds=cfg_params.cv_folds

    

    def multiclassAucScore(self, y_test, y_pred, avarage = 'micro'):
        
        #Make AUC scorer for GridSearch and CrossValidation in multiclass classification
            classes = pd.unique(y_test[::])
            y_test = label_binarize(y_test, classes)
            #n_classes = y.shape[1] #number of classes
            #Binarize the output
            y_pred = label_binarize(y_pred, classes)
            #y_pred = lb.transform(y_pred)
                    
            return roc_auc_score(y_test, y_pred, avarage, multi_class='ovr') 
        

    def gridSearch(self, X, y, d_params):

        #Initialize the grid serach object 
        
        grid_search = GridSearchCV(self.clf, 
                                   param_grid = d_params, 
                                   cv = self.cv_folds,
                                   scoring = make_scorer(self.multiclassAucScore))
        
        grid_search.fit(X, y)
        
        #Update the classifier with the best estimator
        best_estimatorGS = grid_search.best_estimator_
        
        self.means = grid_search.cv_results_['mean_test_score']
        self.stds = grid_search.cv_results_['std_test_score']
        self.params = grid_search.cv_results_['params']
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        if self.model_name == 'LinearSVM':
            self.gridSearchPlot('param_estimator__C', 'Cost')
        elif self.model_name == 'KNN':
            self.gridSearchPlot('param_n_neighbors', 'Liczba sąsiadów')
        elif self.model_name == 'SVM':
            self.gridSearchHeatmap('param_estimator__C', 'param_estimator__gamma', 'Gamma', 'C')
        elif self.model_name =='DecisionTree':
            self.gridSearchHeatmap('param_max_depth', 'param_criterion', 'Kryterium', 'Wysokość drzewa')
        elif self.model_name =='RandomForest':
            self.gridSearchHeatmap('param_n_estimators', 'param_max_features', 'Liczba cech', 'Liczba drzew')


        #Print GridSearch results
        self.printParametersSearch()

        return self.best_params, self.means, self.stds, self.params, best_estimatorGS

    def gridSearchPlot(self, param_name, x_label_name):
        
        plt.figure(figsize = (8,8))
        #plt.title("Grid Search AUC Scores with %s" %(self.model_name), fontsize = 18)
        plt.xlabel(x_label_name, fontsize = 15)
        plt.ylabel("AUC", fontsize = 15)
        plt.plot(self.cv_results[param_name], self.means, label='%s Classifier'%(self.model_name), linewidth=3)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(str(self.folder_path), self.folder_name, 'gridSearchPlot_%s.png' %(self.model_name)))
        plt.show()
              
   
    def gridSearchHeatmap(self, firstparam, secondparam, xLabelName, yLabelName):
        
        plt.figure(figsize = (8,8))
        table = pd.pivot_table(pd.DataFrame(self.cv_results), 
                                 values='mean_test_score', index=firstparam, 
                                 columns=secondparam)
        sns.heatmap(table, vmin=table.values.min(), vmax=table.values.max())

        #sns.heatmap(table, center=midpoint, vmin=table.values.min(), vmax=table.values.max())
        plt.xlabel(xLabelName, fontsize = 15)
        plt.ylabel(yLabelName, fontsize = 15)
        #plt.title('AUC Score', fontsize = 18)
        plt.savefig(os.path.join(str(self.folder_path), self.folder_name, 'OptimizationHeatmap_%s.png' %(self.model_name)))

     
    
    def randomSearch(self, X, y, d_params):

        #Initialize the grid serach object 
        random_search = RandomizedSearchCV(self.clf, 
                                           param_distributions = d_params, 
                                           n_iter = self.n_iter, 
                                           cv = self.cv_folds,
                                           scoring = make_scorer(self.multiclassAucScore))
        random_search.fit(X, y)
        
        #Update the classifier with the best estimator
        best_estimatorRS = random_search.best_estimator_
        
        self.means = random_search.cv_results_['mean_test_score']
        self.stds = random_search.cv_results_['std_test_score']
        self.params = random_search.cv_results_['params']
        self.best_params = random_search.best_params_
        #cv_results = random_search.cv_results_                           
        #Print RandomSearch results
        self.printParametersSearch()

        return self.best_params, self.means, self.stds, self.params, best_estimatorRS
            
    def printParametersSearch(self):
       
        #Print GridSearch results
        print(self.model_name,'Classifier\n')
        print('Best parameters found on training set:')
        print(self.best_params)
        print('\nTested parameters:\n')
                    
        for mean, std, param in zip(self.means, self.stds, self.params):
            print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, param))
        print('\n') 

    def crossValidation(self, X, y, best_estimator):

        
        print('\nThe model is trained on the full dataset.')
        print('The scores are computed on the full dataset.\n')
       
        #Testing classifier  
        y_pred = cross_val_predict(best_estimator, X, y, cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True), method = 'predict_proba')
        auc = cross_val_score(best_estimator, X, y, cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True), scoring = make_scorer(self.multiclassAucScore))
        #y_pred = best_estimator.predict_proba(X)
        #print(y_pred)
        # cv_report = classification_report(y, y_pred)
        #Print Cross Validation results
        self.printCrossValidation(auc)
        
        return y_pred, auc

    def printCrossValidation(self, auc):
        
        #Print Cross Validation results
        print(self.model_name, 'Classifier\n')
        #print('Classification report: \n')
        #print(cv_report)
        print('AUC scores during each CV fold: %s' %(auc))
        print('\nMean AUC: %0.3f (+/- %0.3f)\n\n' % (auc.mean(), auc.std() * 2) )

        
    def plotMulticlassRoc(self, y, y_pred, name, micro, macro, each_class):    
       
        classes = pd.unique(y[::])
        y = label_binarize(y, classes)
        n_classes = y.shape[1] #number of classes
        #Binarize the output
        #y_pred = label_binarize(y_pred, classes)
        #y_pred=y_pred[:, 1]
        #ROC Curve    
        linewidth = 2    
    
        #Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        #Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        #First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        #Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        #Finally average it and compute AUC
        mean_tpr = mean_tpr/n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr

        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        #print('RAUC', roc_auc['macro'])

        #Plot ROC curves
        plt.figure(figsize = (8,8))
        
        #Plot micro ROC
        if micro:
            plt.plot(fpr['micro'], tpr['micro'],
                     label='micro-average ROC curve (AUC = {0:0.3f})'
                     ''.format(roc_auc["micro"]),
                color='navy', linewidth=3)
     
        #Plot macro ROC
        if macro:
            plt.plot(fpr['macro'], tpr['macro'],
                     label='macro-average ROC curve (AUC = {0:0.3f})'
                     ''.format(roc_auc["macro"]),
                 color='deeppink', linewidth=3)
            
        #Plot ROC curve for each and every class    
        if each_class:
                
            colors = cycle(['steelblue', 'darkorange', 'red', 'yellow', 'lime', 'darkviolet'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, linewidth=linewidth,
                         label='ROC curve of class {0} (AUC = {1:0.3f})'
                         ''.format(i+1, roc_auc[i]))
                            
        plt.plot([0, 1], [0, 1], 'k--', linewidth=linewidth)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize = 15)
        plt.ylabel('TPR', fontsize = 15)
        #plt.title('Receiver operating characteristic for %s Classifier' %self.model_name, fontsize = 18)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(str(self.folder_path), self.folder_name, '%sroc_auc_plot_for_%s.png' %(name, self.model_name)))
        plt.show()
        
        return roc_auc['micro'] 
        
    def plotPCA(self, X, y):
        
        classes = pd.unique(y[::])
        n_classes = len(classes)
        colors = ['red','steelblue', 'yellow', 'darkviolet', 'lime', 'darkorange']
        
        # 2 Components PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit(X).transform(X)
        
        #Join features after PCA with labels
        X_pca = pd.DataFrame(data = X_pca
             ,columns = ['principal component 1', 'principal component 2'])
        y_pca = pd.DataFrame(data=y, columns=['labels'])
        X_pca = pd.concat([X_pca, y_pca], axis = 1)
        
        #Plot PCA in first 2 directions
        plt.figure(figsize = (8,8))
        for i, color in zip(range(n_classes), colors):
            
            labels = X_pca['labels'] == i
            plt.scatter(X_pca.loc[labels, 'principal component 1']
                       , X_pca.loc[labels, 'principal component 2']
                       , c = color
                       , s = 50)
            
        plt.legend([i for i in range(n_classes)])
        #plt.title('First two PCA directions', fontsize = 18)
        plt.xlabel('Pierwsza składowa główna', fontsize = 15)
        plt.ylabel('Druga składowa główna', fontsize = 15)
        plt.grid(True)
        plt.savefig(os.path.join(str(self.folder_path), self.folder_name, 'PCA_2directrions.png'))
        plt.show()
        

    
