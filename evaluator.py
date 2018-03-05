#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from liblinear import *
from liblinearutil import *

import numpy as np
from sklearn import metrics
from sklearn import linear_model

class Evaluator:

    def __init__(self, model=None, test_data=None, patient_info=None, task=None, predictor=None):
        self.model = model
        self.data = test_data
        self.patient_info = patient_info
        self.task = task
        self.predictor = predictor
        
             
    def compute_AMI(self, labels_true, labels_pred):
        return metrics.adjusted_mutual_info_score(labels_true, labels_pred) 

        
    def compute_accuracy(self):
        if self.task == 'PD prediction':
            auc, ap = self.compute_pd_accuracy()
            return (auc, ap)
        elif self.task == 'H&Y stage prediction': 
            acc = self.compute_hy_accuracy()
            return acc
        elif self.task == 'MoCA prediction': 
            rmse= self.compute_moca_accuracy()
            return rmse
        else:
            print ("Unexpected error: choose a correct task.")
        
            
    def compute_pd_accuracy(self):

        y, X = self.predictor.get_label(self.data, self.patient_info)
        p_labs, p_acc, p_vals = predict(y, X, self.model, '-q')
        # compute auc 
        fpr, tpr, thresholds = metrics.roc_curve(y, p_labs, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # compute ap (average precision)
        ap = self.compute_ap(y, p_labs)
#       ap = metrics.average_precision_score(y, p_labs)
        return (auc, ap)
        
        
    def compute_hy_accuracy(self):
        y, X = self.predictor.get_label(self.data, self.patient_info)
        p_labs, p_acc, p_vals = predict(y, X, self.model, '-q')
        # compute accuracy 
        acc = metrics.accuracy_score(y, p_labs)
#        precision = metrics.precision_score(y, p_labs, average='micro')
#        recall = metrics.recall_score(y, p_labs, average='micro')
#        f1 = metrics.f1_score(y, p_labs, average='micro')  
        return acc
        
        
    def compute_moca_accuracy(self):
        y, X = self.predictor.get_label(self.data, self.patient_info)
        y_pred = self.predictor.regr.predict(X)
        # comptue mse
        rmse = np.sqrt(np.mean((y_pred - y) ** 2)/len(y))
        return rmse
    
    
    def compute_ap(self, y, p_labs):
        y = np.array(y, dtype = 'int')
        pred_y = np.array(p_labs, dtype = 'int')
        ap = (y == pred_y).sum()/y.size
        return ap
    
        