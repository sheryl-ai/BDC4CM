#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from Liblinear.liblinear import *
from Liblinear.liblinearutil import *

class PDPredictor(object):
    
    def __init__(self, k, patient_info, train_data, resultpath):
        self.k = k
        self.param = dict()
        self.resultpath = resultpath
        self.patient_info = patient_info
        self.data = train_data
        # generate labels and samples for training data
        self.y, self.X = self.get_label()
        self.options = '-s 6' # configuration: 
                              # L1-regularized logistic regression
        
       
    def train_model(self):
        # train model
        print(np.array(self.X).shape)
        self.model = train(self.y, self.X, self.options)
        # predict labels and scores
        p_labs, p_acc, p_vals = predict(self.y, self.X, self.model, '-q')
        return (self.model, p_labs)
    
        
    def save_model(self):
        # save model
        save_model(self.resultpath + 'pd_pred_model_file_' + str(self.k) + '.txt', model)
        
        
    def get_param(self):
        # get learned parameters
        param = parameter(self.options)
        param.set_to_default_values()
        param.parse_options(self.options)
        try: 
            self.param['w'], self.param['b'] = self.model.get_decfun()
        except ValueError:
             print('Train Model Firstly')
        self.param['w'] = self.param['w'][:-1]
        print (len(self.param['w']))
        return (self.param['w'], self.param['b'])
        
        
    def get_label(self, data=None, patient_info=None):
        ### data format 
        ### y, x = [1,-1], [[1,0,1], [-1,0,-1]]
        ###
        if data == None:
            data = self.data
        if patient_info == None:
            patient_info = self.patient_info
            
        y = list()
        X = list()
        eps = 0.0000001
        for td in self.data:
            pid = td[0]
            sample = td[1]
            sample.append(0.1+eps*int(pid)) # make all weights valued
            if self.patient_info[pid].diagnosis == '1':
                y.append(1)
                X.append(sample)
            elif self.patient_info[pid].diagnosis == '17': 
                y.append(-1)
                X.append(sample)
            else:
               continue
           
        return (y, X)

