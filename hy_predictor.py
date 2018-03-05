#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from Liblinear.liblinear import *
from Liblinear.liblinearutil import *


class HYPredictor(object):
    
    def __init__(self, k, patient_info, train_data, resultpath):
        self.k = k
        self.param = dict()
        self.patient_info = patient_info
        self.data = train_data
        self.resultpath = resultpath
        # generate labels and samples for training data
        self.y, self.X = self.get_label()
        self.options = '-s 5' # configuration: 
                              # L1-regularized support vector classification
     
    def train_model(self):
        print(np.array(self.X).shape)
        self.model = train(self.y, self.X, self.options)
        # predict labels and scores
        p_labs, p_acc, p_vals = predict(self.y, self.X, self.model)
        return (self.model, p_labs)

        
    def save_model(self):
        # save model
        save_model(self.resultpath + 'hy_pred_model_file_' + str(self.k) + '.txt', model)
        
        
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
        for td in data:
            pid = td[0]
            sample = td[1]
            sample.append(0.1+eps*int(pid)) # make all weights valued
            y.append(patient_info[pid].hy_stage)
            X.append(sample)
           
        return (y, X)
        
        