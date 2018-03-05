#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model


class MoCAPredictor(object):
    
    def __init__(self, k, patient_info, train_data, resultpath):
        self.k = k
        self.param = dict()
        self.patient_info = patient_info
        self.data = train_data
        self.resultpath = resultpath
        # generate labels and samples for training data
        self.y, self.X = self.get_label()
        self.regr = linear_model.LinearRegression()
        
     
    def train_model(self):
        self.model = self.regr.fit(self.X, self.y)   
        y_pred = self.regr.predict(self.X)
        print ("RMSE: ", np.sqrt(np.mean((y_pred - self.y) ** 2)/len(self.y)))
        return (self.model, y_pred)
        
        
    def save_model(self):
        # save model
        f = open(self.resultpath + 'moca_pred_model_file_' + str(self.k) + '.txt')
        f.writelines(self.regr.coef_)
        f.close()
        
        
    def get_param(self):
        # get learned parameters
        try: 
            self.param['w'] = self.regr.coef_
            print (len(self.param['w']))
        except ValueError:
             print('Train Model Firstly')
        return self.param['w']
        
        
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
        for td in data:
            pid = td[0]
            sample = td[1]
            y.append(patient_info[pid].moca)
            X.append(sample)
           
        return (y, X)