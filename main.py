#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import sys
#sys.path.append('/Users/zhangxi/Dropbox/research_Parkinson/Part-I')


import numpy as np

from dataio import DataIO
from concatenation import Concatenation
from imputation import Imputation
from splitter import Splitter

from pd_predictor import PDPredictor
from hy_predictor import HYPredictor
from moca_predictor import MoCAPredictor
from pat_cluster import PatCluster

from evaluator import Evaluator


############################ DATA MANIPULATION ################################
def data_preprocess(params):
    ### Record Concatenation
    dataio = DataIO(params['input_path'], params['result_path'])
    dataio.read_data()
    ctn = Concatenation(dataio)
    patient_info, n_feature = ctn.get_concatenation() # patient id: Patient
                                           # static feature and dynamic feature
                                           # dynamic feature{time:feature_value}
    ### Data Imputation 
    imp_method = 'simple' 
    imp = Imputation(patient_info, n_feature)
    patient_array = imp.get_imputation(imp_method)
    return (dataio, patient_info, patient_array)
    
    
########################## DISEASE PREDICTION ################################# 
def pd_prediction(dataio, patient_info, patient_array, params):
       
    ### Task Setting 
    task = params['task'][0] # disease prediction
    split_method = 'cross-validation'
    kfold = 5 # 5-fold validation
    ### Initialization
    auc = np.zeros(kfold, dtype='float32') # evaluation metrics
    ap = np.zeros(kfold, dtype='float32')
    n_feature = dataio.feature.feature_len
    param_w = np.zeros([n_feature, kfold], dtype='float32')
    print ('-----')
    split = Splitter(task, patient_array, kfold, split_method)
    for k in range(kfold): # each fold, k is the index of test set
        ### Data Splitting
        train_data, test_data = split.get_splitter(k)
        ### Model Training 
        pd_pred = PDPredictor(k, patient_info, train_data, params['result_path'])
        model, y_pred = pd_pred.train_model()
        param_w[:,k], _ = pd_pred.get_param()
        ### Evaluating
        pd_eval = Evaluator(model, test_data, patient_info, task, pd_pred)
        auc[k], ap[k] = pd_eval.compute_accuracy()
    print ('-----')
    print ('AUC of the %s task: %f' %(task, np.sum(auc)/kfold))
    print ('Average Precision of the %s task: %f' %(task, np.sum(ap)/kfold))    
    ### Displaying Feature (selected by prediction model)
    feature = dataio.feature
    feature.get_pred_feature(param_w, kfold, 'pd')  

    
############################### H&Y PREDICTION ################################  
def hy_prediction(dataio, patient_info, patient_array, params):
    ### Task Setting 
    task = params['task'][1]
    split_method = 'ratio'
    ratio = 0.8 # provide the ratio
    krun = 10 # run 5 times then average the result 
    ### Initialization
    acc = np.zeros(krun, dtype='float32') # evaluation metric
    n_feature = dataio.feature.feature_len
    param_w = np.zeros([n_feature, krun], dtype='float32') # weights parameter
    ### H&Y Reading
    feature = dataio.feature
    patient_info = feature.get_hy_stage(patient_info, patient_array)
    print ('-----')
#    split = Splitter(task, patient_array, ratio, split_method, patient_info)
    split = Splitter(task, patient_array, ratio, split_method)
    for k in range(krun):
        ### Data Splitting
        train_data, test_data = split.get_splitter(k)
        ### Model Training 
        hy_pred = HYPredictor(k, patient_info, train_data, params['result_path'])
        model, y_pred = hy_pred.train_model()
        param_w[:,k], _ = hy_pred.get_param()
        ### Evaluating
        hy_eval = Evaluator(model, test_data, patient_info, task, hy_pred)
        acc[k] = hy_eval.compute_accuracy()
    print ('-----')
    print ('Accuracy of the %s task: %f' %(task, np.sum(acc)/krun)) 
    ### Displaying Feature (selected by prediction model)
    feature = dataio.feature
    feature.get_pred_feature(param_w, krun, 'yh')  

    
############################## MoCA PREDICTION ################################
def moca_prediction(dataio, patient_info, patient_array, params):
    ### Task Setting 
    task = params['task'][2]
    split_method = 'ratio'
    ratio = 0.8 # provide the ratio
    krun = 5 # run 5 times then average the result 
    ### Initialization
    rmse = np.zeros(krun, dtype='float32') # evaluation metric
    n_feature = dataio.feature.feature_len
    param_w = np.zeros([n_feature, krun], dtype='float32')
    ### MoCA Reading
    feature = dataio.feature
    patient_info = feature.get_moca_score(patient_info, patient_array)
    print ('-----')
    split = Splitter(task, patient_array, ratio, split_method)
    for k in range(krun):
        ### Data Splitting
        train_data, test_data = split.get_splitter(k)
        ### Model Training 
        moca_pred = MoCAPredictor(k, patient_info, train_data, params['result_path'])
        model, y_pred = moca_pred.train_model()
        param_w[:,k] = moca_pred.get_param()
        ### Evaluating
        pd_eval = Evaluator(model, test_data, patient_info, task, moca_pred)
        rmse[k] = pd_eval.compute_accuracy()
    print ('-----')
    print ('RMSE of the %s task: %f' %(task, np.sum(rmse)/krun))
    ### Displaying Feature (selected by prediction model)
    feature = dataio.feature
    feature.get_pred_feature(param_w, krun, 'moca')

    
################################ CLUSTERING ################################### 
def pat_clustering(dataio, patient_info, patient_array, params):
    ### Task Setting 
    task = params['task'][3]
    k_cluster = 2
    dtw = False # use dynamic time warping 
    print ('-----')
    pat_cluster = PatCluster(k_cluster, patient_info, 
                             params['result_path'], patient_array, dtw)
    y, pred_y = pat_cluster.train_model()
    pat_cluster.plot_result()
    cls_eval = Evaluator(task)
    if k_cluster == 2:
        ami = cls_eval.compute_AMI(y, pred_y)
        print ('-----')
        print ('AMI of the %s task: %f' %(task, ami))
    elif k_cluster == 3:
        print ('-----')
        print ('statistics analyzing...')
        dataio.patient_cluster = pat_cluster.get_cluster()
        pat_cluster.compute_statistics(dataio, params['input_path'])

    
def main(params):
    dataio, patient_info, patient_array = data_preprocess(params)
    pd_prediction(dataio, patient_info, patient_array, params)
    hy_prediction(dataio, patient_info, patient_array, params)
    moca_prediction(dataio, patient_info, patient_array, params)
    pat_clustering(dataio, patient_info, patient_array, params)
    print ('Done!')

    
if __name__ == '__main__':
    main ({
        'task': ['PD prediction',
                 'H&Y stage prediction',
                 'MoCA prediction',  
                 'Patient clustering'],
        'input_path': 'data/',
        'result_path': 'results/',
        'reload': False})