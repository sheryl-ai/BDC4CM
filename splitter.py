#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Splitter(object):
    
    def __init__(self, task, patient_array, variable, split_method, patient_info=None):
        self.task = task
        self.patient_array = patient_array
        self.split_method = split_method
        if split_method == 'cross-validation':
            self.kfold = variable
            # "cross-validation": dictionary {pat_id: kfold index}
            self.id_dict = self.get_id_dict(split_method) 
        elif split_method == 'ratio':
            self.ratio = variable
        elif split_method == 'cluster':
            pass
        else:
            print("Unexpected error: choose a correct task.")
        if patient_info != None:
            self.patient_info = patient_info
            self.stratified = True
        else:
            self.stratified = False
        
        
    def get_splitter(self, k=5):
        if self.split_method == 'cross-validation':
            train_data, test_data = self.get_cvsplitter(k)
        elif self.split_method == 'ratio':
            # "ratio": dictionary {pat_id: splid index}
            self.id_dict = self.get_id_dict(self.split_method) 
            train_data, test_data = self.get_ratiosplitter()
        elif self.split_method == 'cluster':
            train_data = self.get_aggregator()
            test_data = None
        return (train_data, test_data)
      
        
    def get_cvsplitter(self, k): # given kfold index k
        # data split for PD prediction task
        if self.task == 'PD prediction':
            train_data = list()
            test_data = list()          
            for pid in self.patient_array:
                feat_array = self.patient_array[pid]
                m, _ = np.shape(feat_array) 
                if self.id_dict[pid] == k:
                    feature = np.sum(feat_array, axis=0)/m
                    train_data.append([pid, feature.tolist()]) # the first element is pid
                else: 
                    feature = np.sum(feat_array, axis=0)/m
                    test_data.append([pid, feature.tolist()])
        else:
            print ("Only support PD prediction task currently.")
        return (train_data, test_data)
       
       
    def get_ratiosplitter(self):
        # data split for MoCA/H&Y prediction task
        if self.task == 'MoCA prediction' or self.task == 'H&Y stage prediction':
            obs_ratio =  2/3 # observation ratio 
            train_data = list()
            test_data = list()
            for pid in self.patient_array:
                feat_array = self.patient_array[pid]
                m, n = np.shape(feat_array)
                m_obs = round(m * obs_ratio) # number of observed training records
                feat_array = feat_array[:m_obs, :]
                if self.id_dict[pid] == 0: 
                    feature = np.sum(feat_array, axis=0)/m_obs
                    train_data.append([pid, feature.tolist()]) # the first element is pid
                elif self.id_dict[pid] == 1:
                    feature = np.sum(feat_array, axis=0)/m_obs
                    test_data.append([pid, feature.tolist()])
        else:
            print ("Only support MoCA or H&Y prediction task currently")
        return (train_data, test_data)

        
    def get_aggregator(self):
        if self.task == 'Disease Subtyping':
            data = list()         
            for pid in self.patient_array:
                feat_array = self.patient_array[pid]
                m, _ = np.shape(feat_array) 
                feature = np.sum(feat_array, axis=0)/m
#                feature = feat_array[0, :]
                data.append([pid, feature.tolist()]) # the first element is pid
        else:
            print ("Only support Disease Subtyping task currently.")
        return data  
        
    

    def get_id_dict(self, split_method):
        if split_method == 'cross-validation':
            # randomly assign patient  
            n_sample = len(self.patient_array)
            pat_id = [pid for pid in self.patient_array.keys()]
            k_id = np.random.randint(self.kfold, size=n_sample)
            id_dict = dict(zip(pat_id, k_id))
            
        elif split_method == 'ratio':
            # randomly shuffle patient
            n_sample = len(self.patient_array)
            pat_id = [pid for pid in self.patient_array.keys()]
            rand_idx = np.arange(n_sample)
            np.random.shuffle(rand_idx)
            pat_id = [pat_id[idx] for idx in rand_idx]
            n_tr_sample = round(n_sample * self.ratio) 
            n_te_sample = n_sample - n_tr_sample
            split_id = np.zeros(n_tr_sample).tolist() + np.ones(n_te_sample).tolist()
            if self.stratified == True:
                pat_id = self.get_stratified_pid(pat_id, n_tr_sample, n_te_sample)
            id_dict = dict(zip(pat_id, split_id))
            
        return id_dict
        
    
    def get_stratified_pid(self, pat_id, n_tr_sample, n_te_sample):
        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        if self.task == 'H&Y stage prediction':
            pat_id_tr = list()
            pat_id_te = list()
            print (n_te_sample)
            stage_num = np.zeros(6) 
            for pid in pat_id:
                hy_stage = self.patient_info[pid].hy_stage
                stage_num[int(hy_stage)] += 1
                
            stage_num = np.ceil((1-self.ratio) * stage_num)
            print (stage_num)
            stage_count = np.zeros(6)
            for pid in pat_id:
                hy_stage = self.patient_info[pid].hy_stage
                if stage_count[int(hy_stage)] <= stage_num[int(hy_stage)]:
                    pat_id_te.append(pid)
                else:
                    pat_id_tr.append(pid)
                stage_count[int(hy_stage)] += 1
        else:
             print ("Only support H&Y prediction task currently")
        pat_id_tr.extend(pat_id_te)
        return pat_id_tr
    
        