#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
data imputation 
'''
import numpy as np
import operator 


from fancyimpute.mice import MICE
from fancyimpute.knn import KNN
from fancyimpute.nuclear_norm_minimization import NuclearNormMinimization
#from low_rank_data import XY, XY_incomplete, missing_mask
#from fancyimpute.common import reconstruction_error
#print (XY_incomplete)

class Imputation(object):

    def __init__(self, patient_info, feature_len):
        self.feature_len = feature_len
        # patient dimension
        self.patient_info = patient_info
        self.patient_array, self.patient_time = self.get_array()
        self.patient_mask, self.patient_mask_idx = self.get_mask()
        
    
    def get_imputation(self, method='simple'):
        if method == 'simple':
            self.simple_imputation()
        elif method == 'multiple':  
            self.multiple_imputation() 
        return self.patient_array
            
            
    def simple_imputation(self):
        feat_median, pat_median = self.get_median()
       
        for pat_id in self.patient_array:
            feat_array = self.patient_array[pat_id]
            feat_mask_idx = self.patient_mask_idx[pat_id] # list of (row_idx, col_idx)
            
            for idx in range(len(feat_mask_idx[0])): # each position need imputation
#                print ('-----------')
                row_idx = feat_mask_idx[0][idx]
                col_idx = feat_mask_idx[1][idx]
                m, n = np.shape(feat_array)
                
                if int(feat_array[row_idx-1, col_idx]) != -1: # last occurrence carry forward strategy
                    feat_array[row_idx, col_idx] = feat_array[row_idx-1, col_idx]
                else:
                    
                    if (row_idx < m-1) and int(feat_array[row_idx+1, col_idx]) != -1: # first occurrence carry backward strategy
                        feat_array[row_idx, col_idx] = feat_array[row_idx+1, col_idx]
                    else:
                        if int(pat_median[pat_id][col_idx]) != -1: # fill with patient median value
                            feat_array[row_idx, col_idx] = pat_median[pat_id][col_idx]
                        elif int(pat_median[pat_id][col_idx]) == -1: # fill with feature median value
                            feat_array[row_idx, col_idx] = feat_median[col_idx]
            self.patient_array[pat_id] = feat_array  
        
        
    def multiple_imputation(self):
        # mice: Multiple Imputation by Chained Equations
        X = list()
        X_idx = dict() # pid : patient
        bgn_idx = 0
        end_idx = 0
        row_sum = np.zeros([1, self.feature_len])
        # store into one matrix
        for pid in self.patient_array:
            feat_array = self.patient_array[pid]
            m, _ = np.shape(feat_array)
            end_idx = bgn_idx + m
            for i in range(m):
                row = feat_array[i, :]
                for j in range(len(row)):
                    if row[j] == -1:
                        row[j] = np.nan
                        row_sum[:, j] += 1 
                feat_array[i, :] = row
                X.append(feat_array[i, :])
            X_idx[pid] = (bgn_idx, end_idx)
            bgn_idx = end_idx
        XY_incomplete = np.array(X, dtype='float32')
#        mice = MICE(n_imputations=100, impute_type='col')
#        XY_completed = mice.complete(XY_incomplete)  
#        XY_completed = NuclearNormMinimization().complete(XY_incomplete)
        XY_completed = KNN(k=3).complete(XY_incomplete)
      
        # store into sequential vectors
        for pid in self.patient_array:
            bgn_idx, end_idx = X_idx[pid]
            feat_array = XY_completed[bgn_idx:end_idx, :]
            m, _ = np.shape(feat_array)
            for i in range(m): 
                row = feat_array[i, :]
                for j in range(len(row)):
                    if row[j] == np.nan:
                        row[j] = -1
                feat_array[i, :] = row       
            self.patient_array[pid] = feat_array
        self.simple_imputation()
    
    
#    def delete_missing_col(self, XY, row_sum, m):
#        nan_idx = [idx for idx in range(len(row_sum)) if row_sum[:, idx] == m]
#        print (nan_idx)
#        return nan_idx
        
        
    def get_array(self):
        patient_array = dict() # pat_id: feature value array
        patient_time = dict() # pat_id: record time list
        for pat_id in self.patient_info:
            patient_rec = self.patient_info[pat_id].patient_rec
            if len(patient_rec)==0:
                continue
            patient_rec = sorted(patient_rec.items(), key=operator.itemgetter(0))
            feat_array = [pr[1] for pr in patient_rec]
            patient_array[pat_id] = np.array(feat_array, dtype='float32')
            time_list = [pr[0] for pr in patient_rec]
            patient_time[pat_id] = np.array(time_list, dtype='int')
        return (patient_array, patient_time)
        
        
    def get_mask(self):
        patient_mask = dict() # 1 has value, 0 no value
        patient_mask_idx = dict() # missing values
        for pat_id in self.patient_array:
            feat_mask_idx = []
            feat_array = self.patient_array[pat_id]
            feat_mask_idx = np.where(feat_array == -1) 
#            print (feat_mask_idx)
            idx_len = len(feat_mask_idx[0])
            if idx_len == 0:
                continue
            # store mask array for each patient, 1 has value, 0 no value  
            shape = np.shape(feat_array)
            feat_mask = np.ones(shape, dtype='int')
            feat_mask[feat_mask_idx] = 0
            patient_mask[pat_id] = feat_array
            # store mask id (missing values) for each patient
#            print (len(feat_mask_idx[0]))
#            print (len(feat_mask_idx[1]))
            patient_mask_idx[pat_id] = feat_mask_idx
#            patient_mask_idx[pat_id] = [(feat_mask_idx[0][i], feat_mask_idx[1][i]) for i in range(idx_len)]
        return (patient_mask, patient_mask_idx)
        
        
    def get_median(self):
        # output: feat_median: an array of median value
        #         pat_median: a dict {pat id: {feature name: median value}}
        feat_median = np.zeros(self.feature_len)
        feat_value = dict()
        pat_median = dict() # pat_id: feature median
        for pat_id in self.patient_array:
            feat_array = self.patient_array[pat_id]
            pf_median = np.zeros(self.feature_len)-1
            for col in range(self.feature_len):
                # compute patient median feature value
                rows = np.where(feat_array[:, col]!=-1)
                rows = rows[0]
                if len(rows) == 0:
                    continue
#                print (feat_array[rows, col])
                pf_median[col] = np.round(np.median(feat_array[rows, col]))
#                print (numpy.round(numpy.median(feat_array[rows, col])))
                # store total feature value
                if col not in feat_value:
                    feat_value[col] = list()
                feat_value[col].extend(feat_array[rows, col])
#                print (feat_value[col])
            pat_median[pat_id] = pf_median

        # compute global meidan
        for col in feat_value.keys():
            feat_median[col] = np.round(np.median(feat_value[col]))  
        return (feat_median, pat_median)
        
  
        
        
        