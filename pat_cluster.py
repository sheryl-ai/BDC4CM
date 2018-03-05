#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from feature.variable import Variable


class PatCluster(object):
    
    
    def __init__(self, n_clusters, patient_info, resultpath, patient_array, dtw=False):
        self.n_clusters = n_clusters
        self.param = dict()
        self.patient_info = patient_info
        self.resultpath = resultpath
        self.dtw = dtw
        # generate labels and samples for training data
        self.patient_array = patient_array
        if dtw == True:
            self.aggregate = False
            self.pat_id, self.y, _ = self.get_patient()
            self.get_dtw_sim()
        else:
            self.aggregate = True
            self.pat_id, self.y, self.X = self.get_patient()
        self.patient_cluster = dict()

     
    def train_model(self):
        if self.dtw == True: # use dtw results
            reduced_data = PCA(n_components=32).fit_transform(self.sim_mat)
        else: # use aggregated data
            self.X = preprocessing.scale(self.X)
            reduced_data = PCA(n_components=32).fit_transform(self.X)
        self.k_means = KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)
        self.k_means.fit(reduced_data)
        pred_y = self.k_means.labels_
        return (self.y, pred_y)
        
    
    def get_cluster(self):
        try: 
            labels = [str(lab+1) for lab in self.k_means.labels_]
            self.patient_cluster = dict(zip(self.pat_id, labels))
            return self.patient_cluster
        except ValueError:
             print('Train Model Firstly')
        
        
    def get_dtw_sim(self):
        n_pat = len(self.patient_array)
        sim_mat = np.zeros([n_pat, n_pat], dtype='float32')
        pid_list = [pid for pid in self.patient_array.keys()]
        pid_dict = dict(zip(pid_list, range(n_pat)))
        # reduce patient array
        self.get_reduced_array()
        
        for pid_1 in pid_list:
            feat_array_1 = self.patient_array[pid_1]
            row_idx = pid_dict[pid_1] 
            for col_idx in range(row_idx+1, n_pat):
                pid_2 = pid_list[col_idx]
                feat_array_2 = self.patient_array[pid_2]
                distance, path = fastdtw(feat_array_1, feat_array_2, dist=euclidean)
                sim_mat[row_idx, col_idx] = distance
                sim_mat[col_idx, row_idx] = distance
        self.sim_mat = sim_mat
        print (sim_mat)
        print (sim_mat.shape)
        
    
    def get_reduced_array(self):
        X = list()
        X_idx = dict() # pid : patient
        bgn_idx = 0
        end_idx = 0
        # store into one matrix
        for pid in self.patient_array:
            feat_array = self.patient_array[pid]
            m, _ = np.shape(feat_array)
            end_idx = bgn_idx + m
            for i in range(m):
                X.append(feat_array[i, :])
            X_idx[pid] = (bgn_idx, end_idx)
            bgn_idx = end_idx
        # data reduce
        X = np.array(X, dtype = 'float32')
        X = preprocessing.scale(X)
        X = PCA(n_components=32).fit_transform(X)
        # store into sequential vectors
        for pid in self.patient_array:
            bgn_idx, end_idx = X_idx[pid]
            print (bgn_idx)
            print (end_idx)
            print ('-----')
            feat_array = X[bgn_idx:end_idx, :]
            self.patient_array[pid] = feat_array
        
        
    def plot_result(self):
        labels = self.k_means.labels_
        print (labels)
        cluster_centers = self.k_means.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        plt.figure()

        X = PCA(n_components=2).fit_transform(self.X)
        for k, col in zip(range(n_clusters_), colors):
           my_members = labels == k
           print (len(my_members))
           print (X.shape)
           cluster_center = cluster_centers[k]
           plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
           plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        
     
    def get_patient(self):
        ### data format 
        ### x = [[1,0,1], [-1,0,-1]]
        ###
        pat_id = list()
        y = list()
        X = list()
        for pid, feat_array in self.patient_array.items():
            if self.patient_info[pid].diagnosis == '1':
                y.append(1)
            elif self.patient_info[pid].diagnosis == '17': 
                y.append(-1)
            if self.aggregate == True:
                m, _ = np.shape(feat_array) 
                X.append((np.sum(feat_array, axis=0)/m).tolist())
            pat_id.append(pid)
        X = np.array(X, dtype = 'float32')
#        X_scaled = preprocessing.scale(X)
#        print (X_scaled)
#        print (np.sum(X))
#        X = np.divide(X, np.sum(X, axis=1))
        return (pat_id, y, X)
        
        
    def compute_statistics(self, dataio, filepath):
         
        var = Variable(self.n_clusters, filepath, self.resultpath)
        ftype = 'demographics'
        p = var.get_variables(dataio, ftype)
        var.p_value.extend(p) 
    
        ftype = 'motor'
        _ = var.get_variables(dataio, ftype, 'MDS UPDRS PartI')
        _ = var.get_variables(dataio, ftype, 'MDS UPDRS PartII')
        _ = var.get_variables(dataio, ftype, 'MDS UPDRS PartIII', 'MDS-UPDRS')
        _ = var.get_variables(dataio, ftype, 'MDS UPDRS PartIII', 'H&Y')
        p = var.get_variables(dataio, ftype, 'MDS UPDRS PartIV')
        var.p_value.extend(p)
    
        ftype = 'nonmotor'
        _ = var.get_variables(dataio, ftype, 'BJLO')
        _ = var.get_variables(dataio, ftype, 'ESS')
        _ = var.get_variables(dataio, ftype, 'GDS')
        _ = var.get_variables(dataio, ftype, 'HVLT', 'Immediate Recall')
        _ = var.get_variables(dataio, ftype, 'HVLT', 'Discrimination Recognition')
        _ = var.get_variables(dataio, ftype, 'HVLT', 'Retention')
        _ = var.get_variables(dataio, ftype, 'LNS')
        _ = var.get_variables(dataio, ftype, 'MoCA', pat_edu=var.pat_edu)
        _ = var.get_variables(dataio, ftype, 'QUIP')
        _ = var.get_variables(dataio, ftype, 'RBD')
        _ = var.get_variables(dataio, ftype, 'SCOPA-AUT')
        _ = var.get_variables(dataio, ftype, 'SF')
        _ = var.get_variables(dataio, ftype, 'STAI')  
        _ = var.get_variables(dataio, ftype, 'SDM')
        p = var.get_variables(dataio, ftype, 'MCI')
        var.p_value.extend(p)
    
        ftype = 'biospecimen'
        var.get_variables(dataio, ftype, 'DNA')
        _ = var.get_variables(dataio, ftype, 'CSF', 'Total tau')
        _ = var.get_variables(dataio, ftype, 'CSF', 'Abeta 42')
        _ = var.get_variables(dataio, ftype, 'CSF', 'p-Tau181P')
        p = var.get_variables(dataio, ftype, 'CSF', 'CSF Alpha-synuclein')
        var.p_value.extend(p)
    
        ftype = 'image'
        _ = var.get_variables(dataio, ftype, 'DaTScan SBR', 'CAUDATE RIGHT')
        _ = var.get_variables(dataio, ftype, 'DaTScan SBR', 'CAUDATE LEFT')
        _ = var.get_variables(dataio, ftype, 'DaTScan SBR', 'PUTAMEN RIGHT')
        p = var.get_variables(dataio, ftype, 'DaTScan SBR', 'PUTAMEN LEFT')
        var.get_variables(dataio, ftype, 'MRI')
        var.p_value.extend(p)
    
        ftype = 'medication'
        p = var.get_variables(dataio, ftype, 'MED USE')
        var.p_value.extend(p)
