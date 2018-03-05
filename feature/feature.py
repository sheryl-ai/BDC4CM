# -*- coding: utf-8 -*-
import csv
import operator
import numpy as np

from feature.demo_feature import DemoFeature
from feature.motor_feature import MotorFeature
from feature.nonmotor_feature import NonMotorFeature
from feature.bio_feature import BioFeature
from feature.image_feature import ImageFeature
from feature.med_feature import MedFeature

from utils.numeric import isfloat, isint

class Feature(object):
    
    def __init__(self, filepath, resultpath):
        self.resultpath = resultpath
        
        self.demographics = DemoFeature()
        self.motor = MotorFeature(filepath)
        self.nonmotor = NonMotorFeature(filepath)
        self.biospecimen = BioFeature(filepath)
        self.image = ImageFeature(filepath) 
        self.medication = MedFeature(filepath) 
        
        self.feature_name = dict() 
        self.feature_list = list()
        self.feature_dict = dict()
        self.feature_len = 0
        self.feat_var_map = self.get_mapping() # feature-variable map {feature name: varible}
#        print (self.feat_var_map)
  
       
    def load_feature(self, ftype=None, fname=None, featname=None):
        try:
            if ftype == 'Motor':
                self.motor.load_feature(fname, featname)
            elif ftype == 'Non-Motor':
                self.nonmotor.load_feature(fname, featname)
            elif ftype == 'Biospecimen': 
                self.biospecimen.load_feature(fname, featname)
            elif ftype == 'Image': 
                self.image.load_feature(fname, featname)
            elif ftype == 'Medication':
                self.medication.load_feature(fname, featname)
        except ValueError:
            print ('the type should be one of Motor, Non-Motor, Biospecimen, and Image!')
     
            
    def get_feature_name(self):
        feature_name = dict() # variable type: feature name
        feature_name['motor'] = self.motor.feature_info.keys()
        feature_name['non-motor'] = self.nonmotor.feature_info.keys()
        feature_name['biospecimen'] = self.biospecimen.feature_info.keys()
        feature_name['image'] = self.image.feature_info.keys()
        feature_name['medication'] = self.medication.feature_info.keys() 
        self.feature_name = feature_name
        self.feature_list = list()
        for var_type, fn in self.feature_name.items():
            self.feature_list.extend(fn)
        self.feature_list = sorted(self.feature_list)
        self.feature_len = len(self.feature_list)
        self.feature_dict = dict(zip(self.feature_list, range(self.feature_len)))
   
        
    def get_hy_stage(self, patient_info, patient_array):
#        hy_stage = self.motor.get_hy_stage()
        hy_idx = self.feature_dict['NHY']
        
        for pat_id, patient in patient_info.items():
            if pat_id in patient_array:
                patient.hy_stage = patient_array[pat_id][-1, hy_idx]
            patient_info[pat_id] = patient
        return patient_info
        
        
    def get_moca_score(self, patient_info, patient_array):
        moca_idx = self.feature_dict['MCATOT']
        max_val = list()
        min_val = list()
        for pat_id, patient in patient_info.items():
            if pat_id in patient_array:
                patient.moca = patient_array[pat_id][-1, moca_idx]
                max_val.append(patient.moca)
            patient_info[pat_id] = patient
        return patient_info
                
        
    def get_mapping(self):
        feat_var_map = dict()
        # motor 
        val_set = set(['MDS UPDRS PartI', 'MDS UPDRS PartII', 'MDS UPDRS PartIII', 
        'H&Y', 'MDS UPDRS PartIV'])
        for val in val_set:
            feature_set = self.motor.get_feature_set(val)
            if 'MDS UPDRS' in val:
                val = 'MDS-UPDRS'
            feat_var_map.update(dict(zip(list(feature_set), [val]*len(feature_set))))
        # nonmotor
        val_set = set(['BJLO', 'ESS', 'GDS', 'HVLT', 'LNS', 'MoCA', 'UPSIT', 'QUIP', 
                       'RBD', 'SCOPA-AUT', 'SF', 'STAI', 'SDM', 'MCI'])
        for val in val_set:
            feature_set = self.nonmotor.get_feature_set(val)
            feat_var_map.update(dict(zip(list(feature_set), [val]*len(feature_set))))
        # biospecimen
        val_set = set(['DNA', 'CSF'])
        for val in val_set:
            feature_set = self.biospecimen.get_feature_set(val)
            feat_var_map.update(dict(zip(list(feature_set), [val]*len(feature_set))))
        # imaging
        val_set = set(['DaTScan SBR', 'MRI'])
        for val in val_set:
            if 'DaTScan SBR' in val:
                val = 'DaTScan'
            feature_set = self.image.get_feature_set(val)
            feat_var_map.update(dict(zip(list(feature_set), [val]*len(feature_set))))
        # medication
        val_set = set(['MED USE'])
        for val in val_set:
            feature_set = self.medication.get_feature_set(val)
            if 'MED USE' in val:
                val = 'medication'
            feat_var_map.update(dict(zip(list(feature_set), [val]*len(feature_set))))
        return feat_var_map
        
        
    def expand_variable(self, abbrs, pred_dict):
        nonmotor_val = list()
        if 'MDS-UPDRS' in abbrs:
            abbrs.extend(['motor'])
            pred_dict['motor'] = pred_dict['MDS-UPDRS']
        if 'MoCA' in abbrs:
            abbrs.extend(['cognitive'])
            pred_dict['cognitive'] = pred_dict['MoCA']
            nonmotor_val.append(pred_dict['MoCA'])
        if 'BJLO' in abbrs:
            abbrs.extend(['visuospatial'])
            pred_dict['visuospatial'] = pred_dict['BJLO']
            nonmotor_val.append(pred_dict['BJLO'])
        if 'LNS' in abbrs:
            abbrs.extend(['letter'])
            abbrs.extend(['number'])
            pred_dict['letter'] = pred_dict['LNS']
            pred_dict['number'] = pred_dict['LNS']
            nonmotor_val.append(pred_dict['LNS'])
        if 'ESS' in abbrs:
            abbrs.extend(['sleepiness'])
            pred_dict['sleepiness'] = pred_dict['ESS']
            nonmotor_val.append(pred_dict['ESS'])
        if 'GDS' in abbrs:
            abbrs.extend(['depression'])
            pred_dict['depression'] = pred_dict['GDS']
            nonmotor_val.append(pred_dict['GDS'])
        if 'HVLT' in abbrs:
            abbrs.extend(['verbal'])
            pred_dict['verbal'] = pred_dict['HVLT']
            nonmotor_val.append(pred_dict['HVLT'])
        if 'QUIP' in abbrs:
            abbrs.extend(['impulsive-compulsive'])
            pred_dict['impulsive-compulsive'] = pred_dict['QUIP']
            nonmotor_val.append(pred_dict['QUIP'])
        if 'RBD' in abbrs:
            abbrs.extend(['eye'])
            pred_dict['eye'] = pred_dict['RBD']
            nonmotor_val.append(pred_dict['RBD'])
        if 'SCOPA-AUT' in abbrs:
            abbrs.extend(['autonomic'])
            pred_dict['autonomic'] = pred_dict['SCOPA-AUT'] 
            nonmotor_val.append(pred_dict['SCOPA-AUT'])
        if 'STAI' in abbrs:
            abbrs.extend(['anxiety'])
            pred_dict['anxiety'] = pred_dict['STAI']
            nonmotor_val.append(pred_dict['STAI'])
        if 'MCI' in abbrs:
            abbrs.extend(['cognitive'])
            pred_dict['cognitive'] = pred_dict['MCI']
            nonmotor_val.append(pred_dict['MCI'])
        if 'SF' in abbrs:
            abbrs.extend(['semantic'])
            abbrs.extend(['fluency'])
            pred_dict['semantic'] = pred_dict['SF']
            pred_dict['fluency'] = pred_dict['SF']
            nonmotor_val.append(pred_dict['SF'])
        if 'DaTScan' in abbrs:
            abbrs.extend(['dopamine'])
            abbrs.extend(['transporter'])
            pred_dict['dopamine'] = pred_dict['DaTScan']
            pred_dict['transporter'] = pred_dict['DaTScan']
        if 'MRI' in abbrs:
            abbrs.extend(['Magnetic'])
            abbrs.extend(['Resonance'])
            pred_dict['Magnetic'] = pred_dict['MRI']
            pred_dict['Resonance'] = pred_dict['MRI']
        if 'DaTScan' in abbrs and 'MRI' in abbrs:
            abbrs.extend(['imaging'])
            pred_dict['imaging'] = max(pred_dict['DaTScan'], pred_dict['MRI'])
        if 'MoCA' in abbrs and 'MCI' in abbrs:
            abbrs.extend(['cognitive'])
            pred_dict['cognitive'] = max(pred_dict['MoCA'], pred_dict['MCI'])
        if len(nonmotor_val)>0:
            abbrs.extend(['non-motor'])
            pred_dict['non-motor'] = np.max(nonmotor_val)
        return (abbrs, pred_dict)
        
        
    def get_pred_feature(self, param_w, k, filename):
        # set initial weight for k results
        feature_list = list()
        for var_type, fn in self.feature_name.items():
            feature_list.extend(fn)
        feature_list = sorted(feature_list)
        n_feature = len(feature_list)
        average_weights = np.sum(np.abs(param_w), axis=1)/k
        # rank the feature weight
        feat_weights = dict(zip(feature_list, average_weights))
        feat_weights = sorted(feat_weights.items(), key=operator.itemgetter(1), reverse=True)
        pred_feat = [feat_weights[i][0] for i in range(n_feature) 
                        if feat_weights[i][1] > 0.0]
        pred_val = [feat_weights[i][1] for i in range(n_feature)
                        if feat_weights[i][1] > 0.0]
        
        # save predicted feature name
        f = open(self.resultpath + filename + '_feature_by_predict_model.csv', 'w')
        writer = csv.writer(f)
        pred_feat = [pf for pf in pred_feat]
        pred_val = [pv for pv in pred_val]
        pred_var = [self.feat_var_map[pf] for pf in pred_feat if pf in self.feat_var_map]
        pred_dict = dict()
        for i in range(len(pred_var)):
            pv = pred_var[i]
            if pv not in pred_dict:
                pred_dict[pv] = list()
            pred_dict[pv].append(pred_val[i])
 
        for pv in pred_dict:
            pred_dict[pv] = np.max(pred_dict[pv])    

        pred_var, pred_dict = self.expand_variable(pred_var, pred_dict)
        pred_dict = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        results = list()
        for var_val in pred_dict:
            var = var_val[0]
            val = var_val[1]
            results.append([var, val])
        writer.writerows(results)
        f.close()
        return pred_feat             
                     
                     
                     