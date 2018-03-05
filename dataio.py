# -*- coding: utf-8 -*-

import csv, codecs
from patient.patient import Patient
from feature.feature import Feature


class DataIO(object):
    
    def __init__(self, filepath, resultpath):
        self.filepath = filepath 
        self.resultpath = resultpath
        self.patient_id = list() # pat_id
        self.feature = Feature(filepath, resultpath) # feature_info and other statistics
        self.patient_info = list() # list of Patient()
        
        
    def load_patient_id(self):
        f = codecs.open(self.filepath + 'patient_id.csv', 'r', 'utf-8')
        reader = csv.reader(f)
        line_ctr = 0
        for row in reader:
            # table title
            if line_ctr < 1:
                table_ttl = dict(zip(row, range(len(row))))
                line_ctr += 1
                continue
            
            pid = row[table_ttl['PATNO']]
            self.patient_id.append(pid)
            line_ctr += 1
        f.close()

    
    def load_demographics(self):
        f = codecs.open(self.filepath + 'demographics/' + 'patient_demo.csv', 'r', 'utf-8')
        reader = csv.reader(f)
        line_ctr = 0
        for row in reader:
            # table title
            if line_ctr < 1:
                table_ttl = dict(zip(row, range(len(row))))
                line_ctr += 1
                continue
            if len(row) == 0:
                continue
            pval = Patient()
            pval.id = row[table_ttl['ID']]
            pval.age = row[table_ttl['AGE']]
            pval.gender = row[table_ttl['GENDER']]
            pval.edu_year = row[table_ttl['EDUCATION YEAR']]
            pval.duration = row[table_ttl['DURATION(MONTH)']]
            pval.diagnosis = row[table_ttl['DIAGNOSIS']]
            self.patient_info.append(pval)
            line_ctr += 1
        f.close()
 
        
    def load_feature(self, ftype=None, fname=None, featname=None):
        self.feature.load_feature(ftype, fname, featname)
        
    def read_data(self):
        self.load_patient_id()
        self.load_demographics()
        self.load_feature('Motor', 'MDS UPDRS PartI')
        self.load_feature('Motor', 'MDS UPDRS PartII')
        self.load_feature('Motor', 'MDS UPDRS PartIII')
        self.load_feature('Motor', 'MDS UPDRS PartIV')
    
        self.load_feature('Non-Motor', 'BJLO')
        self.load_feature('Non-Motor', 'ESS')
        self.load_feature('Non-Motor', 'GDS')
        self.load_feature('Non-Motor', 'HVLT')
        self.load_feature('Non-Motor', 'LNS')
        self.load_feature('Non-Motor', 'MoCA')
        self.load_feature('Non-Motor', 'QUIP')
        self.load_feature('Non-Motor', 'RBD')
        self.load_feature('Non-Motor', 'SCOPA-AUT')
        self.load_feature('Non-Motor', 'SF')
        self.load_feature('Non-Motor', 'STAI')
        self.load_feature('Non-Motor', 'SDM')
        self.load_feature('Non-Motor', 'MCI')
    
        self.load_feature('Biospecimen', 'DNA')
        self.load_feature('Biospecimen', 'CSF', 'Total tau')
        self.load_feature('Biospecimen', 'CSF', 'Abeta 42')
        self.load_feature('Biospecimen', 'CSF', 'p-Tau181P')
        self.load_feature('Biospecimen', 'CSF', 'CSF Alpha-synuclein')
            
        self.load_feature('Image', 'DaTScan SBR')
        self.load_feature('Image', 'MRI')
        self.load_feature('Medication', 'MED USE')

        return self.feature.get_feature_name()