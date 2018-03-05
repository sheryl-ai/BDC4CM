#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
data concatenation 
'''
import numpy
import operator 
from utils.numeric import isfloat, isint
from utils.time_converter import convert_int


class Concatenation(object):

    def __init__(self, dataio):
        self.dataio = dataio
        
        # patient dimension
        # get a dictionary with each item pat_id : Patient
        self.patient_id = dataio.patient_id
        self.patient_info = dict() # pat_id : Patinet
        patient_info = dataio.patient_info # list of Patient()
        for patient in patient_info:
            if patient.id in self.patient_id:
                self.patient_info[patient.id] = patient

        # feature dimension 
        self.feature_name = dataio.feature.feature_name
        self.feature_list = dataio.feature.feature_list
        self.feature_dict = dataio.feature.feature_dict
        self.feature_len = dataio.feature.feature_len
   
    def get_concatenation(self):
        for fname in self.feature_list:
            # get triple lists (patient id, time, feature value)    
            if fname in self.feature_name['motor']:
                tpl_list = self.dataio.feature.motor.feature_info[fname]
            elif fname in self.feature_name['non-motor']:
                tpl_list = self.dataio.feature.nonmotor.feature_info[fname]
            elif fname in self.feature_name['biospecimen']:
                tpl_list = self.dataio.feature.biospecimen.feature_info[fname]
            elif fname in self.feature_name['image']:
                tpl_list = self.dataio.feature.image.feature_info[fname]
            elif fname in self.feature_name['medication']:
                tpl_list = self.dataio.feature.medication.feature_info[fname]
#            print (len(tpl_list))
            # store the patient info 
            pat_record = dict() # patient id : a list of (time stamp, feature val)
            for tpl in tpl_list:
                if isint(tpl[2])==True:
                    fval = int(tpl[2])
                elif isfloat(tpl[2])==True:
                    fval = float(tpl[2])
                else:
                    continue
                pat_id = tpl[0]
                time = convert_int(tpl[1])
                if pat_id not in self.patient_id:
                    continue
            
                if pat_id not in pat_record:
                    pat_record[pat_id] = list()
                    pat_record[pat_id].append((time, fval))
                else:
                    pat_record[pat_id].append((time, fval))
                
            # store the records into Patient 
            fidx = self.feature_dict[fname] # index of feature dimension
            for pat_id, tf_list in pat_record.items():
                patient = self.patient_info[pat_id]
                for time, fval in tf_list:
                    if time not in patient.patient_rec:
                        patient.patient_rec[time] = numpy.zeros(self.feature_len, dtype='float32')-1
                    patient.patient_rec[time][fidx] = fval
                self.patient_info[pat_id] = patient
        return (self.patient_info, self.feature_len)
