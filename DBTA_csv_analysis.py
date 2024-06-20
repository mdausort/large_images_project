#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:08:06 2024

@author: tgodelaine
"""

#%% csv analysis

import csv 
import os 
import pandas as pd
import numpy as np

PATH = '/Users/tgodelaine/Desktop/'
path_csv = os.path.join(PATH, 'ICIP/Suite_ICIP/data/DBTA/annotation.csv')
file_csv = pd.read_csv(path_csv)

diagnosis = file_csv['diagnosis']
age = file_csv['age']
sex = file_csv['sex']

child = age[age < 18]
print("Number of children: ", len(child))

male = sex[sex == 'male']
print("Number of male: ", len(male))

child_male = file_csv[(age < 18) & (sex == 'male')]
print("Number of male children: ", len(child_male))

diagnosis_ = list(set(diagnosis))[1:]
print("\nNumber of diagnosis: ", len(diagnosis_))

glioma = [g for g in diagnosis_ if 'glioma' in g]
print("Number of glioma: ", len(glioma))
print("Glioma: ", glioma)

child_diagnosis = list(set(diagnosis[(age < 18)]))[1:]
print("\nNumber of child diagnosis: ", len(child_diagnosis))

child_diagnosis_glioma = [g for g in child_diagnosis if 'glioma' in g]
print("Number of child glioma diagnosis: ", len(child_diagnosis_glioma))
print("Child glioma diagnosis: ", child_diagnosis_glioma)
      
child_glioma = file_csv[(age < 18) & (diagnosis.isin(child_diagnosis_glioma))]
print("Number of child glioma: ", len(child_glioma))

# TO DO: Trouver corresponde classification 2016 and 2021

 
