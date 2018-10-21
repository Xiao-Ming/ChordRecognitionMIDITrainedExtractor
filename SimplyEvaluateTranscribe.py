#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:34:44 2017

@author: wuyiming
"""

import networks
from librosa.util import find_files
import numpy as np
import const

def evaluate(predicted,target):
    predicted = np.concatenate(predicted)
    target = np.concatenate(target)
    hits = (predicted & target)
    
    precision = float(np.sum(predicted & hits))/np.sum(predicted)
    recall = float(np.sum(target & hits))/np.sum(target)
    accr = float(np.sum(predicted==target))/predicted.size
    f_measure = (2*precision*recall)/(precision+recall)
    return precision,recall,accr,f_measure   