#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:39:26 2018

@author: wuyiming
"""

import numpy as np
import mir_eval
from librosa.util import find_files
import const

labfilelist = np.array(find_files(const.PATH_CHORDLAB,ext="lab"))[195:295]

durations = {}

for labfile in labfilelist:
    (intervals,labels) = mir_eval.io.load_labeled_intervals(labfile)
    for interval,label in zip(intervals,labels):
        duration = interval[1] - interval[0]
        split = mir_eval.chord.split(label)
        qual = split[1]
        if split[3] != "1":
            qual = qual + "/" + split[3]
        
        if qual in durations.keys():
            durations[qual] += duration
        else:
            durations[qual] = duration

dur_total = sum(durations.values())

for k in durations.keys():
    durations[k] = durations[k] / dur_total * 100
    
    