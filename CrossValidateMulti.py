#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 22:04:27 2018

@author: wuyiming
"""

import training
import numpy as np
import evaluation
import time
import const as C
import utils

idx = np.load("fold.npy",encoding="bytes")

midisize = [2500,3000,4000,5000,6000]
#list_score_rwc = []
#list_score_isophonics = []
for size in midisize:
    modelname = "fullcnn_crossentropy_%d.model" % size
    utils.ClearEstimate(cross=True)
    for rnd in range(6):
        print("FOLD %d:" % (rnd+1))
        idx_test = idx[0]
        idx_train = np.concatenate(idx[1:])
        #training.TrainTranscribeDNNChord(idx_train)
        training.TrainNStepRNN(idx_train,epoch=20,modelfile=modelname,augment=12)
        training.TrainNStepCRF(idx_train,epoch=5,modelfile=modelname,augment=12)
        evaluation.EstimateChord(idx_test,todir=True,dnnmodel=modelname)
        idx = np.roll(idx,1)
    
    
    print("Estimation complete!")
    idx_isophonics = np.arange(213)
    idx_rwc = np.arange(213,313)
    
    score_majmin,score_sevenths,score_majmininv,score_seventhinv,confmatrix,durations = evaluation.EvaluateChord(idx_isophonics,verbose=False,cross=True)
    
    print("majmin:%.2f" % (np.dot(score_majmin,durations)/np.sum(durations)*100))
    list_score_isophonics.append(score_majmin)
    
    print("RWC:")
    score_majmin,score_sevenths,score_majmininv,score_seventhinv,confmatrix,durations = evaluation.EvaluateChord(idx_rwc,verbose=False,cross=True)
    print("majmin:%.2f" % (np.dot(score_majmin,durations)/np.sum(durations)*100))
    
    
    list_score_rwc.append(score_majmin)