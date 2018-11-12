#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用法

Created on Thu Oct 12 23:50:30 2017

@author: wuyiming
"""

import training
import numpy as np
import evaluation
import const as C
import utils

#training.TrainConvnetExtractor(np.arange(500),epoch=15,saveas=C.DEFAULT_CONVNETFILE)

#training.TrainDNNExtractor(np.arange(1500),epoch=20,saveas=C.DEFAULT_DNNFILE)

idx_train = np.arange(320,dtype=np.int32)
#training.TrainConvnetExtractorDeepChroma(dataidx["train"],epoch=10,saveas=C.DEFAULT_CONVNETFILE_DEEPCHROMA)
print("Training RNN...")
#training.TrainRNN(dataidx["train"],epoch=20)
training.TrainNStepRNN(idx_train,epoch=100,featmodel=C.DEFAULT_CONVNETFILE,augment=6)
#print("Training Classifier...")
#training.TrainConvClassifier(dataidx["train"],epoch=20)
print("Training CRF...")
#training.TrainCRF(dataidx["train"],epoch=10)
training.TrainNStepCRF(idx_train,epoch=10,featmodel=C.DEFAULT_CONVNETFILE,augment=6)
#training.TrainConvCRF(dataidx["train"],epoch=10)



print("Estimating....")
utils.ClearEstimate(False)
evaluation.EstimateChord(idx_train[:50],C.DEFAULT_CONVNETFILE,todir=False)

score_majmin,score_sevenths,score_majmininv,score_seventhinv,confmatrix,durations = evaluation.EvaluateChord(idx_train[:50])

print("majmin:%.2f" % (np.dot(score_majmin,durations)/np.sum(durations)*100))

print("sevenths:%.2f" % (np.dot(score_sevenths,durations)/np.sum(durations)*100))

print("majmininv:%.2f" % (np.dot(score_majmininv,durations)/np.sum(durations)*100))

print("seventhinv:%.2f" % (np.dot(score_seventhinv,durations)/np.sum(durations)*100))


