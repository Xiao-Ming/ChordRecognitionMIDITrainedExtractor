# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:41:03 2016

@author: wuyiming
"""

import training
import numpy as np
import evaluation
import time
import const as C

idx = np.load("fold.npy",encoding="bytes")



for rnd in range(6):
    print("FOLD %d:" % (rnd+1))
    idx_test = idx[0]
    idx_train = np.concatenate(idx[1:])
    #training.TrainTranscribeDNNChord(idx_train)
    training.TrainNStepRNN(idx_train,epoch=10,modelfile="fullcnn_crossentropy_6000.model",augment=12)
    training.TrainNStepCRF(idx_train,epoch=5,modelfile="fullcnn_crossentropy_6000.model",augment=12)
    evaluation.EstimateChord(idx_test,todir=True,dnnmodel="fullcnn_crossentropy_6000.model")
    idx = np.roll(idx,1)


timestamp = int(time.time())
print("Estimation complete!")
logfile = open("logs/log_proposed_%d.log" % timestamp,"w")

def writelog(f,text):
    print(text)
    f.write(text+"\n")

idx_isophonics = np.arange(213)
idx_rwc = np.arange(213,313)

writelog(logfile,"DeepChroma Method")
writelog(logfile,"train seq length: %d" % C.DECODER_TRAIN_SEQLEN)
writelog(logfile,"train batch size: %d" % C.DECODER_TRAIN_BATCH)
writelog(logfile, "Beatles:")

score_majmin,score_sevenths,score_majmininv,score_seventhinv,confmatrix,durations = evaluation.EvaluateChord(idx_isophonics,verbose=False,cross=True)

writelog(logfile,"majmin:%.2f" % (np.dot(score_majmin,durations)/np.sum(durations)*100))
writelog(logfile,"sevenths:%.2f" % (np.dot(score_sevenths,durations)/np.sum(durations)*100))
writelog(logfile,"majmininv:%.2f" % (np.dot(score_majmininv,durations)/np.sum(durations)*100))
writelog(logfile,"seventhinv:%.2f" % (np.dot(score_seventhinv,durations)/np.sum(durations)*100))


conf_iso_c,conf_iso_q,conf_iso_b = evaluation.ConfMatrix(idx_isophonics,cross=True)

writelog(logfile, "RWC:")
score_majmin,score_sevenths,score_majmininv,score_seventhinv,confmatrix,durations = evaluation.EvaluateChord(idx_rwc,verbose=False,cross=True)

writelog(logfile,"majmin:%.2f" % (np.dot(score_majmin,durations)/np.sum(durations)*100))
writelog(logfile,"sevenths:%.2f" % (np.dot(score_sevenths,durations)/np.sum(durations)*100))
writelog(logfile,"majmininv:%.2f" % (np.dot(score_majmininv,durations)/np.sum(durations)*100))
writelog(logfile,"seventhinv:%.2f" % (np.dot(score_seventhinv,durations)/np.sum(durations)*100))

conf_rwc_c,conf_rwc_q,conf_rwc_b = evaluation.ConfMatrix(idx_rwc,cross=True)

conf_c = conf_iso_c+conf_rwc_c
conf_q = conf_iso_q+conf_rwc_q
conf_b = conf_iso_b+conf_rwc_b

conf_c = conf_c / conf_c.sum(axis=1)[:,np.newaxis]
conf_q = conf_q / conf_q.sum(axis=1)[:,np.newaxis]
conf_b = conf_b / conf_b.sum(axis=1)[:,np.newaxis]

np.savez("logs/confmatrix_%d.npz" % timestamp,chord=conf_c,quality=conf_q,bass=conf_b)
print("conf matrix:")
#print(conf_c)
print(conf_q)
print(conf_b)
logfile.close()
print("Done!")
