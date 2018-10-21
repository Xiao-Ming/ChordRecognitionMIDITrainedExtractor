# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:03:36 2016

@author: wuyiming
"""

import training
import networks
import utils
from librosa.util import find_files
import numpy as np
import const
import chainer.functions as F
import chromatemplate
import cupy as cp
import chainer

def evaluate(predicted,target):
    predicted = np.concatenate(predicted)
    target = np.concatenate(target)
    assert(predicted.shape==target.shape)
    hits = (predicted & target)
    precision = float(np.sum(predicted & hits))/np.sum(predicted)
    recall = float(np.sum(target & hits))/np.sum(target)
    accr = float(np.sum(predicted==target))/predicted.size
    f_measure = (2*precision*recall)/(precision+recall)
    return precision,recall,accr,f_measure   

def EvaluateConvnet(modelfile,cqtfilelist):
    predicted_chroma = []
    target_chroma = []
    predicted_bass = []
    target_bass = []
    predicted_top = []
    target_top = []
    #dnn = networks.ConvnetFeatExtractor()
    dnn = networks.FeatureDNN()
    #dnn = networks.FullCNNFeatExtractor()
    dnn.load(modelfile)
    dnn.to_gpu(0)
    chainer.config.train=False
    chainer.config.enable_backprop = False
    
    for cqtfile in cqtfilelist:
        #print(cqtfile)
        dat = np.load(cqtfile)
        cqt = utils.PreprocessSpec(dat["spec"][:,:])
        t = chromatemplate.GetConvnetTargetFromPianoroll(dat["target"])
        #assert(cqt.shape[0]==t.shape[0])
        out = cp.asnumpy(dnn.GetFeature(cp.asarray(cqt)).data)
        #out = dnn.GetFeature(cqt).data
        out_len = out.shape[0]
        pred_chroma = np.zeros((out_len,12),dtype="int32")
        pred_bass = np.zeros((out_len,12),dtype="int32")
        pred_top = np.zeros((out_len,12),dtype="int32")
        pred_chroma[out[:,12:24]>0.5] = 1
        pred_bass[np.arange(out_len),np.argmax(out[:,:12],axis=1)] = 1
        #pred_bass[out[:,:12]<=0.5] = 0
        pred_top[np.arange(out_len),np.argmax(out[:,24:36],axis=1)] = 1
        #pred_top[out[:,24:36]<=0.5] = 0
        predicted_chroma.append(pred_chroma[:,:].astype(np.bool))
        target_chroma.append(t[:out_len,12:24].astype(np.bool))
        predicted_bass.append(pred_bass[:out_len,:].astype(np.bool))
        target_bass.append(t[:out_len,:12].astype(np.bool))
        predicted_top.append(pred_top[:out_len,:].astype(np.bool))
        target_top.append(t[:out_len,24:36].astype(np.bool))
        
    eval_chroma = evaluate(predicted_chroma,target_chroma)
    eval_bass = evaluate(predicted_bass,target_bass)
    eval_top = evaluate(predicted_top,target_top)
    return eval_chroma,eval_bass,eval_top

cqtfilelist = np.array(find_files(const.PATH_MIDIHCQT,ext="npz"))

rand = np.arange(len(cqtfilelist))
idx_test = rand[3000:3500]

midisizelist = np.array([2000],dtype="int32")  
f_chroma = np.array([])
f_top = np.array([])
f_bass = np.array([])
accr_chroma = np.array([])
accr_top = np.array([])
accr_bass = np.array([])

for midisize in midisizelist:    
    print("train size:%d" % midisize)
    idx_train = rand[:midisize]
    filename = "DRN_crossentropy_%d.model" % midisize
    #filename = "fullcnn_crossentropy_%d.model" % midisize
    training.TrainDNNExtractor(idx_train,epoch=20,saveas=filename)
    #training.TrainConvnetExtractor(idx_train,epoch=10,saveas=filename)
    print("Evaluation...")
    eval_chroma,eval_bass,eval_top = EvaluateConvnet(filename,cqtfilelist[idx_test])
    
    f_chroma = np.append(f_chroma,eval_chroma[3])
    f_bass = np.append(f_bass,eval_bass[3])
    f_top = np.append(f_top,eval_top[3])
    accr_chroma = np.append(accr_chroma,eval_chroma[2])
    accr_bass = np.append(accr_bass,eval_bass[2])
    print("results for size %d:" % (midisize))
    
    print(eval_bass)
    print(eval_chroma)
    print(eval_top)


print (f_bass)
print(f_chroma)
print(f_top)


#np.savez("evaltrans_f.npz",chroma=f_chroma,bass=f_bass,midisize=midisizelist)
#np.savez("evaltrans_accr.npz",chroma=accr_chroma,bass=accr_bass)

