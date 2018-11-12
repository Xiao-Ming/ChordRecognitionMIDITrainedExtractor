#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 22:17:09 2017

@author: wuyiming
"""

import numpy as np
from custommidifile import CustomMIDIFile
import const
import ChordVocabulary as voc
from madmom.utils.midi import MIDIFile
from madmom.utils import suppress_warnings
from librosa.util import normalize,find_files
import cupy
import os

def force_numpy(x):
    xp = cupy.get_array_module(x)
    if xp==np:
        return np.array(x)
    elif xp==cupy:
        return cupy.asnumpy(x)
    return None

def PreprocessSpec(x):
    x_log = np.log(np.maximum(x,0.00001))
    return (x_log - np.mean(x_log)) / np.var(x_log)

def MeanVarNormalize(x):
    #_x = np.array(x)
    #_x[:,12:24] = normalize(x[:,12:24],axis=1)
    #return _x
    return x

def GetPianoroll(midifilename):    
    midifile = MIDIFile.from_file(midifilename)       
    notes = suppress_warnings(midifile.notes)()
    #print (midifilename,notes.shape)
    notes[:,0] = np.round(notes[:,0]*const.SR/const.H)
    notes[:,2] = np.round(notes[:,2]*const.SR/const.H)
    endtime = int(notes[-1,0]+notes[-1,2])
    notearr = np.zeros((endtime+10,128),dtype="int32")
    
    for note in notes:
        if note[4] in [9,10]:
            continue
        st = int(note[0])
        ed = st+int(note[2])
        notearr[st:ed,int(note[1])] += 1
        
    return notearr

    
def LoadLabelArr(labfile,hop_size=const.H):
    f = open(labfile)
    line = f.readline()
    labarr = np.zeros(800*const.SR//hop_size,dtype="int32")
    while line != "" and line.isspace()==False:
        items = line.split()
        st = int(round(float(items[0])*const.SR/hop_size))
        ed = int(round(float(items[1])*const.SR/hop_size))
        lab = voc.GetChordIDSign(items[2])
        labarr[st:ed] = lab
        line = f.readline()
    return labarr[:ed]
    
def SaveEstimatedLabelsFramewise(est,filename,feature=None):
    f = open(filename,"w")
    cur_label = est[0]
    st = 0
    ed = 1
    for i in range(est.size):
        if est[i] != cur_label:
            ed = i
            if feature is not None:
                feat = feature[st:ed,:]
                sign = voc.ChordSignature7thbass(cur_label,feat,sevenths=True,inv=True)
            else:
                sign = voc.ChordSignature(cur_label)
            text = "%.4f %.4f %s\n" % (float(st*const.H)/const.SR,float(ed*const.H)/const.SR,sign)
            f.write(text)
            cur_label = est[i]
            st = i
    if feature is not None:
        feat = feature[st:est.size,:]
        text = "%.4f %.4f %s" % (float(st*const.H)/const.SR,float(est.size*const.H)/const.SR,voc.ChordSignature7thbass(cur_label,feat))
    else:
        text = "%.4f %.4f %s" % (float(st*const.H)/const.SR,float(est.size*const.H)/const.SR,voc.ChordSignature(cur_label))
    f.write(text)
    f.close()
    
def ClearEstimate(cross):
    path = const.PATH_ESTIMATE_CROSS if cross else const.PATH_ESTIMATE
    for f in find_files(path,ext="lab"):
        os.remove(f)

def Embed(X,size=5):
    if size==1:
        return X
    d = int(size)//2
    pad_1 = np.array([X[0,:] for i in range(d)])
    pad_2 = np.array([X[-1,:] for i in range(d)])
    X_padded = np.vstack((pad_1,X,pad_2))
    origin_len = X_padded.shape[0]
    newX = np.array([X_padded[i-d:i+d+1,:].flatten() for i in range(d,origin_len-d)]).astype(np.float32)
    return newX