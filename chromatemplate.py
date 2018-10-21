# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:14:26 2016

@author: wuyiming
"""

import numpy as np
from librosa.filters import cq_to_chroma
from librosa.core import midi_to_hz
from librosa.util import normalize
import itertools

bitmap_template = np.zeros(12,dtype="int32")
for i in range(1,4):
    for j in list(itertools.combinations(range(12),i)):
        n = np.zeros(12,dtype="int32")
        for k in list(j):
            n[k] = 1
        bitmap_template = np.vstack((bitmap_template,n))


def _getlabel(chroma):
    chroma[chroma>0] = 1
    mat = (chroma==bitmap_template)
    idx = np.where(np.sum(mat,axis=1)==12)[0][0]
    return idx
    
def GetLabelarrFromPianoroll(arr_pianoroll):
    chroma_filter = cq_to_chroma(n_input = 128, fmin = midi_to_hz(0)).T
    chroma = np.dot(arr_pianoroll,chroma_filter).astype("int32")
    
    lab = np.zeros(chroma.shape[0],dtype="int32")
    for i in range(chroma.shape[0]):
        c = chroma[i,:]
        note = arr_pianoroll[i,:]
        bass = np.where(note>0)[0]
        if bass.size>0:
            c[np.min(bass)%12] = 20     #Force bass note to remain in template
        c_argsort = np.argsort(c,kind="mergesort")
        c[c_argsort[:9]] = 0
        lab[i] = _getlabel(c)
            
    return lab

def GetChromaFromPianoroll(arr_pianoroll,bass=False):
    chroma_filter = cq_to_chroma(n_input=128,fmin=midi_to_hz(0))
    chroma = np.dot(chroma_filter,arr_pianoroll.T).T.astype("float32")
    chroma[chroma>0] = 1
    if not bass:
        return chroma
    bass_chroma = np.zeros(shape=chroma.shape,dtype="float32")
    for t in range(bass_chroma.shape[0]):
        notes = arr_pianoroll[t,:]
        if np.sum(notes)>0:
            bass = np.min(np.where(notes>0)[0])
            bass_chroma[i,bass%12] = 1
    return np.concatenate([bass_chroma,chroma],axis=1)

def GetMultiFeatureFromPianoroll(arr_pianoroll):
    chroma_filter = cq_to_chroma(n_input=128,fmin=midi_to_hz(0))
    T = arr_pianoroll.shape[0]
    bass_chroma = np.zeros((T,12),dtype="float32")
    top_chroma = np.zeros((T,12),dtype="float32")
    for t in range(T):
        sum_notes = np.sum(arr_pianoroll[t,:])
        if sum_notes==0:
            continue
        notes = np.where(arr_pianoroll[t,:]>0)[0]
        bassnote = np.min(notes)
        topnote = np.max(notes)
        arr_pianoroll[t,bassnote]=0
        arr_pianoroll[t,topnote]=0
        bass_chroma[t,bassnote%12]=1
        top_chroma[t,topnote%12]=1
    
    chroma = np.dot(chroma_filter,arr_pianoroll.T).T.astype("float32")
    feature = np.zeros((T,12),dtype="float32")
    feature[chroma>=1] = 1.0
    feature = np.concatenate([bass_chroma,feature],axis=1)
    feature = np.concatenate([feature,top_chroma],axis=1)
    
    return feature

def GetConvnetTargetFromPianoroll(arr_pianoroll):
    chroma_filter = cq_to_chroma(n_input=128,fmin=midi_to_hz(0))
    T = arr_pianoroll.shape[0]
    bass_chroma = np.zeros((T,12),dtype="float32")
    top_chroma = np.zeros((T,12),dtype="float32")
    for t in range(T):
        sum_notes = np.sum(arr_pianoroll[t,:])
        if sum_notes==0:
            continue
        notes = np.where(arr_pianoroll[t,:]>0)[0]
        bassnote = np.min(notes)
        topnote = np.max(notes)
        arr_pianoroll[t,bassnote]=0
        arr_pianoroll[t,topnote]=0
        bass_chroma[t,bassnote%12]=1
        top_chroma[t,topnote%12]=1
    
    chroma = np.dot(chroma_filter,arr_pianoroll.T).T
    feature = np.zeros((T,12),dtype="float32")
    #feature = normalize(chroma,axis=1,norm=np.inf).astype("float32")
    feature[chroma>=1] = 1.0
    feature = np.concatenate([bass_chroma,feature,top_chroma],axis=1)
    return feature
def GetTargetsFromPianoroll(arr_pianoroll):
    chroma_filter = cq_to_chroma(n_input=128,fmin=midi_to_hz(0))
    T = arr_pianoroll.shape[0]
    t_bass = np.zeros(T,dtype="int32")
    t_top = np.zeros(T,dtype="int32")
    for t in range(T):
        sum_notes = np.sum(arr_pianoroll[t,:])
        if sum_notes==0:
            continue
        notes = np.where(arr_pianoroll[t,:]>0)[0]
        bassnote = np.min(notes)
        topnote = np.max(notes)
        arr_pianoroll[t,bassnote]=0
        arr_pianoroll[t,topnote]=0
        t_bass[t]=bassnote%12
        t_top[t]=topnote%12
    
    chroma = np.dot(chroma_filter,arr_pianoroll.T).T.astype("float32")
    feature = np.zeros((T,12),dtype="float32")
    feature[chroma>=1] = 1.0

    return t_bass,feature,t_top    

def GetTemplateChromaFromPianoroll(arr_pianoroll,bass=False):
    chroma_filter = cq_to_chroma(n_input = 128, fmin = midi_to_hz(0)).T
    chroma = np.dot(arr_pianoroll,chroma_filter).astype("int32")
    chroma_template = np.zeros(chroma.shape,dtype="int32")
    bass_template = np.zeros(chroma.shape,dtype="int32")
    for i in range(chroma.shape[0]):
        c = chroma[i,:]
        note = arr_pianoroll[i,:]
        bass = np.where(note>0)[0]
        if bass.size>0:
            if not bass:
                c[np.min(bass)%12] = 20     #Force bass note to remain in template
            bass_template[i,np.min(bass)] = 1
        c_argsort = np.argsort(c,kind="mergesort")
        c[c_argsort[:9]] = 0
        c[c>0] = 1
        chroma_template[i,:] = c
    if bass:
        return np.concatenate([chroma_template,bass_template],axis=1)
    else:        
        return chroma_template
