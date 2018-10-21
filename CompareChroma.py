#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:55:12 2018

@author: wuyiming
"""

import networks
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load,cqt,note_to_hz
from librosa.feature import chroma_cqt
from librosa.display import specshow
import const as C
from chainer import config
import utils as U
import chromatemplate


config.train=False
config.enable_backprop = False

audiofile = "/home/wuyiming/Projects/ChordData/Audio/16_RWC/050.wav"

wav,sr = load(audiofile,sr=C.SR)
fmin = note_to_hz("C1")
spec = U.PreprocessSpec(np.stack([np.abs(cqt(wav,sr=C.SR,hop_length=C.H,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1),filter_scale=2,tuning=None)).T.astype(np.float32) for h in range(C.CQT_H)]))
spec_dnn = U.Embed(U.PreprocessSpec(np.abs(cqt(wav,sr=C.SR,hop_length=C.H,n_bins=144,bins_per_octave=24,filter_scale=2,tuning=None)).T.astype(np.float32)),size=1)

#dat = np.load("/media/wuyiming/TOSHIBA EXT/midihcqt_12/000005.npy")
#dat_24 = np.load("/media/wuyiming/TOSHIBA EXT/midihcqt_24/000005.npz")
#spec_dnn = U.Embed(U.PreprocessSpec(dat_24["spec"]),size=7)

spec = spec[:,:250,:]
spec_dnn = spec_dnn[:250,:]
cnn = networks.FullCNNFeatExtractor()
cnn.load("fullcnn_crossentropy_6000.model")

deepchroma = networks.FeatureDNN()
deepchroma.load("/home/wuyiming/Projects/TranscriptionChordRecognition/dnn3500.model")

chroma_cnn = cnn.GetFeature(spec).data[:,12:24].T
chroma_dnn = deepchroma.GetFeature(spec_dnn).data[:,12:24].T
chroma = np.log(1+chroma_cqt(wav,sr=C.SR,hop_length=C.H,bins_per_octave=24)[:,:250])

target = chromatemplate.GetConvnetTargetFromPianoroll(U.GetPianoroll("/media/wuyiming/TOSHIBA EXT/AIST.RWC-MDB-P-2001.SMF_SYNC/RM-P051.SMF_SYNC.MID"))
target = target[10:260,12:24].T

plt.subplot(4,1,1)
specshow(chroma,y_axis="chroma")
plt.ylabel("(a)")
plt.subplot(4,1,2)
specshow(chroma_dnn,y_axis="chroma")
plt.ylabel("(b)")
plt.subplot(4,1,3)
specshow(chroma_cnn,y_axis="chroma")
plt.ylabel("(c)")
plt.subplot(4,1,4)
specshow(target,y_axis="chroma")
plt.ylabel("ground-truth")