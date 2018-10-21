#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:13:37 2018

@author: wuyiming
"""

import networks as N
from librosa.core import cqt,load,note_to_hz
import const as C
import numpy as np

cnn = N.FullCNNFeatExtractor()
cnn.load("fullcnn_crossentropy_6000.model")

y,sr = load("audio.wav",sr=C.SR)

fmin = fmin = note_to_hz("C1")
spec = np.stack([np.abs(cqt(y,sr=C.SR,hop_length=C.H,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1),filter_scale=2,tuning=None)).T.astype(np.float32) for h in range(C.CQT_H)])

feature = cnn.GetFeature(spec)