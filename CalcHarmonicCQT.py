#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:38:10 2017

@author: wuyiming
"""

import numpy as np
from librosa.core import cqt,load,note_to_hz
from librosa.util import find_files
import const as C


audiolist = find_files(C.PATH_AUDIO)[800:]

path_hcqt = "/media/wuyiming/TOSHIBA EXT/HCQT_12_hop512/"
itemcnt = len(audiolist)
i = 0
for audiofile in audiolist:
    i += 1
    print("Processing %d/%d" % (i,itemcnt))    
    wav,sr = load(audiofile,sr=C.SR)
    fmin = note_to_hz("C1")
    spec = np.stack([np.abs(cqt(wav,sr=C.SR,hop_length=512,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1),filter_scale=2,tuning=None)).T.astype(np.float32) for h in range(C.CQT_H)])
    
    
    filename = audiofile.split('/')[-1].split(".")[0]
    albname = audiofile.split('/')[-2]
    
    np.save(path_hcqt+albname+'/'+filename+'.npy',spec)
