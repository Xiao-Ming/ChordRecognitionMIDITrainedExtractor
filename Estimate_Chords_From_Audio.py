#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:08:30 2018

@author: wuyiming
"""

import numpy as np
import networks as N
from librosa.util import find_files
from librosa.core import cqt,load,note_to_hz
import os
import const as C
import utils as U
import argparse


parser = argparse.ArgumentParser(description="Estimate chords from audio files with pre-trained model.")
parser.add_argument("-f",help="Name of the feature extractor parameter file.(default: 'fullcnn_crossentropy_6000.model')",\
                    type=str, default=C.DEFAULT_CONVNETFILE, action="store")

parser.add_argument("-d", help="Name of the BLSTM-CRF decoder parameter file. (default: 'nblstm_crf.model')",\
                    type=str, default="nblstm_crf.model", action="store")
args = parser.parse_args()

audio_list = find_files("Datas/audios_estimation")

for audiofile in audio_list:
    fname = audiofile.split("/")[-1]
    print("Processing: %s" % fname)
    #load audio
    y,sr = load(audiofile,sr=C.SR)
    
    #extract Harmonic-CQT from audio
    fmin = note_to_hz("C1")
    hcqt = np.stack([np.abs(cqt(y,sr=C.SR,hop_length=C.H,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1),filter_scale=2,tuning=None)).T.astype(np.float32) for h in range(C.CQT_H)])
    
    #extract feature using trained CNN extractor
    cnn_feat_extractor = N.FullCNNFeatExtractor()
    cnn_feat_extractor.load(args.f)
    
    feat = cnn_feat_extractor.GetFeature(U.PreprocessSpec(hcqt)).data
    
    #decode label sequence
    decoder = N.NBLSTMCRF()
    decoder.load(args.d)
    
    labels = decoder.argmax(feat)
    
    #convert into .lab file
    labfile = os.path.join("Datas/labs_estimated",fname+".lab")
    U.SaveEstimatedLabelsFramewise(labels,labfile,feat)
