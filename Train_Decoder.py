#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:00:02 2018

@author: wuyiming
"""

import training
import const as C
from librosa.util import find_files

import argparse

parser = argparse.ArgumentParser(description="Trains the BLSTM-CRF chord decoder with provided data.")
parser.add_argument("-f",help="Name of the feature extractor parameter file.(default: 'fullcnn_crossentropy_6000.model')",\
                    type=str, default=C.DEFAULT_CONVNETFILE, action="store")

parser.add_argument("-s", help="Name of the BLSTM-CRF decoder parameter file. (default: 'nblstm_crf_selftrain.model')",\
                    type=str, default="nblstm_crf_selftrain.model", action="store")

parser.add_argument("-a", help="Maximum amount of pitch shift applied to the training data. \
                    Default value is set to 5 (which means training datas are shifted for random amount between -5 and 5 semitones).",\
                    type=int, default=5, choices=[0,1,2,3,4,5,6], action="store")

parser.add_argument("-e", help="Maximum training epochs for BLSTM and CRF model. 2 numbers are required. (default: 100,10)", \
                    type=int, nargs=2, action="store", default=[100,10])

args = parser.parse_args()


#Check datas

spec_filelist = find_files("Datas/specs_train", ext="npy")
labs_filelist = find_files("Datas/labels_train", ext=["lab","chords"])


assert len(spec_filelist)>0, "Spectrogram datas should be prepared before running training!"
assert len(spec_filelist)==len(labs_filelist), "Number of feature and label files should be the same."


training.TrainNStepRNN(None, epoch=args.e[0], featmodel=args.f, augment=args.a)

training.TrainNStepCRF(None, epoch=args.e[1], featmodel=args.f, augment=args.a, savefile=args.s)
