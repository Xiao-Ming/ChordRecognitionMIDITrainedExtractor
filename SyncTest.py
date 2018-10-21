# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:26:37 2016

@author: wuyiming
"""

from librosa.util import find_files
from librosa.core import load,midi_to_hz
from librosa.output import write_wav
from librosa.filters import cq_to_chroma
import mir_eval.sonify as sonify
import chainer
import chainer.functions as F
import utils
import numpy as np
import const
import networks
import cupy as cp


cqtfilelist = find_files(const.PATH_HCQT,ext="npy")[20:30]
audiofilelist = find_files(const.PATH_AUDIO,ext="wav")[20:30]


chainer.config.train=False
chainer.config.enable_backprop = False
dnn = networks.ConvnetFeatExtractor()
dnn.load("convnet_binarytarget_2000_weightdecay.model")
#dnn.to_gpu(0)

for i in range(len(cqtfilelist)):
    filename = audiofilelist[i].split("/")[-1]
    print("processing: " + filename)
    audio,sr = load(audiofilelist[i],sr=const.SR)    
    cqt = utils.PreprocessSpec(np.load(cqtfilelist[i]))
    y = dnn.GetFeature(cqt).data[:,0:12]
    #y[y>=0.5] = 1
    #y[y<0.5] = 0
    chroma_sonify = sonify.chroma(y.T,np.arange(y.shape[0]).astype(np.float32)*const.H/const.SR,const.SR)
    sz = min([chroma_sonify.shape[0],audio.shape[0]])
    write_wav(const.PATH_SONIFYTEST+filename,np.vstack((audio[:sz],chroma_sonify[:sz])),sr)
