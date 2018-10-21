#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:01:18 2018

@author: wuyiming
"""

from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer
import numpy as np
from librosa.core import stft,load,magphase
import scipy.signal

        
def ConvSTFT(y,size=2048,window="hann",hopsize=2048):
    W = np.zeros(((size+1)//2,1,size),dtype=np.complex64)
    win = scipy.signal.get_window(window,size,fftbins=False)
    for k in range(W.shape[0]):
        W[k,0,:] = np.exp(-2j*np.pi*k*np.arange(size)/size) * win
        
    mag = F.convolution_nd(y,np.real(W),b=None,stride=hopsize)
    mag2 = F.convolution_nd(y.astype(np.complex64),W,b=None,stride=hopsize)
    return mag.data,mag2.data

            
y,sr = load("letitbe.wav",sr=None)
chainer.config.train=False
chainer.config.type_check=False

mag,mag2 = ConvSTFT(y[None,None,:])

libspec = stft(y,2048,2048,center=False)
