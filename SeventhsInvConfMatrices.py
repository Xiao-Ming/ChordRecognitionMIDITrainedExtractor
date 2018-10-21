#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:33:48 2018

@author: wuyiming
"""

import evaluation
import numpy as np
import matplotlib.pyplot as plt
import const
from librosa.util import find_files
import itertools

idx_isophonics = np.arange(217)
idx_rwc = np.arange(217,317)

mat_iso_chr,mat_iso_sev,mat_iso_inv = evaluation.ConfMatrix(idx_isophonics,True)

mat_rwc_chr,mat_rwc_sev,mat_rwc_inv = evaluation.ConfMatrix(idx_rwc,True)


def plotConfMat(mat,classes,title):
    mat = mat / mat.sum(axis=1)[:,np.newaxis]
    plt.figure(title,dpi=500)
    plt.imshow(mat,interpolation="nearest",cmap=plt.cm.Blues)
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes)
    plt.yticks(tick_marks,classes)
    for i,j in itertools.product(range(3),range(3)):
        plt.text(j,i,format(mat[i,j],".3f"),horizontalalignment="center",color="white" if mat[i,j]>0.5 else "black")
    #plt.tight_layout()
    plt.show()
    
c_sev = ["triad","7","maj7"]
c_inv = ["root","first","second"]

plotConfMat(mat_iso_sev,c_sev,"isophonics-sevenths")
plotConfMat(mat_iso_inv,c_inv,"isophonics-inv")
plotConfMat(mat_rwc_sev,c_sev,"rwc-sevenths")
plotConfMat(mat_rwc_inv,c_inv,"rwc-inv")