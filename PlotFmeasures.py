#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:19:29 2018

@author: wuyiming
"""

import numpy as np
import matplotlib.pyplot as plt

fmeasure = np.load("evaltrans_f.npz")

f_chroma = fmeasure["chroma"]
f_chroma[1] = 0.669254
f_chroma = np.append(f_chroma,0.8175)
f_bass = fmeasure["bass"]
f_bass = np.append(f_bass,0.8256)
f_top = np.array([0.50442277,0.52501013,0.53368053,0.54230557,0.56408992,0.57241088,0.58389945,0.61236426,0.61674062,0.627763,0.63700963,0.6435])


midisizelist = np.array([100,200,300,500,700,1000,1500,2000,3000,4000,5000,6000])

plt.subplot(1,3,1)
plt.plot(midisizelist,f_bass,"b-o")
plt.ylim(0.7,0.85)
plt.xlabel("trainset size")
plt.ylabel("F-measure")
plt.title("bass")

plt.subplot(1,3,2)
plt.plot(midisizelist,f_chroma,"b-o")
plt.ylim(0.6,0.85)
plt.xlabel("trainset size")
plt.ylabel("F-measure")
plt.title("middle")

plt.subplot(1,3,3)
plt.plot(midisizelist,f_top,"b-o")
plt.ylim(0.45,0.7)
plt.xlabel("trainset size")
plt.ylabel("F-measure")
plt.title("top")