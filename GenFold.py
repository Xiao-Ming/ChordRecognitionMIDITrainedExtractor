# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:49:19 2016

@author: wuyiming
"""

import numpy as np

rand = np.random.permutation(313)

idx = np.array([rand[0:52],rand[52:104],rand[104:156],rand[156:208],rand[208:260],rand[260:313]])
#idx = np.array([rand[0:25],rand[25:50],rand[50:75],rand[75:100],rand[100:125],rand[125:150],rand[150:175],rand[175:202]])
np.save("fold.npy",idx)