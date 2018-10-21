# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:57:34 2016

@author: mirlearning
"""

import numpy as np
from librosa.util import find_files
from Global import *

itemcnt = 313
test_size = itemcnt//5

rand = np.random.permutation(itemcnt).astype(np.int32)


np.savez("dataidx.npz",train=rand[test_size:],test=rand[:test_size])