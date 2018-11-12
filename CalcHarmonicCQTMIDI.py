#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:47:11 2017

@author: wuyiming
"""

import numpy as np
from librosa.core import cqt,load,note_to_hz
from librosa.util import find_files
import pretty_midi
import const as C
import utils
import chromatemplate

midilist = find_files(C.PATH_MIDI,ext="mid")

itemcnt = len(midilist)
i = 5000
for midifile in midilist:
    i += 1
    print("Processing %d" % (i))    
    wav,sr = load(midifile,sr=C.SR)
    try:
        pm = pretty_midi.PrettyMIDI(midifile)
        wav = pm.fluidsynth(fs=C.SR)
        target = chromatemplate.GetConvnetTargetFromPianoroll(utils.GetPianoroll(midifile))
        fmin = note_to_hz("C1")
        spec = np.vstack([np.abs(cqt(wav,sr=C.SR,hop_length=C.H,n_bins=C.BIN_CNT,bins_per_octave=C.OCT_BIN,fmin=fmin*(h+1))).T.astype(np.float32) for h in range(C.CQT_H)])
    except(Exception):
        print("Got error.Skip...")
        continue
    
    
    
    minsz = min([spec.shape[1],target.shape[1]])
    np.savez(C.PATH_MIDIHCQT+"additional/" + "%06d.npz" % i,spec=spec[:minsz],target=target[:minsz])