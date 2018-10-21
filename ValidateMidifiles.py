#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:26:14 2017

@author: wuyiming
"""

from librosa.util import find_files
from utils import GetPianoroll


midilist = find_files("/home/wuyiming/Projects/TranscriptionChordRecognition/Datas/midi",ext="mid")

print("%d files." % len(midilist))

for f in midilist:
    try:
        pianoroll = GetPianoroll(f)
    except(Exception):
        print("Got error on %s" % f.split("/")[-1])
        