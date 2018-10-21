# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:10:16 2016

@author: wuyiming
"""

import vamp
from librosa.core import load
from librosa.util import find_files
import const
import ChordVocabulary as voc
import numpy as np
import mir_eval
import ChordinoChordTransform as transform

PATH_ESTIMATE_CHORDINO = "chordino/19_Billboard"

audio_list = find_files(const.PATH_AUDIO+"/19_Billboard")
itemcnt = len(audio_list)
i = 0

for audiofile in audio_list:
    i+=1
    print("estimate %d/%d" % (i,itemcnt))
    wav,sr=load(audiofile,sr=None)
    filename = audiofile.split("/")[-1].split(".")[0]
    chordlist = vamp.collect(wav,sr,"nnls-chroma:chordino")["list"]
    time_start = 0
    l = chordlist[0]["label"]
    estfile = open(PATH_ESTIMATE_CHORDINO+"/"+filename+".lab","w")
    for item in chordlist:
        time_end = item["timestamp"]
        estfile.write("%.4f %.4f %s\n" % (time_start,time_end,transform.TransToMirex(l)))
        l = item["label"]
        time_start = time_end
        
    #estfile.write("%s %.4f %.4f" % (l,time_start,wav_duration))
    estfile.close()

#Estimation end; Do evaluation

chordlab_list = find_files(const.PATH_CHORDLAB+"/19_Billboard",ext=["lab","chords"])
estimate_list = find_files(PATH_ESTIMATE_CHORDINO,ext="lab")

assert len(chordlab_list)==len(estimate_list)

scorelist_majmin = np.array([])
scorelist_sevenths = np.array([])
scorelist_inv = np.array([])
scorelist_sevinv = np.array([])
durlist = np.array([])
print("Evaluating...")
for labfile,estfile in zip(chordlab_list,estimate_list):
    (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
    (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
    durlist = np.append(durlist,np.sum(mir_eval.util.intervals_to_durations(est_intervals)))
    scores = mir_eval.chord.evaluate(ref_intervals,ref_labels,est_intervals,est_labels)
    scorelist_majmin = np.append(scorelist_majmin,scores["majmin"])
    scorelist_sevenths = np.append(scorelist_sevenths,scores["sevenths"])
    scorelist_inv = np.append(scorelist_inv,scores["majmin_inv"])
    scorelist_sevinv = np.append(scorelist_sevinv,scores["sevenths_inv"])
    print("%s --- %.03f" % (labfile.split("/")[-1],scores["majmin"]))

      
print("majmin: %.3f" % (np.dot(scorelist_majmin,durlist)/np.sum(durlist)))
print("sevenths: %.3f" % (np.dot(scorelist_sevenths,durlist)/np.sum(durlist)))
print("inv: %.3f" % (np.dot(scorelist_inv,durlist)/np.sum(durlist)))
print("sevenths_inv: %.3f" % (np.dot(scorelist_sevinv,durlist)/np.sum(durlist)))
"""
scorelist_majmin = np.array([])
scorelist_sevenths = np.array([])
scorelist_inv = np.array([])
scorelist_sevinv = np.array([])
durlist = np.array([])

for labfile,estfile in zip(chordlab_list[202:302],estimate_list[202:302]):
    (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
    (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
    durlist = np.append(durlist,np.sum(mir_eval.util.intervals_to_durations(est_intervals)))
    scores = mir_eval.chord.evaluate(ref_intervals,ref_labels,est_intervals,est_labels)
    scorelist_majmin = np.append(scorelist_majmin,scores["majmin"])
    scorelist_sevenths = np.append(scorelist_sevenths,scores["sevenths"])
    scorelist_inv = np.append(scorelist_inv,scores["majmin_inv"])
    scorelist_sevinv = np.append(scorelist_sevinv,scores["sevenths_inv"])
    print("%s --- %.03f" % (labfile.split("/")[-1],scores["majmin"]))

print("RWC:")        
print("majmin: %.3f" % (np.dot(scorelist_majmin,durlist)/np.sum(durlist)))
print("sevenths: %.3f" % (np.dot(scorelist_sevenths,durlist)/np.sum(durlist)))
print("inv: %.3f" % (np.dot(scorelist_inv,durlist)/np.sum(durlist)))
print("sevenths_inv: %.3f" % (np.dot(scorelist_sevinv,durlist)/np.sum(durlist)))


scorelist_majmin = np.array([])
scorelist_sevenths = np.array([])
scorelist_inv = np.array([])
scorelist_sevinv = np.array([])
durlist = np.array([])

for labfile,estfile in zip(chordlab_list[302:],estimate_list[302:]):
    (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
    (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
    durlist = np.append(durlist,np.sum(mir_eval.util.intervals_to_durations(est_intervals)))
    scores = mir_eval.chord.evaluate(ref_intervals,ref_labels,est_intervals,est_labels)
    scorelist_majmin = np.append(scorelist_majmin,scores["majmin"])
    scorelist_sevenths = np.append(scorelist_sevenths,scores["sevenths"])
    scorelist_inv = np.append(scorelist_inv,scores["majmin_inv"])
    scorelist_sevinv = np.append(scorelist_sevinv,scores["sevenths_inv"])
    print("%s --- %.03f" % (labfile.split("/")[-1],scores["majmin"]))

print("USPOP:")        
print("majmin: %.3f" % (np.dot(scorelist_majmin,durlist)/np.sum(durlist)))
print("sevenths: %.3f" % (np.dot(scorelist_sevenths,durlist)/np.sum(durlist)))
print("inv: %.3f" % (np.dot(scorelist_inv,durlist)/np.sum(durlist)))
print("sevenths_inv: %.3f" % (np.dot(scorelist_sevinv,durlist)/np.sum(durlist)))


durations = np.array([])
confmatrix_chord = np.zeros((const.N_CHORDS,const.N_CHORDS))
confmatrix_quality = np.zeros((3,3))
confmatrix_bass = np.zeros((3,3))
for labfile,estfile in zip(chordlab_list,estimate_list):
    (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
    (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
    est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,ref_intervals.min(),ref_intervals.max(),
                                                              mir_eval.chord.NO_CHORD,mir_eval.chord.NO_CHORD)
    (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals,ref_labels,est_intervals,est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    for i in range(len(ref_labels)):
        confmatrix_chord[voc.GetChordIDSign(ref_labels[i]),voc.GetChordIDSign(est_labels[i])] += durations[i]
        ref_b = voc.GetBassFromSign(ref_labels[i])
        est_b = voc.GetBassFromSign(est_labels[i])
        if voc.TriadIsequal(ref_labels[i],est_labels[i]) and (ref_b>=0) and (est_b>=0):
            confmatrix_bass[ref_b,est_b] += durations[i]
        ref_q = voc.GetQualityFromSign(ref_labels[i])
        est_q = voc.GetQualityFromSign(est_labels[i])
        if voc.TriadIsequal(ref_labels[i],est_labels[i]) and (ref_q>=0) and (est_q>=0):
            confmatrix_quality[ref_q,est_q] += durations[i]
            
confmatrix_chord = confmatrix_chord / confmatrix_chord.sum(axis=1)[:,np.newaxis]
confmatrix_quality = confmatrix_quality / confmatrix_quality.sum(axis=1)[:,np.newaxis]
confmatrix_bass = confmatrix_bass / confmatrix_bass.sum(axis=1)[:,np.newaxis]

print(confmatrix_quality)
print(confmatrix_bass)

np.savez("confmatrix_chordino.npz" ,chord=confmatrix_chord,quality=confmatrix_quality,bass=confmatrix_bass)
"""