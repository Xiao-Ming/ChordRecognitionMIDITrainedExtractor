#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 02:47:22 2017

@author: wuyiming
"""

import mir_eval
import numpy as np
from librosa.util import find_files
import const
import ChordVocabulary as voc
import utils
import networks
import chainer

cp = chainer.cuda.cupy

def EstimateChord(idx,dnnmodel,todir=False):
    #dnn = networks.FeatureDNN()
    #dnn = networks.ConvnetFeatExtractor()
    dnn = networks.FullCNNFeatExtractor()
    #dnn = networks.NoOperation()
    dnn.load(dnnmodel)
    dnn.to_gpu(0)
    decoder = networks.NBLSTMCRF()
    decoder.load()
    decoder.to_gpu(0)
    cqtfilelist = np.array(find_files(const.PATH_HCQT,ext="npy"))[idx]
    i = 0
    chainer.config.train=False
    chainer.config.enable_backprop = False
    for cqtfile in cqtfilelist:
        cqt= utils.Embed(utils.PreprocessSpec(np.load(cqtfile)[:,:,:]),1)
        chroma = dnn.GetFeature(cp.asarray(cqt)).data
        path = decoder.argmax(chroma)
        feat = cp.asnumpy(chroma)
        if todir:
            fname = cqtfile.split("/")[-1] + ".lab"
            alb = cqtfile.split("/")[-2]
            utils.SaveEstimatedLabelsFramewise(path,const.PATH_ESTIMATE_CROSS+alb+"/"+fname,feat)
        else:
            utils.SaveEstimatedLabelsFramewise(path,const.PATH_ESTIMATE+"%03d.lab" % i,feat)
        i+=1


def EvaluateChord(idx,verbose=True,sonify=False,cross=False):
    lablist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    est_lablist = np.array(find_files(const.PATH_ESTIMATE_CROSS,ext="lab"))[idx] if cross else find_files(const.PATH_ESTIMATE,ext="lab")
    scorelist_majmin = np.array([])
    scorelist_sevenths = np.array([])
    scorelist_majmininv = np.array([])
    scorelist_seventhinv = np.array([])
    durations = np.array([])
    confmatrix = np.zeros((const.N_CHORDS,const.N_CHORDS))
    song_durations = np.array([])
    for labfile,estfile in zip(lablist,est_lablist):
        (ref_intervals,ref_labels) = mir_eval.io.load_labeled_intervals(labfile)
        (est_intervals,est_labels) = mir_eval.io.load_labeled_intervals(estfile)
        est_intervals,est_labels = mir_eval.util.adjust_intervals(est_intervals,est_labels,ref_intervals.min(),ref_intervals.max(),
                                                                  mir_eval.chord.NO_CHORD,mir_eval.chord.NO_CHORD)
        (intervals,ref_labels,est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals,ref_labels,est_intervals,est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        
        comparisons_sevenths = mir_eval.chord.sevenths(ref_labels,est_labels)
        comparisons_majmininv = mir_eval.chord.majmin_inv(ref_labels,est_labels)
        comparisons_seventhinv = mir_eval.chord.sevenths_inv(ref_labels,est_labels)
        comparisons_majmin = mir_eval.chord.majmin(ref_labels,est_labels)
        
        score_majmin = mir_eval.chord.weighted_accuracy(comparisons_majmin,durations)
        scorelist_majmin = np.append(scorelist_majmin,score_majmin)
        score_sevenths = mir_eval.chord.weighted_accuracy(comparisons_sevenths,durations)
        scorelist_sevenths = np.append(scorelist_sevenths,score_sevenths)
        score_majmininv = mir_eval.chord.weighted_accuracy(comparisons_majmininv,durations)
        scorelist_majmininv = np.append(scorelist_majmininv,score_majmininv)
        score_seventhinv = mir_eval.chord.weighted_accuracy(comparisons_seventhinv,durations)
        scorelist_seventhinv = np.append(scorelist_seventhinv,score_seventhinv)        
        if verbose:
            print("%s --- %.3f" % (labfile.split('/')[-1],score_majmin))
        
        for i in range(len(ref_labels)):
            confmatrix[voc.GetChordIDSign(ref_labels[i]),voc.GetChordIDSign(est_labels[i])] += durations[i]
        song_durations = np.append(song_durations,np.sum(durations))
    return scorelist_majmin,scorelist_sevenths,scorelist_majmininv,scorelist_seventhinv,confmatrix,song_durations
    
def ConfMatrix(idx,cross=False):
    lablist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    est_lablist = np.array(find_files(const.PATH_ESTIMATE_CROSS,ext="lab"))[idx] if cross else find_files(const.PATH_ESTIMATE,ext="lab")
    durations = np.array([])
    confmatrix_chord = np.zeros((const.N_CHORDS,const.N_CHORDS))
    confmatrix_quality = np.zeros((3,3))
    confmatrix_bass = np.zeros((3,3))
    for labfile,estfile in zip(lablist,est_lablist):
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
    return confmatrix_chord,confmatrix_quality,confmatrix_bass                