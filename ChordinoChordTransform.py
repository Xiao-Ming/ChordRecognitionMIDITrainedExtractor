# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:45:38 2016

@author: wuyiming
"""
import mir_eval.chord as evalchord

ch_bass = ["0","b2","2","b3","3","4","b5","5","b6","6","b7","7"]

def calcbass(root,bass):
    p_root = evalchord.pitch_class_to_semitone(root)
    p_bass = evalchord.pitch_class_to_semitone(bass)
    if p_bass<p_root:
        p_bass+=12
    
    return ch_bass[p_bass-p_root]

def TransToMirex(label):
    if label=="N":
        return "N"
    items = label.split("/")
    label_nobass = items[0]
    bass = ""
    root = ""
    quality = ""
    if len(items)>1:
        bass = items[-1]
    if len(label_nobass) == 1:
        return label_nobass
    if (label_nobass[1]=="#") or (label_nobass[1]=="b"):
        root=label_nobass[:2]
        if len(label_nobass)>2:
            quality=label_nobass[2:]
    else:
        root=label_nobass[0]
        quality=label_nobass[1:]
    
    if quality=="m7b5":
        quality="dim"
    elif quality=="m":
        quality="min"
    elif quality=="m7":
        quality="min7"
    elif quality=="6":
        quality="maj6"
    elif quality=="m6":
        quality="min6"
    
    ret =  ("%s:%s" % (root,quality))
    if bass != "":
        ret += ("/%s" % calcbass(root,bass))
    return ret
    