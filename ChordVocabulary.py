# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:19:35 2015

@author: xming
"""
import mir_eval.chord as evalchord
from scipy.stats import gmean
from librosa.util import normalize
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import const

Quality = {
    'maj':     0,
    'min':     1,
    'aug':     0,
    'dim':     1,
    'sus4':    0,
    'sus2':    0,
    '7':       0,
    'maj7':    0,
    'min7':    1,
    'minmaj7': 1,
    'maj6':    0,
    'min6':    1,
    'dim7':    1,
    'hdim7':   1,
    'maj9':    0,
    'min9':    1,
    '9':       0,
    'b9':      0,
    '#9':      0,
    'min11':   1,
    '11':      0,
    '#11':     0,
    'maj13':   0,
    'min13':   1,
    '13':      0,
    'b13':     0,
    '1':       0,
    '5':       0,
    '' :       0
}

SeventhType = {
    'maj':     0,
    'min':     0,
    'aug':     0,
    'dim':     0,
    'sus4':    0,
    'sus2':    0,
    '7':       1,
    'maj7':    2,
    'min7':    1,
    'minmaj7': 2,
    'maj6':    0,
    'min6':    0,
    'dim7':    1,
    'hdim7':   1,
    'maj9':    2,
    'min9':    1,
    '9':       0,
    'b9':      0,
    '#9':      0,
    'min11':   0,
    '11':      0,
    '#11':     0,
    'maj13':   0,
    'min13':   0,
    '13':      0,
    'b13':     0,
    '1':       0,
    '5':       0,
    '' :       0
}

"""

Quality = {
    'maj':     0,
    'min':     1,
    'aug':     0,
    'dim':     1,
    'sus4':    0,
    'sus2':    0,
    '7':       4,
    'maj7':    2,
    'min7':    3,
    'minmaj7': 1,
    'maj6':    0,
    'min6':    1,
    'dim7':    3,
    'hdim7':   3,
    'maj9':    2,
    'min9':    3,
    '9':       4,
    'b9':      4,
    '#9':      4,
    'min11':   3,
    '11':      4,
    '#11':     4,
    'maj13':   2,
    'min13':   3,
    '13':      4,
    'b13':     4,
    '1':       0,
    '5':       0,
    '' :       0
}
"""
PitchChr = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
Triads = ['C:maj','Db:maj','D:maj','Eb:maj','E:maj','F:maj','Gb:maj','G:maj','Ab:maj','A:maj','Bb:maj','B:maj','C:min','Db:min','D:min','Eb:min','E:min','F:min','Gb:min','G:min','Ab:min','A:min','Bb:min','B:min',"N"]
OutputQualityList = ['maj','min','maj7','min7','7']
QualityMap = {
    "maj" : 0,
    "min" : 0,
    "maj7" : 2,
    "min7" : 1,
    "7" : 1,
    "minmaj7":2    
    }

#QualityList = ['maj','min']
#svm_seventh = joblib.load("svm_seventh.pkl")

    
def GetChordID(tone,ctype):
    if ctype<0 or tone<0:
        return -1
    return ctype*12+tone

def GetChordIDSign(sign):
    sp = evalchord.split(sign)
    pitch = sp[0]
    quality = sp[1]
    if pitch == 'N':
        return const.N_CHORDS-1
    semitone = evalchord.pitch_class_to_semitone(pitch)
    quality_id = Quality[quality]
    return quality_id*12 + semitone

def GetBassFromSign(sign):
    sp = evalchord.split(sign)
    bass = sp[3]
    if bass=="1":
        return 0
    elif bass == "3" or bass=="b3":
        return 1
    elif bass == "5":
        return 2
    return -1
    
def GetQualityFromSign(sign):
    sp = evalchord.split(sign)
    quality = sp[1]
    if quality in QualityMap.keys():
        return QualityMap[quality]
    else:
        return -1
def TriadIsequal(sign1,sign2):
    sp1 = evalchord.split(sign1)
    sp2 = evalchord.split(sign2)
    root1 = sp1[0]
    root2 = sp2[0]
    quality1 = Quality[sp1[1]]
    quality2 = Quality[sp2[1]]
    return (root1==root2) and (quality1==quality2)
    
def ChordSignature(chordid):
    if chordid == const.N_CHORDS-1:
        return 'N'
    chordid = int(chordid)
    return "%s:%s" % (PitchChr[chordid%12],OutputQualityList[chordid//12])

def ChordSignature7thbass(chordid,feature,sevenths=True,inv=True):
    if chordid == const.N_CHORDS-1:
        return "N"
    chroma = normalize(feature[:,12:24],axis=1)
    bass_chroma = normalize(feature[:,:12],axis=1)
    root_note = chordid % 12
    third_note = (root_note + 4 - (chordid // 12))%12
    fifth_note = (root_note + 7) % 12
    seventh_note = (root_note + 10) % 12
    majseventh_note = (root_note + 11) % 12
    mean_root = np.mean(bass_chroma[:,root_note])
    mean_3rd = np.mean(bass_chroma[:,third_note])
    mean_5th = np.mean(bass_chroma[:,fifth_note])
    mean_7th = np.mean(chroma[:,seventh_note])
    mean_maj7th = np.mean(chroma[:,majseventh_note])
    root = PitchChr[root_note]
    quality = OutputQualityList[chordid//12]
    bass = ""
    #determine seventh
    if sevenths:
        if (mean_7th>0.5) or (mean_maj7th>0.5):
            if mean_7th >= mean_maj7th:
                if quality == "min":
                    quality = "min7"
                else:
                    quality = "7"
            else:
                if quality == "maj":
                    quality = "maj7"
                else:
                    quality = "minmaj7"
    #determine bass
    if inv:
        if (mean_3rd>0.6 and mean_3rd>mean_root) or (mean_5th>0.6 and mean_5th>mean_root):
            if mean_3rd>mean_5th:
                if (quality == "min") or (quality =="min7"):
                    bass="b3"
                else:
                    bass = "3"
            else:
                bass = "5"
    sign = "%s:%s" % (root,quality)
    if bass != "":
        sign += ("/"+bass)
    return sign

def GetSeventhType(sign):
    sp = evalchord.split(sign)
    return SeventhType[sp[1]]
    
def TransposeToC(vec,sign):
    sp = evalchord.split(sign)
    root = sp[0]
    semitone = evalchord.pitch_class_to_semitone(root)
    return np.roll(vec,-semitone)

def ChordSignatureSVM(chordid,feature):
    if chordid == const.N_CHORDS-1:
        return "N"
    root_note = chordid % 12
    root = PitchChr[root_note]
    quality = OutputQualityList[chordid//12]
    feat_transposed = np.roll(np.mean(feature,axis=0),-root_note).reshape(1,-1)
    sev_type = svm_seventh.predict(feat_transposed)[0]
    if sev_type==1:
        if quality=="min":
            quality="min7"
        else:
            quality="7"
    elif sev_type==2:
        if quality=="min":
            quality="minmaj7"
        else:
            quality="maj7"
    
    sign = "%s:%s" % (root,quality)
    return sign

def LoadChromaTarget(labfile):
    f = open(labfile)
    line = f.readline()
    labarr = np.zeros((600*const.SR//const.H,12),dtype="int32")
    bassarr = np.zeros(labarr.shape,dtype="int32")
    while line != "" and line.isspace()==False:
        items = line.split()
        st = int(round(float(items[0])*const.SR/const.H))
        ed = int(round(float(items[1])*const.SR/const.H))
        lab = items[2]
        root,bitmap,bass = evalchord.encode(lab)
        chroma = evalchord.rotate_bitmap_to_root(bitmap,root)
        bassnumber = (root+bass)%12
        labarr[st:ed,:] = chroma
        bassarr[st:ed,bassnumber] = 1
        line = f.readline()
    return np.concatenate((bassarr[:ed],labarr[:ed]),axis=1).astype(np.float32)