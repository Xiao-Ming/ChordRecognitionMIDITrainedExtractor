#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 22:00:14 2017

@author: wuyiming
"""

import networks
from librosa.util import find_files,normalize
import chainer
from chainer import optimizers,config,optimizer,Variable
import numpy as np
import cupy as cp
import const
import utils
from chromatemplate import GetMultiFeatureFromPianoroll,GetTargetsFromPianoroll,GetConvnetTargetFromPianoroll
import ChordVocabulary as voc


def TrainConvnetExtractor(trainidx,epoch=20,saveas="convnet.model"):
    cqtfilelist = np.array(find_files(const.PATH_MIDIHCQT,ext="npz"))[trainidx]
    #midifilelist = find_files(const.PATH_MIDI,ext="mid")[:filecnt]
    config.train = True
    config.enable_backprop = True
    convnet = networks.FullCNNFeatExtractor()
    model =networks.ConvnetPredictor(convnet)
    model.to_gpu(0)
    opt = optimizers.AdaDelta()
    opt.setup(model)
    print("train set length: %d" % trainidx.size)
    print("start epochs...")
    S = []
    T = []
    
    for cqtfile in cqtfilelist:
        dat = np.load(cqtfile)
        spec = utils.PreprocessSpec(dat["spec"])[:const.CQT_H,:,:]
        targ = GetConvnetTargetFromPianoroll(dat["target"]).astype(np.int32)
        assert(spec.shape[1]==targ.shape[0])
        S.append(spec)
        T.append(targ)
    S = np.concatenate(S,axis=1)
    T = np.concatenate(T,axis=0)
    
    for ep in range(epoch):
        sum_loss = 0
        
        assert(S.shape[1]==T.shape[0])
        randidx = np.random.randint(0,S.shape[1]-const.CONV_TRAIN_SEQLEN-1,S.shape[1]//const.CONV_TRAIN_SEQLEN*4)
        for i in range(0,randidx.size-const.CONV_TRAIN_BATCH,const.CONV_TRAIN_BATCH):
            x_batch = np.stack([S[:,randidx[j]:randidx[j]+const.CONV_TRAIN_SEQLEN,:] for j in range(i,i+const.CONV_TRAIN_BATCH)])
            t_batch = np.stack([T[randidx[j]:randidx[j]+const.CONV_TRAIN_SEQLEN,:] for j in range(i,i+const.CONV_TRAIN_BATCH)])
            x_in = cp.asarray(x_batch)
            t_in = cp.asarray(t_batch)
            model.cleargrads()
            loss = model(x_in,t_in)
            loss.backward()
            opt.update()
            sum_loss += loss.data
                
        convnet.save(saveas)
        print("epoch: %d/%d  loss:%.04f" % (ep+1,epoch,sum_loss/const.CONV_TRAIN_BATCH))
        
    convnet.save(saveas)

def TrainConvnetExtractorDeepChroma(trainidx,epoch=20,saveas="convnet.model"):
    cqtfilelist = np.array(find_files(const.PATH_HCQT,ext="npy"))[trainidx]
    labfilelist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))[trainidx]
    #midifilelist = find_files(const.PATH_MIDI,ext="mid")[:filecnt]
    config.train = True
    config.enable_backprop = True
    convnet = networks.ConvnetFeatExtractor()
    model =networks.ConvnetPredictor(convnet)
    model.to_gpu(0)
    opt = optimizers.MomentumSGD()
    opt.setup(model)
    print("DeepChroma Convnet Training...")
    print("start epochs...")
    S = []
    T = []
    
    for cqtfile,labfile in zip(cqtfilelist,labfilelist):
        cqt = np.load(cqtfile)
        spec = utils.PreprocessSpec(cqt[:const.CQT_H,:,:])
        targ = voc.LoadChromaTarget(labfile)
        minlen = min([cqt.shape[1],targ.shape[0]])
        S.append(spec[:,:minlen,:])
        T.append(targ[:minlen,:])
    S = np.concatenate(S,axis=1)
    T = np.concatenate(T,axis=0)
    assert(S.shape[1]==T.shape[0])
    
    for ep in range(epoch):
        sum_loss = 0
        
        randidx = np.random.randint(0,S.shape[1]-const.CONV_TRAIN_SEQLEN-1,S.shape[1]//const.CONV_TRAIN_SEQLEN*4)
        for i in range(0,randidx.size-const.CONV_TRAIN_BATCH,const.CONV_TRAIN_BATCH):
            x_batch = np.stack([S[:,randidx[j]:randidx[j]+const.CONV_TRAIN_SEQLEN,:] for j in range(i,i+const.CONV_TRAIN_BATCH)])
            t_batch = np.stack([T[randidx[j]:randidx[j]+const.CONV_TRAIN_SEQLEN,:] for j in range(i,i+const.CONV_TRAIN_BATCH)])
            x_in = cp.asarray(x_batch)
            t_in = cp.asarray(t_batch)
            model.cleargrads()
            loss = model(x_in,t_in)
            loss.backward()
            opt.update()
            sum_loss += loss.data
                
        convnet.save(saveas)
        print("epoch: %d/%d  loss:%.04f" % (ep+1,epoch,sum_loss/const.CONV_TRAIN_BATCH))
        
    convnet.save(saveas)


def TrainDNNExtractor(trainidx,epoch=20,saveas="dnn.model"):
    cqtfilelist = np.array(find_files(const.PATH_MIDIHCQT,ext="npz"))[trainidx]
    #midifilelist = find_files(const.PATH_MIDI,ext="mid")[:filecnt]
    filecnt = cqtfilelist.size
    chainer.config.train=True
    chainer.config.enable_backprop = True
    dnn = networks.FeatureDNN()
    model =networks.DNNModel(dnn)
    model.to_gpu(0)
    opt = optimizers.MomentumSGD()
    opt.setup(model)
    spl = np.arange(0,filecnt,2000)
    if spl[-1] < filecnt:
        spl = np.append(spl,filecnt)
    
    print("split count:%d" % (spl.size-1))
    print(spl)
    print("start epochs...")
    S = []
    T = []
    
    for cqtfile in cqtfilelist:
        dat = np.load(cqtfile)
        spec = utils.PreprocessSpec(dat["spec"][:,:])
        targ = GetConvnetTargetFromPianoroll(dat["target"]).astype(np.int32)
        assert(spec.shape[0]==targ.shape[0])
        S.append(spec)
        T.append(targ)
    S = np.concatenate(S,axis=0)
    T = np.concatenate(T,axis=0)
    
    for ep in range(epoch):
        sum_loss = 0
        
        assert(S.shape[0]==T.shape[0])
        randidx = np.random.permutation(S.shape[0])
        for i in range(0,randidx.size,const.CONV_TRAIN_BATCH):
            x_batch = S[randidx[i:i+const.CONV_TRAIN_BATCH],:]
            t_batch = T[randidx[i:i+const.CONV_TRAIN_BATCH],:]
            x_in = cp.asarray(x_batch)
            t_in = cp.asarray(t_batch)
            model.cleargrads()
            loss = model(x_in,t_in)
            loss.backward()
            opt.update()
            sum_loss += loss.data * const.CONV_TRAIN_BATCH
                
        dnn.save(saveas)
        print("epoch: %d/%d  loss:%.04f" % (ep+1,epoch,sum_loss/const.CONV_TRAIN_BATCH))



def TrainTranscribeDNNChord(idx,epoch=20,saveas="dnn_deepchroma.model"):
    cqtfilelist = np.array(find_files(const.PATH_HCQT,ext="npy"))[idx]
    chordlablist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))[idx]
    
    featurelist = []
    targetlist = []    
    chainer.config.train=True
    chainer.config.enable_backprop = True
    
    for cqtfile,labfile in zip(cqtfilelist,chordlablist):
        cqt = np.load(cqtfile)[0,:,:]
        chroma = voc.LoadChromaTarget(labfile)
        min_sz = min([cqt.shape[0],chroma.shape[0]])
        cqt = utils.Embed(utils.PreprocessSpec(cqt[:min_sz]),size=7)
        chroma = chroma[:min_sz]
        featurelist.append(cqt)
        targetlist.append(chroma.astype(np.int32))
    featurelist = np.concatenate(featurelist)
    targetlist = np.concatenate(targetlist)
    itemcnt = targetlist.shape[0]
    print("DNN Training begin...")
    dnn = networks.FeatureDNN()
    dnn.train=True
    model = networks.DNNModel(predictor=dnn)
    model.to_gpu()
    opt = optimizers.AdaDelta()
    opt.setup(model)
    for ep in range(epoch):
        randidx = np.random.permutation(itemcnt)
        sumloss = 0.0     
        for i in range(0,itemcnt,const.DNN_TRAIN_BATCH):
            X = cp.asarray(featurelist[randidx[i:i+const.DNN_TRAIN_BATCH]])
            T = cp.asarray(targetlist[randidx[i:i+const.DNN_TRAIN_BATCH]])
            opt.update(model,X,T)
            sumloss += model.loss.data * const.DNN_TRAIN_BATCH
        print("epoch %d/%d  loss=%.3f" % (ep+1,epoch,sumloss/itemcnt))
    
    dnn.save(saveas)    



def shift_data(feat,lab,shift):
    newlab = np.array(lab)
    for i in range(lab.size):
        if lab[i]<12:
            newlab[i] = (lab[i] + shift) % 12
        elif lab[i]<24:
            newlab[i] = (lab[i] + shift) % 12 + 12
        
    newfeat = np.concatenate((np.roll(feat[:,:12],shift,axis=1),np.roll(feat[:,12:24],shift,axis=1),np.roll(feat[:,24:],shift,axis=1)),axis=1)
    #newfeat = np.concatenate((np.roll(feat[:,:12],shift,axis=1),np.roll(feat[:,12:24],shift,axis=1)),axis=1)
    return newfeat,newlab


def TrainNStepRNN(idx,epoch=100,featmodel=const.DEFAULT_CONVNETFILE,augment=0,savefile="blstm.model"):
    cqtfilelist = np.array(find_files(const.PATH_HCQT,ext="npy"))
    chordlablist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))
    if idx is not None:
        cqtfilelist = cqtfilelist[idx]
        chordlablist = chordlablist[idx]
    chainer.config.train=False
    chainer.config.enable_backprop = False
    #dnn = networks.TripleDNNExtractor()
    #dnn = networks.FeatureDNN()
    dnn = networks.FullCNNFeatExtractor()
    #dnn = networks.NoOperation()
    #dnn = networks.ConvnetFeatExtractor()
    dnn.load(featmodel)
    dnn.to_gpu(0)
    
    rnn = networks.NSBLSTM()
    rnn.to_gpu(0)
    opt = optimizers.RMSprop()
    opt.setup(rnn)
    #opt.add_hook(optimizer.WeightDecay(0.01))
    X = []
    T = []
    for cqtfile,labfile in zip(cqtfilelist,chordlablist):
        cqt= utils.Embed(utils.PreprocessSpec(np.load(cqtfile)[:,:,:]),1)
        feature = cp.asnumpy(dnn.GetFeature(cp.asarray(cqt)).data)
        lab = utils.LoadLabelArr(labfile)
        min_sz = min([feature.shape[0],lab.shape[0]])
        X.append(feature[:min_sz,:])
        T.append(lab[:min_sz])
    sizes = np.array([x.shape[0] for x in X],dtype="int32")
    print("start epoch:")
    chainer.config.train=True
    chainer.config.enable_backprop = True
    last_loss = np.inf
    for ep in range(epoch):
        sum_loss = 0.0
        rand_songid = np.random.randint(len(X),size=np.sum(sizes) // const.DECODER_TRAIN_SEQLEN * 8)
        for i in range(0,rand_songid.size,const.DECODER_TRAIN_BATCH):
            xbatch = []
            tbatch = []
            for songid in rand_songid[i:i+const.DECODER_TRAIN_BATCH]:
                seq_len = sizes[songid]
                idx = np.random.randint(seq_len - const.DECODER_TRAIN_SEQLEN - 1)
                x_snip = X[songid][idx:idx+const.DECODER_TRAIN_SEQLEN,:]
                t_snip = T[songid][idx:idx+const.DECODER_TRAIN_SEQLEN]
                if augment>0:
                    shift = np.random.randint(augment)
                    x_snip,t_snip = shift_data(x_snip,t_snip,shift)
                xbatch.append(Variable(cp.asarray(x_snip)))
                tbatch.append(Variable(cp.asarray(t_snip)))
            rnn.cleargrads()
            opt.update(rnn,xbatch,tbatch)
            sum_loss += rnn.loss.data
            
        print("epoch %d/%d loss=%.3f" % (ep+1,epoch,sum_loss/12800.0))  
        if(sum_loss < last_loss):
            rnn.save(savefile)
            last_loss = sum_loss
        else:
            break
    #rnn.save()


def TrainNStepCRF(idx,epoch=20,augment=0,featmodel=const.DEFAULT_CONVNETFILE,path_blstm="blstm.model",savefile="nblstm_crf.model"):
    cqtfilelist = np.array(find_files(const.PATH_HCQT,ext="npy"))
    chordlablist = np.array(find_files(const.PATH_CHORDLAB,ext=["lab","chords"]))
    if idx is not None:
        cqtfilelist = cqtfilelist[idx]
        chordlablist = chordlablist[idx]
    chainer.config.train=False
    chainer.config.enable_backprop = False
    #dnn = networks.TripleDNNExtractor()
    #dnn = networks.FeatureDNN()
    dnn = networks.FullCNNFeatExtractor()
    #dnn = networks.NoOperation()
    #dnn = networks.ConvnetFeatExtractor()
    dnn.load(featmodel)
    dnn.to_gpu(0)
    
    rnn = networks.NBLSTMCRF()
    rnn.blstm.load(path_blstm)
    rnn.to_gpu(0)
    opt = optimizers.MomentumSGD()
    opt.setup(rnn)
    #opt.add_hook(optimizer.WeightDecay(0.001))
    X = []
    T = []
    for cqtfile,labfile in zip(cqtfilelist,chordlablist):
        cqt= utils.Embed(utils.PreprocessSpec(np.load(cqtfile)[:,:,:]),1)
        feature = cp.asnumpy(dnn.GetFeature(cp.asarray(cqt)).data)
        lab = utils.LoadLabelArr(labfile)
        min_sz = min([feature.shape[0],lab.shape[0]])
        X.append(feature[:min_sz,:])
        T.append(lab[:min_sz])
    sizes = np.array([x.shape[0] for x in X],dtype="int32")
    print("start epoch:")
    chainer.config.train = False
    chainer.config.enable_backprop = True
    last_loss = np.inf
    for ep in range(epoch):
        sum_loss = 0.0
        rand_songid = np.random.randint(len(X),size=np.sum(sizes) // const.DECODER_TRAIN_SEQLEN * 8)
        for i in range(0,rand_songid.size,const.DECODER_TRAIN_BATCH):
            xbatch = []
            tbatch = []
            for songid in rand_songid[i:i+const.DECODER_TRAIN_BATCH]:
                seq_len = sizes[songid]
                idx = np.random.randint(seq_len - const.DECODER_TRAIN_SEQLEN - 1)
                x_snip = X[songid][idx:idx+const.DECODER_TRAIN_SEQLEN,:]
                t_snip = T[songid][idx:idx+const.DECODER_TRAIN_SEQLEN]
                if augment>0:
                    shift = np.random.randint(augment)
                    x_snip,t_snip = shift_data(x_snip,t_snip,shift)
                xbatch.append(Variable(cp.asarray(x_snip)))
                tbatch.append(Variable(cp.asarray(t_snip)))
            rnn.cleargrads()
            opt.update(rnn,xbatch,tbatch)
            sum_loss += rnn.loss.data
            
        
        print("epoch %d/%d loss=%.3f" % (ep+1,epoch,sum_loss/12800.0))
        rnn.save(savefile)
