#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:59:08 2017

@author: wuyiming
"""

from chainer import Chain,serializers,ChainList,Variable,cuda
import chainer.links as L
import chainer.functions as F
import const
import utils

import numpy as np
cp = cuda.cupy


class ResBlock(Chain):
    def __init__(self,channels,ksize,stride,pad):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(channels,channels,ksize,stride,pad)
            self.norm1 = L.BatchRenormalization(channels)
            self.conv2 = L.Convolution2D(channels,channels,ksize,stride,pad)
            self.norm2 = L.BatchRenormalization(channels)
            self.conv3 = L.Convolution2D(channels,channels,ksize,stride,pad)
            self.norm3 = L.BatchRenormalization(channels)
    def __call__(self,x):
        h = F.leaky_relu(self.norm1(self.conv1(x)))
        h = F.leaky_relu(self.norm2(self.conv2(h)))
        h = F.leaky_relu(self.norm3(self.conv3(h))+x)
        return h


class FullCNNFeatExtractor(Chain):
    def __init__(self):
        super(FullCNNFeatExtractor,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(const.CQT_H,64,(3,11),(1,1),(1,5))
            self.res1 = ResBlock(64,(5,5),1,(2,2))
            self.res2 = ResBlock(64,(5,5),1,(2,2))
            self.conv2 = L.Convolution2D(64,36,(3,const.OCTAVE*12),1,(1,0))  #(batch,36,T,1)
            #self.lnorm = L.LayerNormalization(64)
            #self.out = L.Linear(64,36)    
            
    def __call__(self,x):
        h1 = F.leaky_relu(self.conv1(x))
        h1 = self.res1(h1)
        h1 = self.res2(h1)
        h1 = self.conv2(h1)
        #assert(h1.shape[3]==1)
        return F.transpose(h1[:,:,:,0],axes=(0,2,1))
    
    def GetFeature(self,x):
        o = F.sigmoid(self(x[None,:,:,:]))
        return o[0,:,:]
        #return Variable(x[0,:,:])        
    def GetFeatureSeq(self,x):
        o = self(x[None,:,:,:])
        o.unchain_backward()
        return F.separate(o,axis=1)
    def save(self,fname=const.DEFAULT_CONVNETFILE):
        serializers.save_npz(fname,self)
    def load(self,fname=const.DEFAULT_CONVNETFILE):
        serializers.load_npz(fname,self)
        
class ConvnetPredictor(Chain):
    def __init__(self,predictor):
        super(ConvnetPredictor,self).__init__()
        with self.init_scope():
            self.predictor = predictor
            
    def __call__(self,x,t):
        y_list = self.predictor(x)
        #t_list = F.separate(t,axis=1)
        self.loss = F.sigmoid_cross_entropy(y_list,t)
        return self.loss

class NoOperation(Chain):
    def __call__(self,x):
        return x[0,:,:]
    
    def GetFeature(self,x):
        return Variable(x[0,:,:])
    def save(self,fname="dnn1000.model"):
        pass
    def load(self,fname="dnn1000.model"):
        pass

class FeatureDNN(ChainList):
    def __init__(self):
        super(FeatureDNN,self).__init__()
        self.add_link(L.Linear(144*7,const.HIDDEN_UNITS[0]))
        for i in range(len(const.HIDDEN_UNITS)-1):
            self.add_link(L.Linear(const.HIDDEN_UNITS[i],const.HIDDEN_UNITS[i+1]))
        self.add_link(L.Linear(const.HIDDEN_UNITS[-1],24))
    
    def __call__(self,x):
        li = self.children()
        h = F.relu(next(li)(x))
        for i in range(self.__len__()-2):
            h = F.relu(next(li)(h))+h
        y = next(li)(h)
        
        return y
    
    def GetFeature(self,x):
        y = self(x)
        #return F.concat((F.softmax(y[:,:12]),F.sigmoid(y[:,12:24]),F.softmax(y[:,24:])),axis=1)
        return F.sigmoid(y)
    def save(self,fname="dnn1000.model"):
        serializers.save_npz(fname,self)
    def load(self,fname="dnn1000.model"):
        serializers.load_npz(fname,self) 

class DNNModel(Chain):
    def __init__(self,predictor):
        super(DNNModel,self).__init__()
        with self.init_scope():
            self.predictor = predictor
    
    def __call__(self,x,t):
        y = self.predictor(x)
        #t_bass = np.argmax(t[:,:12],axis=1)
        #t_top = np.argmax(t[:,24:],axis=1)
        #self.loss = F.softmax_cross_entropy(y[:,:12],t_bass)+F.sigmoid_cross_entropy(y[:,12:24],t[:,12:24])+F.softmax_cross_entropy(y[:,24:],t_top)
        self.loss = F.sigmoid_cross_entropy(y,t)
        #self.loss = F.mean_squared_error(F.sigmoid(y),t)
        return self.loss



class ConvnetChordClassifier(Chain):
    def __init__(self):
        super(ConvnetChordClassifier,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3,16,(5,5),1,(2,0))
            self.conv2 = L.Convolution2D(16,64,(5,5),1,(2,0))
            self.conv3 = L.Convolution2D(64,64,(3,4),1,(1,0))
            self.out = L.Linear(64,const.N_CHORDS)
    
    def __call__(self,x):
        x = F.stack((x[:,:,:12],x[:,:,12:24],x[:,:,24:]),axis=1)
        hd = F.relu(self.conv1(x))
        hd = F.relu(F.dropout(self.conv2(hd)))
        out_list = [self.out(o[:,:,0]) for o in F.separate(F.relu(F.dropout(self.conv3(hd))),axis=2)]
        return out_list
    def save(self,fname="convclassifier.model"):
        serializers.save_npz(fname,self)
    def load(self,fname="convclassifier.model"):
        serializers.load_npz(fname,self)        
 

class NSBLSTM(Chain):
    def __init__(self):
        super(NSBLSTM,self).__init__()
        with self.init_scope():
            self.blstm = L.NStepBiLSTM(1,36,128,0.0)
            self.out = L.Linear(256,const.N_CHORDS)
            #self.norm = L.LayerNormalization(256)
    
    def __call__(self,xs,ts):
        _, _, ys = self.blstm(None,None,xs)
        ys = F.concat(ys,axis=0)
        ts = F.concat(ts,axis=0)
        self.loss = F.sum(F.softmax_cross_entropy(self.out(F.dropout(ys)),ts,reduce="no"))
        
        #ys_t = F.transpose_sequence(ys)
        #ts_t = F.transpose_sequence(ts)
        #self.loss = sum([F.softmax_cross_entropy(self.out(F.dropout(y)),t) for y,t in zip(ys_t,ts_t)])
        
        return self.loss
    
    def GetFeat(self,xs):
        _,_,ys = self.blstm(None,None,xs)
        #o_list = [self.out(y) for y in ys]
        return [self.out(y) for y in ys]
        #return ys
    def save(self,fname="blstm.model"):
        serializers.save_npz(fname,self)
        
    def load(self,fname="blstm.model"):
        serializers.load_npz(fname,self)
        #self.blstm.dropout = 0.0

class NBLSTMCRF(Chain):
    def __init__(self):
        super(NBLSTMCRF,self).__init__()
        with self.init_scope():
            self.blstm = NSBLSTM()
            self.crf = L.CRF1d(const.N_CHORDS)
            #self.li = L.Linear(256,const.N_CHORDS)
        #self.blstm.load()
        #self.li.copyparams(self.blstm.out)
        #self.blstm.blstm.dropout = 0
    def __call__(self,xs,ts):
        ys = self.blstm.GetFeat(xs)
        for y in ys:
            y.unchain_backward()
        #ys_t = F.transpose_sequence([self.li(y) for y in ys])
        ys_t = F.transpose_sequence(ys)
        ts_t = F.transpose_sequence(ts)
        self.loss = self.crf(ys_t,ts_t)
        return self.loss
    def argmax(self,x):
        ys = self.blstm.GetFeat([Variable(x)])
        #y_t = F.transpose_sequence([self.li(y) for y in ys])
        y_t = F.transpose_sequence(ys)
        _,path = self.crf.argmax(y_t)
        return utils.force_numpy(path).flatten().astype(np.int32)
    def argmax_rnn(self,x):
        y = self.blstm.GetFeat([Variable(x)])[0]
        path = F.argmax(y,axis=1).data
        return cp.asnumpy(path).astype(np.int32)
    def save(self,fname="nblstm_crf.model"):
        serializers.save_npz(fname,self)
        
    def load(self,fname="nblstm_crf.model"):
        serializers.load_npz(fname,self)
        #self.blstm.dropout = 0.0


class ConvnetDecoder(Chain):
    def __init__(self):
        super(ConvnetDecoder,self).__init__()
        with self.init_scope():
            self.convnet = ConvnetChordClassifier()
    def __call__(self,x,t_list):
        self.loss = sum([F.softmax_cross_entropy(y,t) for y,t in zip(self.convnet(x),t_list)])
        return self.loss
    def save(self,fname="convclassifier.model"):
        serializers.save_npz(fname,self.convnet)
    def load(self,fname="convclassifier.model"):
        serializers.load_npz(fname,self.convnet)    


        
class RNNCRFModel(Chain):
    def __init__(self,rnn):
        super(RNNCRFModel,self).__init__()
        with self.init_scope():
            self.rnn=rnn
            self.crf=L.CRF1d(const.N_CHORDS)
    def __call__(self,x_list,t_list):
        self.rnn.reset_state()
        y_list = self.rnn(x_list)
        for y in y_list:
            y.unchain_backward()
        self.loss = self.crf(y_list,t_list)
        return self.loss
    def argmax(self,x_list):
        self.rnn.reset_state()
        y_list = self.rnn(x_list)
        _,path = self.crf.argmax(y_list)
        return cp.asnumpy(path).flatten().astype(np.int32)
    
    def argmax_rnn(self,x_list):
        self.rnn.reset_state()
        y_list = self.rnn(x_list)
        path = [F.argmax(y).data for y in y_list]
        return np.array(path,dtype="int32").flatten()
    def save(self,fname="crfrnn.model"):
        serializers.save_npz(fname,self)
    def load(self,fname="crfrnn.model"):
        serializers.load_npz(fname,self)

