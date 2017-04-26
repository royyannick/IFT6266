#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date: April 2017
@author: Yannick Roy

Adapted from mcomin 
https://github.com/massimilianocomin/Class-Project-IFT-6266
He did an awesome job for making a small model that can actually produce "something"
and that can be trained in a decent amount of time for preliminary encouraging results.
"""

import sys
import os

if os.environ['LOC'] == 'local':
    libpath = './lib'
    filepath = os.getcwd()
elif os.environ['LOC'] == 'hades':
    libpath = '/home2/ift6ed54/lib'
    filepath = '/home2/ift6ed54/results'
sys.path.append(libpath)

import numpy as np
import pickle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from Img import *
from Layers import *

theano.config.floatX = 'float32'
theano.config.intX = 'int32'

class Model:
    """
    Implementation of a Convolutional Auto-Encoder.
    """
    def __init__(self,bs,n=None):
    
        print('Building computational graph...')

        self.bs = bs
        self.path = filepath
        self.I = Img()
        self.n = n
        self.I.trainlist = self.I.trainlist[0:n]

        x = T.tensor4('inputs') 
        y = T.tensor4('targets') 
        
        loss = self.build(x,y)

        updat = Tool.adam(loss, self.params)

        self.train = theano.function([x,y],loss,updates=updat)
        self.predict = theano.function([x],self.Y)
        
        print('Computational graph built.')

    def build(self,x,y): # Activations ?

        L = {}
        x = x.reshape((x.shape[0],3,64,64))
        y = y.reshape((y.shape[0],3,32,32))
        self.params = []


        Conv = lambda x,y,z,p : ConvLayer(x, nchan = y, nkernels = z, kernelsize = (3,3),
                                    activation = None, pad = 'half', pmode = 'average_inc_pad',
                                    poolsize=(p,p))
        
        TConv = lambda x,shp,args,tied : TConvLayer(x, batch = self.bs, tied=tied, shape = shp, **args)
        
        BN = lambda x,y : BatchNorm(x, y, activation = 'tanh', dims=4) # mode = 'convolution'
        
        Reshape = lambda x, shape : ReshapeLayer(x,shape)

        L[0] = InputLayer(x)

        L[1] = Conv(L[0], 3, 15, 1)
        L[2] = Conv(L[1], 15, 30, 2)

        L[3] = BN(L[2], 30)

        L[4] = Conv(L[3], 30, 35, 1)
        L[5] = Conv(L[4], 35, 50, 2)

        L[6] = BN(L[5], 50)

        L[7] = Conv(L[6], 50, 55, 1)
        L[8] = Conv(L[7], 55, 70, 2)

        L[9] = BN(L[8], 70)

        sizeout = L[8].shape(L[7].shape(L[5].shape(L[4].shape(L[2].shape(L[1].shape(64))))))

        L[10] = DenseLayer(L[9], 70*sizeout**2, 35*sizeout**2, 'elu')
        L[11] = DenseLayer(L[10], 35*sizeout**2, 70*sizeout**2, 'elu')

        sizein = (self.bs,70,sizeout,sizeout)

        rshp = ReshapeLayer(L[11], sizein)

        L[12] = TConv(rshp, (16,16), L[8].arguments, tied=False)
        L[13] = TConv(L[12], (16,16), L[7].arguments, tied=True)

        L[14] = BN(L[13], 50)

        L[15] = TConv(L[14], (32,32), L[5].arguments, tied=False)
        L[16] = TConv(L[15], (32,32), L[4].arguments, tied=True)

        L[17] = BN(L[16], 30)

        modified_args = L[2].arguments.copy()
        modified_args['poolsize'] = (1,1)

        L[18] = TConv(L[17], (32,32), modified_args, tied=False)
        L[19] = TConv(L[18], (32,32), L[1].arguments, tied=True)

        self.Y = L[19].output

        self.params = [x for i in L.keys() for x in L[i].params]

        return Tool.Mse(self.Y, y, dims=4) #mode=4

    def Train(self,epochs=1,save=True):

        for i in range(epochs):
            
            with Tool.Timer() as t :
                
                for j in range(self.n // self.bs):
                    
                    trainx,trainy = self.I.load_batch(self.bs,j,mode='train')
                    out = self.train(trainx,trainy)
                    
            print('Epoch {0} ## Loss : {1:.4} ## Time : {2:.3} s'.format(i+1,float(out),t.interval))

            if ((i+1)%5 == 0 or i+1 == epochs) and save:
                self.Generate('train',1)
                self.Generate('valid',1)
                self.__save__(str(i+1))
            
    def Generate(self,mode,nbatch,save=False):

        base_img = []
        center_pred = []
        
        for j in range(nbatch):
            crop,center = self.I.load_batch(self.bs,j,mode=mode)
            base_img += [crop]
            center_pred += [self.predict(crop)]
        
        pred = np.concatenate(center_pred)
        recon = np.concatenate(base_img)
        recon[:,:,16:48,16:48] += pred
 
        if mode == 'train':
            self.train_recon = recon
        elif mode == 'valid':
            self.valid_recon = recon
           
            
    def __save__(self,epoch):
        
        train_directory = self.path+'/CAE_train/
        
        numpy_params = [self.params[k].get_value() for k in range(len(self.params))]
    
        with open(self.path+'/CAE_params_' + epoch,'wb') as file:
            pickle.dump(numpy_params,file, 2)

        if not os.path.exists(train_directory):
            os.makedirs(train_directory)
            
        for i in range(10):
            self.I.save(self.train_recon[i],train_directory'+epoch+'_'+str(i))
            self.I.save(self.valid_recon[i],train_directory'+epoch+'_'+str(i))


    def __load__(self,epoch):
        
        with open(self.path+'/CAE_params_' + str(epoch),'rb') as file:
            loaded_params = pickle.load(file)
            
        for k in range(len(self.params)):
            self.params[k].set_value(loaded_params[k])
