#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:01:33 2017

@author: mcomin
"""
import time
import os
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import lasagne.updates as updt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.nnet.neighbours as neigh

theano.config.floatX = 'float32'
theano.config.intX = 'int32'

class Tool:
    """
    Class containing miscellaneous utilities :
    ------------------------------------------
    
        Functions : Activation functions and Optimizers from lasage/theano
        Methods : Upsampling methods, seeds, ...
        Classes : Timers, ...
    """

########################## Functions #############################

    tanh = T.tanh
    sigmoid = T.nnet.sigmoid
    softmax = T.nnet.softmax
    relu = T.nnet.relu
    elu = lambda x :T.switch(x > 0, x, T.exp(x) - 1)
    relux = lambda x: T.switch(x > 0, x, .02*x)
    softplus = T.nnet.softplus
    
    fct = {None:lambda x,**args: x,
                    'sigmoid':T.nnet.sigmoid,
                    'tanh':T.tanh,
                    'softmax':T.nnet.softmax,
                    'relu':T.nnet.relu,
                    'relux':relux,
                    'elu' : elu,
                    'softplus':T.nnet.softplus}
    
    conv = T.nnet.conv2d
    pool = T.signal.pool.pool_2d
    upsamp = T.nnet.abstract_conv.bilinear_upsampling
    deconv = T.nnet.abstract_conv.conv2d_grad_wrt_inputs
    batchnorm = T.nnet.bn.batch_normalization
    
    sgd = updt.sgd
    momentum = updt.momentum
    nesterov = updt.nesterov_momentum
    adagrad = updt.adagrad
    rmsprop = updt.rmsprop
    adadelta = updt.adadelta
    adam = updt.adam
    apply_momentum = updt.apply_momentum
    apply_nesterov = updt.apply_nesterov_momentum
    
    Cce = lambda p,y : T.mean(T.nnet.categorical_crossentropy(p,y))
    Bce = lambda p,y : T.mean(T.nnet.binary_crossentropy(p,y))
    Acc = lambda p,y : T.mean(T.eq(T.argmax(p, axis = 1),y))
    Ce = lambda p,y : -T.mean(T.log(p)[T.arange(y.shape[0]), y])
    Nll = lambda p,y,n: -T.sum(T.log(p)*T.extra_ops.to_one_hot(y, n))
    L1 = lambda w : T.sum(T.sqrt(T.pow(w,2)))
    L2 = lambda w : T.sum(T.pow(w,2))

########################### Methods ##############################

    @staticmethod
    def Mse(x,y,dims=4):
        mod = np.arange(1,dims)
        return T.mean(T.sum(T.pow(x - y,2),axis=mod)) 
    
    @staticmethod
    def Mae(x,y,dims=4,eps=1e-7):
        mod = np.arange(1,dims)
        return T.mean(T.sum(T.sqrt(T.pow(x - y,2) + eps),axis=mod))

    @staticmethod
    def DSSIM(p, y, eps = 1e-7):
        # Taken/Modified from https://github.com/fchollet/keras/issues/4292
        # Nan issue : T.maximum(x, eps) 
        
        y_patch = neigh.images2neibs(y, [4,4], mode='ignore_borders')
        p_patch = neigh.images2neibs(p, [4,4], mode='ignore_borders')
        
        y_mean = T.mean(y_patch, axis=-1)
        p_mean = T.mean(p_patch, axis=-1)
        
        y_var = T.var(y_patch, axis=-1, corrected=True)
        p_var = T.var(p_patch, axis=-1, corrected=True)

        y_std = T.sqrt(T.maximum(y_var,eps))
        p_std = T.sqrt(T.maximum(p_var,eps))
        
        c1 = 0.01 ** 2
        c2 = 0.02 ** 2
        
        num = (2 * y_mean * p_mean + c1)*(2 * y_std * p_std + c2) 
        denom = (T.pow(y_mean,2) + T.pow(p_mean,2) + c1)*(y_var + p_var + c2)
        
        ssim = num/T.maximum(denom,eps)
        
        return T.mean(1.0 - ssim)

    @staticmethod
    def rmsprop_momentum(loss,params,eta=1e-3,alpha=0.9,**kwargs):
        rms = updt.rmsprop(loss, params, learning_rate = eta, **kwargs)
        return updt.apply_momentum(rms, params, momentum = alpha)
    
    @staticmethod
    def rmsprop_nesterov(loss,params,eta=1e-3,alpha=0.9,**kwargs):
        rms = updt.rmsprop(loss, params, learning_rate = eta, **kwargs)
        return updt.apply_nesterov_momentum(rms, params, momentum = alpha)

    @staticmethod
    def getpath():
        if os.environ['LOC'] == 'local':
            return '..'
        elif os.environ['LOC'] == 'hades':
            return '/home2/ift6ed13'

    @staticmethod
    def setinit(fan_in, fan_out, act, size=None):
        if not size:
            size = (fan_in,fan_out)
        x = np.sqrt(6. / (fan_in + fan_out)) * (4. if act == 'sigmoid' else 1.)
        return rng.uniform(-x,x,size=size).astype(theano.config.floatX)

    @staticmethod
    def setseed():
        raise NotImplementedError

########################### Classes ##############################

    class Timer:
        """
        Embedding timer class. Ex:
        with Timer() as t:
            run_code
        print('Code took %.03f sec.' % t.interval)
        """
        def __enter__(self):
            self.start = time.clock()
            return self
    
        def __exit__(self, *args):
            self.end = time.clock()
            self.interval = self.end - self.start

#*****************************************************************************#
#*****************************************************************************#

class InputLayer:
    def __init__(self,inputs):
        self.output = inputs
        self.params = []
        
class ReshapeLayer:
    def __init__(self,inputs,shape):
        inputs = inputs.output
        self.output = inputs.reshape(shape)
        self.params = []

class SumLayer:
    c=0
    def __init__(self,input1,input2,activation=None,ratio=None):
        input1 = input1.output
        input2 = input2.output
        
        SumLayer.c += 1
        c = SumLayer.c
        
        alpha = rng.uniform(0.,1.,size=(2,)).astype(theano.config.floatX)
        A = theano.shared(alpha[0],'A'+str(c)) if ratio is None else ratio

        self.output = A*input1 + (1. - A)*input2
        self.params = [A] if ratio is None else []

class DenseLayer:
    """
    Fully-connected dense layer class.
    ----------------------------------

    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : n_in : int : Number of units in the previous layer

        : n_out : int : Number of output units

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : Layer's output
    """
    c = 0
    def __init__(self,inputs,n_in,n_out,activation=None):

        assert activation in Tool.fct.keys()

        DenseLayer.c += 1
        c = DenseLayer.c

        inputs = inputs.output.flatten(2)

        w = Tool.setinit(n_in,n_out,activation)
        b = np.zeros(n_out,dtype=theano.config.floatX)

        self.W = theano.shared(w,'W'+str(c))
        self.B = theano.shared(b,'B'+str(c))

        self.params = [self.W,self.B]
        self.output = Tool.fct[activation](T.dot(inputs,self.W) + self.B)


class ConvLayer:
    """
    Convolutional + Maxpooling layer class.
    ---------------------------------------
    
    # Arguments :
        : inputs : 4D tensor : Shape must be (batch size, channels, height, width)

        : nkernels : int : Number of kernels
        
        : kernerlsize : 2-tuple : Height and width of kernel

        : poolsize : 2-tuple : Height and width for maxpooling

        : act : string : Activation function (None,sigmoid,tanh,relu,softmax)
        
        : pad : string, int, tuple : Padding mode ('full','half','valid') or (pad height, pad width)
        
        : stride : int, tuple : Strides
        
        : pmode : string : 'max' or 'average_inc_pad'

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = outshape

    # Functions : 
        : outshape : Returns the shape of the output image
    """
    c = 0
    def __init__(self,inputs,nchan,nkernels,kernelsize,poolsize=(1,1),
                 activation='relu',pad='valid',stride=(1,1), pmode='max',
                 pstride=None,ppad=(0,0),train=True):

        assert activation in Tool.fct.keys()

        ConvLayer.c += 1
        c = ConvLayer.c
        
        inputs = inputs.output
        
        filter_shape = (nkernels,nchan) + kernelsize

        fan_in = nchan * np.prod(kernelsize)
        fan_out = (nkernels * np.prod(kernelsize)) // np.prod(poolsize)

        w = Tool.setinit(fan_in, fan_out, activation, size = filter_shape)
        b = np.zeros(nkernels,dtype=theano.config.floatX)

        self.W = theano.shared(w,'W_conv'+str(c))
        self.B = theano.shared(b,'B_conv'+str(c))

        convolution = Tool.conv(inputs, self.W, border_mode = pad, subsample = stride)
        out = Tool.pool(convolution, poolsize, True, pstride, ppad, pmode)

        self.params = [self.W,self.B]            
        self.output = Tool.fct[activation](out + self.B.dimshuffle('x', 0, 'x', 'x'))

        self.shape = lambda x: self.outshape(x, kernelsize[0], pad, stride[0], poolsize[0])
        self.arguments = {'nchan':nchan, 'nkernels':nkernels,'kernelsize':kernelsize,'poolsize':poolsize,
                          'activation':activation,'pad':pad,'stride':stride,'W':self.W,'B':self.B}

    def outshape(self,inp, k, p, s, pool):
        if p == 'valid':
            return int((np.floor((inp - k)/s) + 1)/pool)
        elif p == 'full':
            return int((np.floor((inp - k + 2*(k-1))/s) + 1)/pool)
        elif p == 'half':
            return int((np.floor((inp - k + 2*(k//2))/s) + 1)/pool)
        else:
            return int(np.floor(((inp - k + 2*p)/s) + 1 )/pool)


class TConvLayer:
    """
    Transposed Convolutional + Upsampling layer class.
    --------------------------------------------------
    IMPORTANT NOTE
    --------------
    This class is implemented to perform the transposed convolution of a given ConvLayer.
    All the arguments except the first two should be provided by the corresponding ConvLayer :
    
        TransposeConvLayer(some_input, shape, **direct_conv_layer.arguments)

    # Arguments :
        : inputs : 4D tensor : Previous layer
        
        : shape : tuple : Shape of the corresponding convolution
            
        : **args : dict : Corresponding direct convolution layer arguments

        : tied : bool : If true, will use the same kernels and biases as the direct convolution
        
    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = floor((i + 2*pmode - kernelsize) / stride) + 1
    """
    c=0
    def __init__(self,inputs,shape,nchan,nkernels,kernelsize,W,B,poolsize,
                 activation,pad,stride,batch,tied=True):

        assert activation in Tool.fct.keys()

        TConvLayer.c += 1
        c = TConvLayer.c
        
        inputs = inputs.output
        
        filter_shape = (nkernels,nchan) + kernelsize
        inp_shape = (None,nchan) + shape
                    
        if tied:
            self.W = theano.shared(W.eval(),'W_dconv'+str(c))
            self.B = theano.shared(B.eval(),'B_dconv'+str(c))
            
            inputs += self.B.dimshuffle('x',0,'x','x')
    
            upsampled = Tool.upsamp(inputs, poolsize[0], batch, nkernels) # batch =  inputs.shape[0]
            deconved = Tool.deconv(upsampled, self.W, inp_shape, border_mode=pad, subsample=stride)
            
        else:
            del W,B 
            
            fan_in = nchan * np.prod(kernelsize)
            fan_out = (nkernels * np.prod(kernelsize)) // np.prod(poolsize)
            
            w = Tool.setinit(fan_in, fan_out, activation, size = filter_shape)
            b = np.zeros((nchan,) + shape,dtype=theano.config.floatX)
    
            self.W = theano.shared(w,'W_dconv'+str(c))
            self.B = theano.shared(b,'B_dconv'+str(c))
        
            upsampled = Tool.upsamp(inputs, poolsize[0], batch, nkernels) # batch =  inputs.shape[0]
            deconved = Tool.deconv(upsampled, self.W, inp_shape, border_mode=pad, subsample=stride)
            
            deconved += self.B.dimshuffle('x',0,1,2)

        self.params = [self.W,self.B] if tied else [self.W,self.B]
        self.output = Tool.fct[activation](deconved)


class RecurrentLayer:
    """
    Recurrent layer class.
    ----------------------
    
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer

        : channels : int : Word size or number of Channels

        : hdim : int : Dimension of hidden layer

        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)

    # Attributes :
        : params : list : List of all the parameters of the layer

        : output : ndarray or T.tensor : size = hdim

    # Functions :
        : __step__ : Updates cell state
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.fct.keys()

        RecurrentLayer.c += 1
        c = RecurrentLayer.c
        
        inputs = inputs.output
        
        w = Tool.setinit(channels, hdim, activation)
        v = Tool.setinit(hdim, hdim, activation)

        h0 = theano.shared(np.zeros((hdim), dtype=theano.config.floatX),'h0')
        b = np.zeros(hdim, dtype=theano.config.floatX)

        W = theano.shared(w,'Wrec'+str(c))
        V = theano.shared(v,'Vrec'+str(c))
        B = theano.shared(b,'Brec'+str(c))
        self.params = [W,V,B]
        self.weights = [W,V]

        H, _ = theano.scan(self.__step__,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.fct[activation](H[-1])

    def __step__(self, x, h_prev, W, V, B):
        return T.tanh(T.dot(x,W) + T.dot(h_prev,V) + B)



class LSTMLayer:
    """
    LSTM layer class.
    -----------------
    
    # Arguments :
        : inputs : ndarray or T.tensor : Previous layer
    
        : channels : int : Word size or number of Channels
    
        : hdim : int : Dimension of hidden layer
    
        : activation : string : Activation function (None,sigmoid,tanh,relu,softmax)
    
    # Attributes :
        : params : list : List of all the parameters of the layer
    
        : output : ndarray or T.tensor : size = hdim
    
    # Functions :
        : __step__ : Updates cell state
    """
    c = 0
    def __init__(self,inputs,channels,hdim,truncate=-1,activation='tanh'):

        assert activation in Tool.fct.keys()

        LSTMLayer.c += 1
        c = LSTMLayer.c
        
        inputs = inputs.output
        
        self.V = Tool.setinit(hdim, hdim, activation)
        self.V = theano.shared(self.V,'V'+str(c))

        self.W, self.U, self.B = {},{},{}

        for k in ['i','f','c','o']: # Input, Forget, Cell, Output
            self.W[k] = Tool.setinit(channels, hdim, activation)
            self.W[k] = theano.shared(self.W[k],'Wlstm'+str(k)+str(c))

            self.U[k] = Tool.setinit(hdim, hdim, activation)
            self.U[k] = theano.shared(self.U[k],'Ulstm'+str(k)+str(c))

            self.B[k] = np.zeros(hdim, dtype=theano.config.floatX)
            self.B[k] = theano.shared(self.B[k],'Blstm'+str(k)+str(c))

        h0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX),'h0')
        c0 = theano.shared(np.zeros((hdim),dtype=theano.config.floatX),'c0')

        val = lambda x : [x[k] for k in ['i','f','c','o']] #list(x.values())
        self.params = val(self.W) + val(self.U) + val(self.B)
        self.weights = val(self.W) + val(self.U)

        [H,C], _ = theano.scan(self.__step__,
                           sequences=inputs,
                           non_sequences=self.params,
                           outputs_info=[T.repeat(h0[None, :],inputs.shape[1], axis=0),
                                         T.repeat(c0[None, :], inputs.shape[1], axis=0)],
                           truncate_gradient=truncate,
                           strict=True)

        self.output = Tool.fct[activation](H[-1])
        self.state = C[-1]

    def __step__(self, xt, h_prev, c_prev, *yolo):
        # Not taking the parameters from scan but from self (Should not make a difference)

        i,f,c,o = [T.nnet.sigmoid(T.dot(xt,self.W[k]) + T.dot(h_prev,self.U[k]) + self.B[k])
                    for k in ['i','f','c','o']]

        c = i * c + f * c_prev
        h = o * T.tanh(c)
        
        return h,c



class Dropout:
    """
    Dropout layer class.
    --------------------
    
    # Arguments :
        : weight : ndarray or T.tensor : The weights we want to drop out

        : train : True if training phase, False else

        : drop : float32 : Proportion to dropout from the weight

        : seed : int : Random seed for generator (optional)
    """
    def __init__(self,inputs,drop=.5,train=True):

        inputs = inputs.output
        self.drop = drop
        self.srng = RandomStreams(rng.randint(2**31))
        self.output = self.__drop__(inputs) if train else self.__scale__(inputs)

    def __drop__(self, weight):
        """
        # Returns: Dropped out matrix with binomial probability
        """
        mask = self.srng.binomial(n=1, p=1-self.drop, size=weight.shape, dtype=theano.config.floatX)
        return T.cast(weight * mask, theano.config.floatX)

    def __scale__(self, weight):
        """
        # Returns: Scaled matrix
        """
        return (1 - self.drop) * T.cast(weight, theano.config.floatX)



class BatchNorm:
    """
    Batch Normalization layer class.
    --------------------------------
    """
    c=0
    def __init__(self,inputs,channels,activation,dims=2,batch=None):
        
        assert dims in [2,4,None]
        
        BatchNorm.c += 1
        c = BatchNorm.c
        
        inputs = inputs.output
        
        
        if dims == 2:
            
            g = np.ones((channels,), dtype=theano.config.floatX)
            b = np.zeros((channels,), dtype=theano.config.floatX)
            self.G = theano.shared(g,'G_bn'+str(c))
            self.B = theano.shared(b,'B_bn'+str(c))
            
            mean = T.mean(inputs,axis=0) if batch is None else T.mean(batch,axis=0)
            std = T.std(inputs,axis=0) if batch is None else T.std(batch,axis=0)

            self.params = [self.G,self.B]
            self.stats = [mean,std]
            A = self.G * (inputs - mean) / std + self.B
            self.output = Tool.fct[activation](A)

        elif dims == 4:
            
            g = np.ones((channels,), dtype=theano.config.floatX)
            b = np.zeros((channels,), dtype=theano.config.floatX)
            self.G = theano.shared(g,'G_bn'+str(c))
            self.B = theano.shared(b,'B_bn'+str(c))
            
            if batch is None:
                mean = T.mean(inputs,axis=(0,2,3)).dimshuffle('x',0,'x','x')
                std = T.std(inputs,axis=(0,2,3)).dimshuffle('x',0,'x','x') 
            else :
                mean = T.mean(batch,axis=(0,2,3)).dimshuffle('x',0,'x','x')
                std = T.std(batch,axis=(0,2,3)).dimshuffle('x',0,'x','x')

            self.params = [self.G,self.B]
            self.stats = [mean,std]
            A =  self.G.dimshuffle('x',0,'x','x') * (inputs - mean) / std + self.B.dimshuffle('x',0,'x','x')
            self.output = Tool.fct[activation](A)

        elif dims == None:
            mean,std = None,None
            self.output = Tool.fct[activation](inputs)
            self.params = []
        
        self.mean = mean
        self.std = std
