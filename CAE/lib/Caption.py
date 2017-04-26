#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:31:28 2017

@author: mcomin
"""

import sys
import os
import numpy as np
import _pickle as pickle
import gensim
import string
from collections import OrderedDict

class Caption:
    """
    Caption class.
    --------------
    
    Contains the necessary utilities to perform a word embedding of the captions.
    
    """
    def __init__(self):
        
        if os.environ['LOC'] == 'hades':
            self.datapath = '/home2/ift6ed54/data'
        elif os.environ['LOC'] == 'local':
            self.datapath = '/Users/yannick/Documents/PhD/UdeM - Cours/IFT6266/Project/inpainting'
        
        caption_path = self.datapath + '/dict_key_imgID_value_caps_train_and_valid.pkl'

        with open(caption_path,'rb') as f :
            self.caption = pickle.load(f)
            self.caption_keys = list(self.caption.keys()) # Name of image
            self.caption_vals = list(self.caption.values()) # 5 captions per image
    
    def convert(self,phrase):
        
        rmv_punct = phrase.translate(str.maketrans('','',string.punctuation))
        
        return rmv_punct.split()
    
    def build(self,name):

        
        if name == 'google':
            
            print('Loading Google\'s pre-trained model...')
            
            path = self.datapath + '/GoogleNews-vectors-negative300.bin.gz'
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
            
            print('Loaded.')
            
        elif name == 'local':
            
            print('Training Word2Vec on captions...')
            
            sentences = [self.convert(x) for y in self.caption_vals for x in y]
            self.model = gensim.models.Word2Vec(sentences, size=150, window=5, workers=4, min_count = 1)
            
            print('Model built.')
    
        else:
            sys.exit('No such model exists.')



        print('Creating vector dictionary...')

        self.caption_vec = OrderedDict({})
        
        for i,j in self.caption.items():
            try:
                self.caption_vec[i] = [[self.model[x] for x in self.convert(y)] for y in j]
            except KeyError:
                continue

        print('Vector word embedding created.')
        
    def save(self,filename):
        with open(os.getcwd()+'/'+filename+'.pkl','wb') as file:
            pickle.dump(self.caption_vec, file, 2)
    
    def load(self,filename):
        with open(os.getcwd()+'/'+filename+'.pkl','rb') as file:
            self.caption_vec = pickle.load(file)
