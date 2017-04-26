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
import os
import sys
import numpy as np

if os.environ['LOC'] == 'local':
    path = '/Users/yannick/Documents/PhD/UdeM - Cours/IFT6266/Project/inpainting'
    libpath = '/Users/yannick/Documents/Playground/Python/massim/Class-Project-IFT-6266/lib'

elif os.environ['LOC'] == 'hades':
    path = '/home2/ift6ed54/data'
    libpath = '/home2/ift6ed54/lib'
    import matplotlib
    matplotlib.use('Agg')
else:
    sys.exit('Environment variable LOC not found. Verify .bash_profile.')
    
    
import matplotlib.pyplot as plt
import PIL.Image as Image
import theano
import glob
import pickle

theano.config.floatX = 'float32'
theano.config.intX = 'int32'


class Img:
    
    def __init__(self):

        with open(libpath+'/SKIP_NAMES','rb') as file:
            self.IMG_TO_SKIP = pickle.load(file)

        if len(self.IMG_TO_SKIP) == 0:
            print('No Image to Skip, according to :', libpath+'/SKIP_NAMES', ' ... Weird? Checking again!')
            self._DETECT_GRAYSCALE_IMG()
            
        trainlist = glob.glob(path+'/train2014/*.jpg')
        validlist = glob.glob(path+'/val2014/*.jpg')
        
        #print('Train List Len: ', len(trainlist), ' from ', path+'/train2014/*.jpg')
        #print('Valid List Len: ', len(validlist), ' from ', path+'/val2014/*.jpg')
        
        if len(trainlist) == 0:
            print('## ERROR ## Empty Train List')
        if len(validlist) == 0:
            print('## ERROR ## Empty Valid List')

        self.trainlist = [x for x in trainlist if x not in self.IMG_TO_SKIP]
        self.validlist = [x for x in validlist if x not in self.IMG_TO_SKIP]

        #print('Train List Len (in color): ', len(self.trainlist), ' from ', path+'/train2014/*.jpg')
        #print('Valid List Len (in color): ', len(self.validlist), ' from ', path+'/val2014/*.jpg')

    def load(self,n=None):
        """
        Loads the whole dataset, or the first n examples.
        """
        
        print('Loading COCO dataset...')
        
        train = [np.array(Image.open(fname)) for fname in trainlist[0:n]] # self.
        train = np.array(train)
        train = (train[0:n]/256).astype(theano.config.floatX)
        train = train.transpose(0,3,1,2)

        valid = [np.array(Image.open(fname)) for fname in self.validlist[0:n]]
        valid = np.array(valid)
        valid = (valid[0:n]/256).astype(theano.config.floatX)
        valid = valid.transpose(0,3,1,2)

        train_crop = np.copy(train)
        train_crop[:,:,16:48,16:48] = 0
        valid_crop = np.copy(valid)
        valid_crop[:,:,16:48,16:48] = 0

        print('Dataset loaded.')

        return train_crop, train, valid_crop, valid


    def load_batch(self,batchsize,i,mode):
        """
        Loads successive minibatches of the dataset.
        """
        
        if mode == 'train':
            batch_list = self.trainlist[i*batchsize:(i+1)*batchsize]
        elif mode == 'valid':
            batch_list = self.validlist[i*batchsize:(i+1)*batchsize]
        else:
            sys.exit('Img.load_batch Error: Please select a valid mode.')
            
        #print('Batch No: ', i, 'Batch Size: ', batchsize, ' Batch Len: ', len(batch_list), ' Batch Mode: ', mode, ' | trainlist : ', len(self.trainlist), ' | validlist : ', len(self.validlist))
        
        batch = [np.array(Image.open(fname)) for fname in batch_list]
        batch = np.array(batch[0:batchsize],dtype=theano.config.floatX)/256.
        batch = batch.transpose(0,3,1,2)

        batch_crop = np.copy(batch)
        batch_crop[:,:,16:48,16:48] = 0
        
        return batch_crop, batch[:,:,16:48,16:48]


    def plot(self,inp,imgpath=None):
        
        image = inp.transpose(1,2,0)

        plt.axis("off")
        plt.imshow(image)
        if imgpath:
            plt.savefig(imgpath+'.png')
    
    
    def plotandcompare(self,original,reconstruction):
        image_original = original.transpose(1,2,0)
        image_reconstruction = reconstruction.transpose(1,2,0)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(image_original)
        ax1.axis("off")
        ax2 = fig.add_subplot(122)
        ax2.imshow(image_reconstruction)
        ax2.axis("off")
    
        
    def save(self,image,imgpath):
        
        image = image.transpose(1,2,0)
        
        with open(imgpath,'wb') as file:
            pickle.dump(image,file, 2)


    def _DETECT_GRAYSCALE_IMG(self):
        
        trainlist = glob.glob(path+'/train2014/*.jpg')
        validlist = glob.glob(path+'/val2014/*.jpg')

        skipnames = []
        
        for name in trainlist:
            op = Image.open(name)
            img = np.array(op)
            if img.shape != (64,64,3):
                skipnames += [name]
                print('Found Error')
            op.close()

        for name in validlist:
            op = Image.open(name)
            img = np.array(op)
            if img.shape != (64,64,3):
                skipnames += [name]
                print('Found Error')
            op.close()
        
        with open(libpath+'/SKIP_NAMES','wb') as file:
            pickle.dump(skipnames,file,-1)
    