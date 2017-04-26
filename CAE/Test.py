#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date: April 2017
@author: Yannick Roy

Adapted from mcomin 
https://github.com/massimilianocomin/Class-Project-IFT-6266
He did an awesome job for making a small model that can actually produce "something"
and that can be trained in a decent amount of time for preliminary encouraging results.

--

This script allows to test a pretrained model. 
This script loads parameters from different epoc recording and display
images to compare the evolution of the training.
"""
import os
os.environ['LOC'] = 'local'

from ConvAE import *
from Img import *
import matplotlib.pyplot as plt
import numpy as np

batch_size = 10
batch_no = 1
batch_type = 'valid'
image_no = 6

#Insert the 5 epoc numbers you'd like to compare.
model_trained = [5,10,15,20,25]

for imgno in range(9):
    image_no = imgno
    reconstructions = []
    
    for i in range(5):
        M = Model(batch_size)
        M.__load__(model_trained[i])
        M.Generate(batch_type, batch_no)  
        reconstructions.append(M.valid_recon[image_no])
        
    I = Img()
    curbatch, curcrop = I.load_batch(batch_size, batch_no, batch_type)
    curcropped = np.copy(curbatch)
    curbatch[:,:,16:48,16:48] = curcrop
        
    #I.plotandcompare(M.valid_recon[image_no], curbatch[image_no])
    
    for i in range(len(reconstructions)):
        reconstructions[i] = reconstructions[i].transpose(1,2,0)
    original = curbatch[image_no].transpose(1,2,0)    
    challenge = curcropped[image_no].transpose(1,2,0)    
    
    fig = plt.figure(figsize=(17,3))
     
    ax1 = fig.add_subplot(171)
    ax1.imshow(challenge)
    ax1.axis("off")
    ax1.set_title('Crop')
    
    ax2 = fig.add_subplot(172)
    ax2.imshow(reconstructions[0])
    ax2.axis("off")
    ax2.set_title('#' + str(model_trained[0]))
    ax3 = fig.add_subplot(173)
    ax3.imshow(reconstructions[1])
    ax3.axis("off")
    ax3.set_title('#' + str(model_trained[1]))
    ax4 = fig.add_subplot(174)
    ax4.imshow(reconstructions[2])
    ax4.axis("off")
    ax4.set_title('#' + str(model_trained[2]))
    ax5 = fig.add_subplot(175)
    ax5.imshow(reconstructions[3])
    ax5.axis("off")
    ax5.set_title('#' + str(model_trained[3]))
    ax6 = fig.add_subplot(176)
    ax6.imshow(reconstructions[4])
    ax6.axis("off")
    ax6.set_title('#' + str(model_trained[4]))
    
    ax7 = fig.add_subplot(177)
    ax7.imshow(original)
    ax7.axis("off")
    ax7.set_title('Original')
    
    plt.savefig('Results_Img_' + str(image_no) + '.png')
