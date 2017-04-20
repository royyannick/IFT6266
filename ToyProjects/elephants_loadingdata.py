#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:19:27 2017

@author: nick
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import cv2
#
# folderO = "/Users/yannick/Documents/Playground/Python/data/x_vs_o/circles"
# folderX = "/Users/yannick/Documents/Playground/Python/data/x_vs_o/crosses"
#
# N = len(os.listdir(folderO))
# dataO = np.empty((N, 32, 32), dtype=np.uint8)
# N = len(os.listdir(folderX))
# dataX = np.empty((N, 32, 32), dtype=np.uint8)
#
# for i, fpath in enumerate(os.listdir(folderO)):
#     print(i, ":", fpath)
#     dataO[i] = cv2.resize(cv2.imread(os.path.join(folderO, fpath), cv2.IMREAD_GRAYSCALE), (32,32), interpolation=cv2.INTER_AREA)
#
# for i, fpath in enumerate(os.listdir(folderX)):
#     print(i, ":", fpath)
#     dataX[i] = cv2.resize(cv2.imread(os.path.join(folderX, fpath), cv2.IMREAD_GRAYSCALE), (32,32), interpolation=cv2.INTER_AREA)
#
# data_x = np.append(dataO, dataX, axis=0)
# data_y = np.append(np.zeros(len(dataO)), np.ones(len(dataX)))
#
# f, axes = plt.subplots(2, 2, sharey=True)
# axes[0][0].set_axis_off()
# axes[0][0].imshow(dataO[100])
# axes[1][0].set_axis_off()
# axes[1][0].imshow(dataX[100])
# axes[0][1].set_axis_off()
# if data_y[100] == 0:
#     axes[0][1].set_title("Circle")
# else:
#     axes[0][1].set_title("Cross")
# axes[0][1].imshow(data_x[100])
# axes[1][1].set_axis_off()
# if data_y[1100] == 0:
#     axes[1][1].set_title("Circle")
# else:
#     axes[1][1].set_title("Cross")
# axes[1][1].imshow(data_x[1100])


## VERSION 2

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

size_pix = 64
folderO = "/home/nick/Documents/data/elephants_crop"

N = len(os.listdir(folderO))
dataO = np.empty((N, size_pix, size_pix, 3), dtype=np.uint8)

for i, fpath in enumerate(os.listdir(folderO)):
    print(i, ":", fpath)
    dataO[i] = cv2.resize(cv2.imread(os.path.join(folderO, fpath), cv2.IMREAD_COLOR), (size_pix,size_pix), interpolation=cv2.INTER_AREA)
    #dataO[i] = cv2.resize(cv2.imread(os.path.join(folderO, fpath), cv2.IMREAD_GRAYSCALE), (size_pix,size_pix), interpolation=cv2.INTER_AREA)


data_x_train = dataO[:1200, ...]
data_x_test = dataO[1200:, ...]

plt.figure()
plt.axis("off")
plt.imshow(dataO[245])


## VERSION 3

import os, sys
import glob
#import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize

#img = Image.open("/home/nick/Documents/data/elephants/n02504013_1673.JPEG")
#img.show()

def resize_images():
    # From IFT6266 2017 Project 
    
    ### PATH need to be fixed
    data_path="/home/nick/Documents/data/elephants/"
    save_dir = "/home/nick/Documents/data/elephants_crop/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)

    imgs = glob.glob(data_path+"/*.JPEG")


    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print(i, len(imgs), img_path)

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

        img.save(save_dir + os.path.basename(img_path))
        
        
img = Image.open("/home/nick/Documents/data/elephants/n02504013_1673.JPEG")
img.show()