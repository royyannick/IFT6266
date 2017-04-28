from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import numpy as np
import os
import pickle
from utils import get_image
import scipy.misc
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

LR_HR_RETIO = 4
IMSIZE = 64
LOAD_SIZE = int(IMSIZE * 64 / 64)
BIRD_DIR = 'Data/inpainting_64'


def load_filenames(data_dir):
    filepath = BIRD_DIR + '/dict_key_imgID_value_caps_train_and_valid.pkl'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def save_data_list(inpath, outpath, filenames):
    hr_images = []
    lr_images = []
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    cnt = 0
    for key in filenames:
        #bbox = filename_bbox[key]
        f_name = '%s/train2014_64/%s.jpg' % (inpath, key)
        if os.path.isfile(f_name): 
            img = get_image(f_name, LOAD_SIZE, is_crop=True)#, bbox=bbox)
            img = img.astype('uint8')
            hr_images.append(img)
            lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic')
            lr_images.append(lr_img)
            cnt += 1
            if cnt % 100 == 0:
                print('Load %d......' % cnt)
        
    #
    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    #
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)


def convert_birds_dataset_pickle(inpath):
    # ## For Train data
    train_dir = os.path.join(inpath, 'train2014_64/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames)

    # ## For Test data
    test_dir = os.path.join(inpath, 'val2014_64/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames)


if __name__ == '__main__':
    convert_birds_dataset_pickle(BIRD_DIR)
