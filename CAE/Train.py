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
from ConvAE import *

M = Model(bs=100,n=80000)
M.Train(epochs=50)

# You can now run Test.py
