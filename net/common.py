SEED = 202

#### kitti dataset orijection from lidar to top, front and rgb ####

TOP_Y_MIN=-30  #40
TOP_Y_MAX=+30
TOP_X_MIN=-4
TOP_X_MAX=40   #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX= 0.4

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.1

Top_Y0, Top_Yn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
Top_X0, Top_Xn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1

Car_Z0=-1.6
Car_Zn=0.4

import numpy as np
PIXEL_MEANS = np.array([[[95.8814, 98.7743, 93.8549]]])

#rgb camera
MATRIX_Mt = ([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
              [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
              [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
              [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])

MATRIX_Kt = ([[ 721.5377,    0.    ,    0.    ],
              [   0.    ,  721.5377,    0.    ],
              [ 609.5593,  172.854 ,    1.    ]])


MATRIX_TR = ([[6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02],
			 [-1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02],
			 [9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01],
			 [0.0				, 0.0				, 0.0				, 1.0				]])

kitti_dir = "/home/hhs/4T/datasets/KITTI/object/training"
train_data_root= '/home/hhs/4T/datasets/Last_14000'
# train_data_root= '/home/hhs/4T/datasets/dummy_datas/seg'
from net.utility.file import *
makedirs(train_data_root)
import pdb

#----------------------------------

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


#----------------------------------

# std libs
import os
import pickle
from timeit import default_timer as timer
from datetime import datetime
import csv
import pandas as pd
import pickle

# deep learning libs
import tensorflow as tf
tf.set_random_seed(SEED)

# import keras
# from keras import backend as K
#sess = tf.Session()
# K.set_session(sess)
# assert(K._BACKEND=='tensorflow')
#K.learning_phase() #0:test,  1:train
# from keras.models import Sequential, Model
# from keras.layers import Deconvolution2D, Convolution2D, Cropping2D, Cropping1D, Input, merge
# from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation, Reshape
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import PReLU,SReLU,ELU
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras import initializations
#
# from keras.models import load_model
# from keras.optimizers import Adam, SGD
# from keras.regularizers import l2
# from keras.callbacks import LearningRateScheduler




# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)


import cv2
import matplotlib.pyplot as plt
# import mayavi.mlab as mlab


# my libs
