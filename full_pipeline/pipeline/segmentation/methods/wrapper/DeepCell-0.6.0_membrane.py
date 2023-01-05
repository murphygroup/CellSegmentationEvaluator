from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from deepcell.applications import MultiplexSegmentation
import os
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt
import bz2
import pickle

file_dir = sys.argv[1]
im1 = imread(join(file_dir, 'nucleus.tif'))
im2 = imread(join(file_dir, 'membrane.tif'))
im = np.stack((im1, im2), axis=-1)
im = np.expand_dims(im, 0)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
#
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
app = MultiplexSegmentation(use_pretrained_weights=True)


labeled_image = app.predict(im, compartment='both')
labeled_image = np.squeeze(labeled_image, axis=0)
cell_mask = labeled_image[:, :, 0]
nuc_mask = labeled_image[:, :, 1]

imsave(join(file_dir, 'mask_DeepCell-0.6.0_membrane.png'), cell_mask)
np.save(join(file_dir, 'mask_DeepCell-0.6.0_membrane.npy'), cell_mask)
imsave(join(file_dir, 'nuclear_mask_DeepCell-0.6.0_membrane.png'), nuc_mask)
np.save(join(file_dir, 'nuclear_mask_DeepCell-0.6.0_membrane.npy'), nuc_mask)
