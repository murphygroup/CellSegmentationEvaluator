from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from deepcell.applications import MultiplexSegmentation
# from deepcell.applications import NuclearSegmentation
# from deepcell.applications import CytoplasmSegmentation
import os
from os.path import join
import numpy as np
import sys
import bz2
import pickle
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
# file_dir = '/home/hrchen/Documents/Research/hubmap_local/data/img/R001_X001_Y001'
# im1 = imread(os.path.join(file_dir, 'xafy.tif'))
# im2 = imread(os.path.join(file_dir, 'xach.tif'))
# file_dir = '/home/hrchen/Documents/Research/hubmap_local/data/van'
# file_dir = '/home/hrchen/Documents/Research/hubmap_local/data/van/VAN_datasets'
# file_name = 'VAN0006-LK-2-85-PAS_registered.ome.tiff'
# file_name = 'VAN0006-LK-2-85-AF_preIMS_registered.ome.tiff'
# im = imread(os.path.join(file_dir, file_name))
# plt.imshow(im)
# plt.show()
file_dir = sys.argv[1]
# file_dir = '/home/hrchen/Documents/Research/hubmap_local/data/van/VAN_datasets'
# file_name = 'VAN0006-LK-2-85-PAS_registered.ome.tiff'
# file1_name = 'R001_X001_Y001_c43z5_Ki67.ome.tif'
# file2_name = 'R001_X001_Y001_c7z5_CD16.ome.tif'
# file_name = sys.argv[1]
im1 = imread(join(file_dir, 'nucleus.tif'))
im2 = imread(join(file_dir, 'membrane.tif'))
# im1 = imread(os.path.join(file_dir, file1_name))
# im2 = imread(os.path.join(file_dir, file2_name))
# im = np.squeeze(im, 0)
# im = np.squeeze(im, -1)
im = np.stack((im1, im2), axis=-1)
# Combined together and expand to 4D
# im = np.stack((im1, im2), axis=-1)
# im = np.expand_dims(im, -1)
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
app = MultiplexSegmentation()


labeled_image = app.predict(im, compartment='both')
labeled_image = np.squeeze(labeled_image, axis=0)
cell_mask = labeled_image[:, :, 0]
nuc_mask = labeled_image[:, :, 1]

# import bz2
# import pickle
imsave(join(file_dir, 'mask_DeepCell-0.9.0_membrane.png'), cell_mask)
np.save(join(file_dir, 'mask_DeepCell-0.9.0_membrane.npy'), cell_mask)

# mask_dir = bz2.BZ2File(join(file_dir, 'mask_deepcell_membrane_new.pickle'), 'wb')
# pickle.dump(cell_mask, mask_dir)

imsave(join(file_dir, 'nuclear_mask_DeepCell-0.9.0_membrane.png'), nuc_mask)
np.save(join(file_dir, 'nuclear_mask_DeepCell-0.9.0_membrane.npy'), nuc_mask)

