import numpy as np
import time, os, sys
from os.path import join
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from urllib.parse import urlparse
import mxnet as mx
from cellpose import utils

use_GPU = 1
print('GPU activated? %d'%use_GPU)




file_dir = sys.argv[1]
im1 = imread(join(file_dir, 'cytoplasm.tif'))
im2 = imread(join(file_dir, 'nucleus.tif'))
im = np.stack((im1, im2))
from cellpose import models

# DEFINE CELLPOSE MODEL
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=use_GPU, model_type='cyto')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# or if you have different types of channels in each image
channels = [[0,1]]

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images


masks, flows, styles, diams = model.eval(im, diameter=None, flow_threshold=None, channels=channels)
imsave(join(file_dir, 'mask_Cellpose-0.0.3.1.png'), masks)
np.save(join(file_dir, 'mask_Cellpose-0.0.3.1.npy'), masks)



model_nuc = models.Cellpose(gpu=use_GPU, model_type='nuclei')
channels = [[0, 0]]
nuc_mask, flows, styles, diams = model_nuc.eval(im2, diameter=None, flow_threshold=None, channels=channels, do_3D=False)
imsave(join(file_dir, 'nuclear_mask_Cellpose-0.0.3.1.png'), nuc_mask)
np.save(join(file_dir, 'nuclear_mask_Cellpose-0.0.3.1.npy'), nuc_mask)
