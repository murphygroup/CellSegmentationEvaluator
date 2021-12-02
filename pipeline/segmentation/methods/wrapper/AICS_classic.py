#%% md

# Segmenting NUP153

#%%

# import python packages
import glob
import os
import sys
from skimage.io import imread
# import packages
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from aicssegmentation.core.visual import seg_fluo_side_by_side,  single_fluorescent_view, segmentation_quick_view
# package for io
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from itkwidgets import view

# function for core algorithm
from aicssegmentation.core.vessel import filament_3d_wrapper
from aicssegmentation.core.vessel import filament_2d_wrapper
import aicssegmentation.core.seg_dot as seg_dot
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, watershed, dilation, erosion, ball       # function for post-processing (size filter)
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.io import imsave
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

#%%

os.getcwd()

#%% md

## Loading the Data


file_dir = sys.argv[1]
Data1 = imread(join(file_dir, 'membrane.tif'))
Data2 = imread(join(file_dir, 'nucleus.tif'))
Data = Data1 + Data2
#%%


#####################
structure_channel = 1
#####################

# struct_img0 = IMG[0,structure_channel,:,:,:].copy()
struct_img0 = Data.copy()
# view(single_fluorescent_view(struct_img0))

#%%

################################
## PARAMETERS for intensity normalization ##
intensity_scaling_param = [3, 3]
################################
struct_img = intensity_normalization(struct_img0.copy(), scaling_param=intensity_scaling_param)




################################
## PARAMETERS for smoothing ##
gaussian_smoothing_sigma = 1
################################

# 3d gaussian smoothing makes more sense
structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

# edge-preserving smoothing


################################
## PARAMETERS for this step ##
s3_param = [[8, 0.05]]
################################

bw_3d = seg_dot.dot_3d_wrapper(structure_img_smooth, s3_param)



# watershed
minArea = 10
Mask = remove_small_objects(bw_3d>0, min_size=minArea, connectivity=1, in_place=False)
Seed = dilation(peak_local_max(struct_img, labels=label(Mask), min_distance=2, indices=False), selem=ball(1)[1,:,:])
Watershed_Map = -1*distance_transform_edt(bw_3d)
seg = watershed(Watershed_Map, label(Seed), mask=Mask, watershed_line=False)



minArea = 20

final_seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)

final_seg = label(final_seg, background=0, connectivity=2)
np.save(join(file_dir, 'mask_AICS_classic.npy'), final_seg)
