#import re
#import xml.etree.ElementTree as ET
#import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
#import json
#from aicsimageio import AICSImage
import numpy as np
import xmltodict
#import pandas as pd
#from PIL import Image
#from pint import Quantity, UnitRegistry
#from scipy.sparse import csr_matrix
#from scipy.stats import variation
#from skimage.filters import threshold_mean, threshold_otsu
#from skimage.morphology import area_closing, closing, disk
#from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
#from skimage.segmentation import find_boundaries
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
#import tifffile
from CellSegmentationEvaluator.CSE3D import CSE3D
from CellSegmentationEvaluator.functions import get_voxel_volume
import warnings

warnings.filterwarnings('ignore')

"""
Package functions that evaluate a single cell segmentation mask for a single image
Authors: Haoran Chen and Ted Zhang and Robert F. Murphy
Version: 1.4 December 11, 2023 R.F.Murphy
        repair nuclear masks outside cell masks and mismatched cells and nuclei
         1.5 January 18, 2024 R.F.Murphy
        add CSE3D as simpler function for 3D evaluation
         1.5.13 April 3, 2025 R.F.Murphy
        remove pint
"""

def single_method_eval_3D(img, mask, PCA_model, output_dir: Path, unit='um', pixelsizex=1, pixelsizey=1,
                          pixelsizez=2) -> Tuple[Dict[str, Any], float, float]:
	print("Calculating evaluation metrics v1.6 for", img['name'])

	# get compartment masks
	try:
		mask_channel_names = mask["img"].get_channel_names()
		mask_cell_channel_index = all_channel_names.index(seg_channel_names["Cell"])[0]
		mask_nuclear_channel_index = all_channel_names.index(seg_channel_names["Nucleus"])[0]
		#print('mask channels=',mask_cell_channel_index,mask_nuclear_channel_index)
	except:
		mask_cell_channel_index = 0
		mask_nuclear_channel_index = 1
	matched_mask = mask["data"][0, :, :, :, :]
	cell_matched_mask = matched_mask[mask_cell_channel_index]
	nuclear_matched_mask = matched_mask[mask_nuclear_channel_index]
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask

	metric_mask = np.expand_dims(cell_matched_mask, 0)
	metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
	metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))

	# find channels to use to separate image foreground from background
	thresh_params={}
	try:
		img_xmldict = xmltodict.parse(img["img"].metadata.to_xml())
		seg_channel_names = img_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"]["Value"][
			"OriginalMetadata"
		]["Value"]
		all_channel_names = img["img"].get_channel_names()
		nuclear_channel_index = all_channel_names.index(seg_channel_names["Nucleus"])
		cell_channel_index = all_channel_names.index(seg_channel_names["Cell"])
		thresholding_channels = [nuclear_channel_index, cell_channel_index]
		#print('nuclear_channel_index')
		#print(nuclear_channel_index)
		#print('cell_channel_index')
		#print(cell_channel_index)
		img4thresh = np.squeeze(sum(thresholding(img["data"][0,c,:,:,:]) for c in thresholding_channels))
		print('Using sum of provided "Nucleus" and "Cell" channels for segmenting image foreground from background')

	except:
		img4thresh=0.2
		print('Using sum of all channels for segmenting image foreground from background')
	try:
		units, voxel_size = get_voxel_volume(img["img"])
		#print("get_voxel_volume successful")
	except:
		#print("in exception")
		reg = UnitRegistry()
		reg.define("cell = []")
		units = reg(unit)
		sizes = [pixelsizex * units, pixelsizey * units, pixelsizez * units]
		#print('pixel sizes=',sizes)
		units = reg
		voxel_size = sizes[0] * sizes[1] * sizes[2]

	vox_size = voxel_size.to("micrometer ** 3")
	#print(vox_size)
	vox_size_unitless = vox_size.magnitude

	#disksizes = (1, 2, 20, 10) #these were used in CellSegmentationEvaluator v1.4
	disksizes = (1, 2, 10, 3) #these were used by 3DCellComposer v1.1
	#areasizes = (20000, 1000) #these were used in CellSegmentationEvaluator v1.4
	#areasizes = (5000, 1000) #these were used by 3DCellComposer v1.1
	areasizes = (np.round(20000/vox_size_unitless), np.round(1000/vox_size_unitless)) #20000 is approximately 5 cell volumes (cubic microns)

	metrics = CSE3D(np.squeeze(img["data"]),metric_mask,PCA_model,img4thresh,vox_size_unitless,disksizes,areasizes)
	return metrics
