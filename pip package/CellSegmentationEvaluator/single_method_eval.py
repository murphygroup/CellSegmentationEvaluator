#import re
#import xml.etree.ElementTree as ET
import math
from pathlib import Path
#from typing import Any, Callable, Dict, List, Tuple
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
from CellSegmentationEvaluator.functions import get_matched_masks, flatten_dict
from CellSegmentationEvaluator.functions import cell_size_uniformity, foreground_separation, fraction, foreground_uniformity, cell_type, cell_uniformity, getPCAmodel, get_quality_score, get_matched_masks, thresholding, get_matched_fraction
#from CellSegmentationEvaluator.functions import *
import warnings

warnings.filterwarnings('ignore')

"""
Package functions that evaluate a single cell segmentation mask for a single image
Authors: Haoran Chen and Ted Zhang and Robert F. Murphy
Version: 1.4 December 11, 2023 R.F.Murphy
        repair nuclear masks outside cell masks and mismatched cells and nuclei
         1.5 January 18, 2024 R.F.Murphy
        add CSE3D as simpler function for 3D evaluation
         1.5.18 July 24, 2025 R.F.Murphy
        import get_matched_masks, thresholding
        improve handling of pixel sizes and remove default values
        deprecate bz and unit
"""

def single_method_eval(img, mask, PCA_model, output_dir: Path, bz=False, unit=False, pixelsizex=False, pixelsizey=False):
	print("Calculating evaluation metrics v1.5 for", img["name"])
	if bz:
		print('bz argument deprecated, it will be ignored')
	if unit:
		print('unit argument deprecated, it will be ignored')
#	if not pixelsizex:
#		print('required pixelsizex argument missing from single_method_eval, exiting...')
#		sys.exit()
#	if not pixelsizex:
#		print('required pixelsizex argument missing from single_method_eval, exiting...')
#		sys.exit()

	cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask = get_matched_masks(mask["data"][:,0,0,:,:],mask["data"][:,1,0,:,:])
	#print(cell_matched_mask.shape)
	cell_matched_mask = np.squeeze(cell_matched_mask)
	nuclear_matched_mask = np.squeeze(nuclear_matched_mask)
	cell_outside_nucleus_mask = np.squeeze(cell_outside_nucleus_mask)
	#print(cell_matched_mask.shape)
	#print(len(np.unique(np.ndarray.flatten(cell_matched_mask))),len(np.unique(np.ndarray.flatten(nuclear_matched_mask))),len(np.unique(np.ndarray.flatten(cell_outside_nucleus_mask))))


	metric_mask = np.expand_dims(cell_matched_mask, 0)
	metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
	metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))

	# separate image foreground background
	try:
		img_xmldict = xmltodict.parse(img["img"].metadata.to_xml())
		seg_channel_names = img_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"]["Value"][
			"OriginalMetadata"
		]["Value"]
		all_channel_names = img["img"].get_channel_names()
		nuclear_channel_index = all_channel_names.index(seg_channel_names["Nucleus"])
		cell_channel_index = all_channel_names.index(seg_channel_names["Cell"])
		thresholding_channels = [nuclear_channel_index, cell_channel_index]
		seg_channel_provided = True
	except:
		thresholding_channels = range(img["data"].shape[1])
		seg_channel_provided = False
		img_thresholded = sum(
			thresholding(np.squeeze(img["data"][0, c, 0, :, :]))
			for c in thresholding_channels
		)
	if not seg_channel_provided:
		img_thresholded[img_thresholded <= round(img["data"].shape[1] * 0.1)] = 0
	# fg_bg_image = Image.fromarray(img_thresholded.astype(np.uint8) * 255, mode="L").convert("1")
	# fg_bg_image.save(output_dir / (img["name"] + "img_thresholded.png"))
	disksizes = (1, 2, 20, 10) #these were used in CellSegmentationEvaluator v1.4
	#disksizes = (1, 2, 10, 3) #these were used by 3DCellComposer v1.1
	areasizes = (20000, 1000) #these were used in CellSegmentationEvaluator v1.4
	#areasizes = (5000, 1000) #these were used by 3DCellComposer v1.1
	img_binary = foreground_separation(img_thresholded,disksizes,areasizes)
	img_binary = np.sign(img_binary)
	background_pixel_num = np.argwhere(img_binary == 0).shape[0]
	fraction_background = background_pixel_num / (img_binary.shape[0] * img_binary.shape[1])
	# np.savetxt(output_dir / f"{img["name"]}_img_binary.txt.gz", img_binary)
	# fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode="L").convert("1")
	# fg_bg_image.save(output_dir / (img["name"] + "img_binary.png"))

	# set mask channel names
	channel_names = [
		"Matched Cell",
		"Nucleus (including nuclear membrane)",
		"Cell Not Including Nucleus (cell membrane plus cytoplasm)",
	]
	metrics = {}
	for channel in range(metric_mask.shape[0]):
		current_mask = metric_mask[channel]
		mask_binary = np.sign(current_mask)
		metrics[channel_names[channel]] = {}
		if channel_names[channel] == "Matched Cell":
			try:
				mask_xmldict = xmltodict.parse(mask["img"].metadata.to_xml())
				matched_fraction = mask_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"][
					"Value"
				]["OriginalMetadata"]["Value"]
			except:
				#matched_fraction = 1.0
				matched_fraction = get_matched_fraction('nonrepaired_matched_mask', np.squeeze(mask["data"][:,0,0,:,:]), cell_matched_mask, np.squeeze(mask["data"][:,1,0,:,:]))
				#print('Matched fraction='+str(matched_fraction))

			if not pixelsizex or not pixelsizey:
				try:
					pixel_size = get_pixel_area(img["img"])
					print('Pixel size=',pixel_size)
				except:
					print('Pixel sizes not specified and cannot be determined from the image metadata')
					sys.exit()
			else:
				pixel_size = pixelsizex * pixelsizey
                                
			pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
			total_area = pixel_size * pixel_num
			# calculate number of cell per 100 squared micron
#			cell_num = units("cell") * len(np.unique(current_mask)) - 1
			cell_num = len(np.unique(current_mask)) - 1
			cells_per_area = cell_num / total_area
#			units.define("hundred_square_micron = micrometer ** 2 * 100")
#			cell_num_normalized = cells_per_area.to("cell / hundred_square_micron")
			cell_num_normalized = cells_per_area*100.
			# calculate the standard deviation of cell size

			_, _, cell_size_std = cell_size_uniformity(current_mask)

			# get coverage metrics
			foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
				img_binary, mask_binary
			)

			img_channels = np.squeeze(img["data"][0, :, 0, :, :])

			foreground_CV, foreground_PCA = foreground_uniformity(
				img_binary, mask_binary, img_channels
			)
			# background_CV, background_PCA = background_uniformity(img_binary, img_channels)
			metrics[channel_names[channel]][
				"NumberOfCellsPer100SquareMicrons"
			] = cell_num_normalized
			metrics[channel_names[channel]][
				"FractionOfForegroundOccupiedByCells"
			] = foreground_fraction
			metrics[channel_names[channel]]["1-FractionOfBackgroundOccupiedByCells"] = (
					1 - background_fraction
			)
			metrics[channel_names[channel]][
				"FractionOfCellMaskInForeground"
			] = mask_foreground_fraction
			metrics[channel_names[channel]]["1/(ln(StandardDeviationOfCellSize)+1)"] = 1 / (
					np.log(cell_size_std) + 1
			)
			metrics[channel_names[channel]]["FractionOfMatchedCellsAndNuclei"] = matched_fraction
			metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
					foreground_CV + 1
			)
			metrics[channel_names[channel]][
				"FractionOfFirstPCForegroundOutsideCells"
			] = foreground_PCA

			# get cell type labels
			cell_type_labels = cell_type(current_mask, img_channels)
		else:
			img_channels = np.squeeze(img["data"][0, :, 0, :, :])
			# get cell uniformity
			cell_CV, cell_fraction, cell_silhouette = cell_uniformity(
				current_mask, img_channels, cell_type_labels
			)
			avg_cell_CV = np.average(cell_CV)
			avg_cell_fraction = np.average(cell_fraction)
			avg_cell_silhouette = np.average(cell_silhouette)

			metrics[channel_names[channel]][
				"1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"
			] = 1 / (avg_cell_CV + 1)
			metrics[channel_names[channel]][
				"AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"
			] = avg_cell_fraction
			metrics[channel_names[channel]][
				"AvgSilhouetteOver2~10NumberOfClusters"
			] = avg_cell_silhouette

	metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
	if isinstance(PCA_model, np.ndarray):
		PCAmodel=PCA_model
	else:
		PCAmodel=getPCAmodel("2Dv1.5")
	try:
		quality_score = get_quality_score(metrics_flat, PCAmodel)
	except:
		quality_score = float('nan')
	metrics["QualityScore"] = quality_score

	# return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
	return metrics
