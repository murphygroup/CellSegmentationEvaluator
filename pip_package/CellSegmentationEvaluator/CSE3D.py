#import re
#import xml.etree.ElementTree as ET
#import math
#from pathlib import Path
#from typing import Any, Callable, Dict, List, Tuple
#import json
#from aicsimageio import AICSImage
#import xmltodict
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
import numpy as np
from CellSegmentationEvaluator.functions import cell_size_uniformity, foreground_separation, fraction, foreground_uniformity, cell_type, cell_uniformity, getPCAmodel, get_quality_score, thresholding

import warnings


warnings.filterwarnings('ignore')

"""
Part of CellSegmentationEvaluator package that evaluates a single cell segmentation mask for a single image
Authors: Haoran Chen and Ted Zhang and Robert F. Murphy
Version: 1.4 December 11, 2023 R.F.Murphy
        repair nuclear masks outside cell masks and mismatched cells and nuclei
         1.5 January 18, 2024 R.F.Murphy
        add CSE3D as simpler function for 3D evaluation
"""

def CSE3D(img_channels, metric_mask, PCA_model, img4thresh=0.2, voxel_size=1.0, disksizes=[1,2,10,3],areasizes="none"):

	# img_channels should in in the order CZYX
	# mask channels assumed to be in the order cell, nucleus, cell-nucleus
	# metrics will only be calculated for the first three channels in masks
	# img4thresh is either an ndarray containing the image to threshold or the value at which to threshold the summed image
	# voxel_size is in cubic microns
	# disksizes and areasizes are used for foreground thresholding

	print("CSE3D v1.5")
	print('Voxel size is ',voxel_size,' cubic micrometers')
	print('Using foreground identification disk filters of sizes ',disksizes)

	if areasizes=="none":
		areasizes=(np.round(20000/voxel_size),np.round(1000/voxel_size))
	print('Using foreground identification area filters of sizes ',areasizes)
	#get image to threshold to find foreground
	if isinstance(img4thresh, np.ndarray):
		img_thresholded = img4thresh
	else:
		if not isinstance(img4thresh, float):
			print("Invalid img4thresh; using default sum/scaling")
			img4thresh = 0.2
		# sum all channels
		img_thresholded = sum(thresholding(img_channels[c, :, :, :]) for c in range(img_channels.shape[0]))
		img_thresholded[img_thresholded <= round(img_channels.shape[0] * img4thresh)] = 0

	#print('in CSE3D')
	#print(np.amin(img_thresholded),np.amax(img_thresholded))
	# threshold each slice separately and restack
	img_binary_pieces = []
	for z in range(img_thresholded.shape[0]):
		img_binary_pieces.append(foreground_separation(img_thresholded[z],disksizes,areasizes))
	img_binary = np.stack(img_binary_pieces, axis=0)
	#print(np.amin(img_binary),np.amax(img_binary))
	img_binary = np.sign(img_binary)
	#print(np.sum(img_binary))
	# set mask channel names
	channel_names = [
		"Matched Cell",
		"Nucleus (including nuclear membrane)",
		"Cell Not Including Nucleus (cell membrane plus cytoplasm)",
	]
	#print('sums=',np.sum(img_thresholded),np.sum(img_binary))

	metrics = {}
	# calculate metrics for first three mask channels
	for channel in range(max(metric_mask.shape[0],3)):
		current_mask = metric_mask[channel]
		mask_binary = np.sign(current_mask)
		metrics[channel_names[channel]] = {}
		if channel_names[channel] == "Matched Cell":

			# get number of cells per 100 cubic micron
			voxel_num = mask_binary.shape[0] * mask_binary.shape[1] * mask_binary.shape[2]
			total_volume = voxel_size * voxel_num
			#print('total_volume=',total_volume,' voxel_num=',voxel_num)

			# TODO: match 3D cell and nuclei and calculate the fraction of match, assume cell and nuclei are matched for now

			# calculate number of cells per 100 cubic micron
			cell_num = len(np.unique(current_mask)) - 1
			cells_per_volume = cell_num / total_volume
			cell_num_normalized = cells_per_volume*100.
			#print('cell_num=',cell_num,' normalized=',cell_num_normalized)
			metrics[channel_names[channel]][
				"NumberOfCellsPer100CubicMicrons"
			] = cell_num_normalized

			# calculate the coefficient of variation of cell sizes
			cell_size_CV, cell_sizes_voxels, cell_size_std = cell_size_uniformity(current_mask)
			# print(cell_size_CV, cell_sizes_voxels, cell_size_std)

			# calculate cell size metrics
			cell_sizes_microns = [size * voxel_size for size in cell_sizes_voxels]
			weighted_avg_microns = sum(size * size for size in cell_sizes_microns) / sum(cell_sizes_microns)
			metrics[channel_names[channel]][
				"WeightedAvgCellSizeinCubicMicrons"
			] = weighted_avg_microns
			metrics[channel_names[channel]]["1/(CVOfCellSize+1)"] = 1 / (
					cell_size_CV + 1
			)

			# calculate coverage metrics
			foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
				img_binary, mask_binary
			)
			foreground_CV, foreground_PCA = foreground_uniformity(
				img_binary, mask_binary, img_channels
			)
			metrics[channel_names[channel]][
				"FractionOfForegroundOccupiedByCells"
			] = foreground_fraction
			metrics[channel_names[channel]]["1-FractionOfBackgroundOccupiedByCells"] = (
					1 - background_fraction
			)
			metrics[channel_names[channel]][
				"FractionOfCellMaskInForeground"
			] = mask_foreground_fraction

			metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
					foreground_CV + 1
			)
			metrics[channel_names[channel]][
				"FractionOfFirstPCForegroundOutsideCells"
			] = foreground_PCA

			cell_type_labels = cell_type(current_mask, img_channels)
		else:
			# get uniformity for nucleus and non-nucleus
			cell_CV, cell_fraction, cell_silhouette = cell_uniformity(
				current_mask, img_channels, cell_type_labels
			)
			avg_cell_CV = np.average(cell_CV) #was cell_CV[0]
			avg_cell_fraction = np.average(cell_fraction) #was cell_fraction[0]
			avg_cell_silhouette = np.average(cell_silhouette)
			metrics[channel_names[channel]][
				"1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"
			] = 1 / (avg_cell_CV + 1)
			metrics[channel_names[channel]][
				"AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"
			] = avg_cell_fraction
			metrics[channel_names[channel]][
				"AvgSilhouetteOver1~10NumberOfClusters"
			] = avg_cell_silhouette

	# generate quality score
	#print(metrics)
	#metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
	#print(metrics_flat)
	metrics_in_order = (metrics['Matched Cell']["NumberOfCellsPer100CubicMicrons"],metrics['Matched Cell']["FractionOfForegroundOccupiedByCells"],metrics['Matched Cell']["1-FractionOfBackgroundOccupiedByCells"],metrics['Matched Cell']["FractionOfCellMaskInForeground"],metrics['Matched Cell']["1/(CVOfCellSize+1)"],metrics['Matched Cell']["WeightedAvgCellSizeinCubicMicrons"],metrics['Matched Cell']["1/(AvgCVForegroundOutsideCells+1)"],metrics['Matched Cell']["FractionOfFirstPCForegroundOutsideCells"],metrics['Nucleus (including nuclear membrane)']["1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"],metrics['Nucleus (including nuclear membrane)']["AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"],metrics['Nucleus (including nuclear membrane)']["AvgSilhouetteOver1~10NumberOfClusters"],metrics['Cell Not Including Nucleus (cell membrane plus cytoplasm)']["1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"],metrics['Cell Not Including Nucleus (cell membrane plus cytoplasm)']["AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"],metrics['Cell Not Including Nucleus (cell membrane plus cytoplasm)']["AvgSilhouetteOver1~10NumberOfClusters"])
	#print(metrics_in_order)
	metrics_flat = np.array(metrics_in_order)
	#print(metrics_flat)
	metrics_flat = metrics_flat.reshape(1,-1)
	#print(metrics_flat)
	if isinstance(PCA_model, np.ndarray):
		PCAmodel=PCA_model
	else:
		PCAmodel=getPCAmodel("3Dv1.6")
	try:
		quality_score = get_quality_score(metrics_flat, PCAmodel)
	except:
		quality_score = float('nan')
	metrics["QualityScore"] = quality_score
	metrics["CSEMetricsVersion"] = "v1.6"
	return metrics
