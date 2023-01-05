import pickle
import sys
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.stats import variation
from skimage.filters import threshold_mean
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
import bz2
from os.path import join
import glob
import os
import json
from skimage.segmentation import find_boundaries


"""
Package functions that evaluate a single segmentation method
Author: Haoran Chen
"""

def get_matched_fraction(repair_mask, mask, cell_matched_mask, nuclear_mask):

	if repair_mask == 'nonrepaired_matched_mask':
		matched_cell_num = len(np.unique(cell_matched_mask))
		total_cell_num = len(np.unique(mask))
		total_nuclei_num = len(np.unique(nuclear_mask))
		mismatched_cell_num = total_cell_num - matched_cell_num
		mismatched_nuclei_num = total_nuclei_num - matched_cell_num
		fraction_matched_cells = matched_cell_num / (mismatched_cell_num + mismatched_nuclei_num + matched_cell_num)
	else:
		fraction_matched_cells = 1
	return fraction_matched_cells


def get_image_channels(img_dir):
	channel_dirs = glob.glob(join(img_dir, 'channels', '*tif'))
	channel_pieces = []
	for channel_dir in channel_dirs:
		channel_pieces.append(imread(channel_dir))
	channels = np.dstack(channel_pieces)
	channels = np.moveaxis(channels, -1, 0)
	return channels

def thresholding(img):
	threshold = threshold_mean(img.astype(np.int64))
	img_thre = img > threshold
	img_thre = img_thre * 1
	return img_thre


def fraction(img_bi, mask_bi):
	foreground_all = np.sum(img_bi)
	background_all = img_bi.shape[0] * img_bi.shape[1] - foreground_all
	mask_all = np.sum(mask_bi)
	background = len(np.where(mask_bi - img_bi == 1)[0])
	foreground = np.sum(mask_bi * img_bi)
	return foreground / foreground_all, background / background_all, foreground / mask_all


def foreground_separation(img_thre):
	contour_ref = img_thre.copy()
	img_thre = closing(img_thre, disk(1))
	
	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(1))
	img_thre = -img_thre + 1
	
	img_thre = closing(img_thre, disk(10))
	
	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(10))
	img_thre = -img_thre + 1
	
	img_thre = area_closing(img_thre, 5000, connectivity=2)
	contour_ref = contour_ref.astype(float)
	img_thre = img_thre.astype(float)
	img_binary = MorphGAC(
		-contour_ref + 1, 5, -img_thre + 1, smoothing=1, balloon=0.8, threshold=0.5
	)
	img_binary = area_closing(img_binary, 1000, connectivity=2)
	
	return -img_binary + 1


def uniformity_CV(loc, channels):
	CV = []
	n = len(channels)
	for i in range(n):
		channel = channels[i]
		channel = channel / np.mean(channel)
		intensity = channel[tuple(loc.T)]
		CV.append(np.std(intensity))
	return np.average(CV)


def uniformity_fraction(loc, channels) -> float:
	n = len(channels)
	feature_matrix_pieces = []
	for i in range(n):
		channel = channels[i]
		ss = StandardScaler()
		channel_z = ss.fit_transform(channel.copy())
		intensity = channel_z[tuple(loc.T)]
		feature_matrix_pieces.append(intensity)
	feature_matrix = np.vstack(feature_matrix_pieces)
	pca = PCA(n_components=1)
	model = pca.fit(feature_matrix.T)
	fraction = model.explained_variance_ratio_[0]
	return fraction


def foreground_uniformity(img_bi, mask, channels):
	foreground_loc = np.argwhere((img_bi - mask) == 1)
	if len(foreground_loc) == 0:
		CV = 0
		fraction = 0
	else:
		CV = uniformity_CV(foreground_loc, channels)
		fraction = uniformity_fraction(foreground_loc, channels)
	return CV, fraction


def background_uniformity(img_bi, channels):
	background_loc = np.argwhere(img_bi == 0)
	CV = uniformity_CV(background_loc, channels)
	background_pixel_num = background_loc.shape[0]
	background_loc_fraction = 1
	while background_loc_fraction > 0:
		try:
			background_loc_sampled = background_loc[
			                         np.random.randint(
				                         background_pixel_num,
				                         size=round(background_pixel_num * background_loc_fraction),
			                         ),
			                         :,
			                         ]
			fraction = uniformity_fraction(background_loc_sampled, channels)
			break
		except:
			background_loc_fraction = background_loc_fraction / 2
	return CV, fraction


def cell_uniformity_CV(feature_matrix):
	CV = []
	for i in range(feature_matrix.shape[1]):
		if np.sum(feature_matrix[:, i]) == 0:
			CV.append(np.nan)
		else:
			CV.append(variation(feature_matrix[:, i]))
	
	if np.sum(np.nan_to_num(CV)) == 0:
		return 0
	else:
		return np.nanmean(CV)


def cell_uniformity_fraction(feature_matrix):
	if np.sum(feature_matrix) == 0 or feature_matrix.shape[0] == 1:
		return 1
	else:
		pca = PCA(n_components=1)
		model = pca.fit(feature_matrix)
		fraction = model.explained_variance_ratio_[0]
		return fraction


def weighted_by_cluster(vector, labels):
	for i in range(len(vector)):
		vector[i] = vector[i] * len(np.where(labels == i)[0])
	weighted_average = np.sum(vector) / len(labels)
	return weighted_average


def cell_size_uniformity(mask):
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	cell_size = []
	for i in range(cell_coord_num):
		cell_size_current = len(cell_coord[i][0])
		if cell_size_current != 0:
			cell_size.append(cell_size_current)
	cell_size_std = np.std(np.expand_dims(np.array(cell_size), 1))
	return cell_size_std


def cell_type(mask, channels):
	label_list = []
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	print(cell_coord_num)
	ss = StandardScaler()
	feature_matrix_z_pieces = []
	for i in range(n):
		channel = channels[i]
		channel_z = ss.fit_transform(channel)
		cell_intensity_z = []
		for j in range(cell_coord_num):
			cell_size_current = len(cell_coord[j][0])
			if cell_size_current != 0:
				single_cell_intensity_z = (
						np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
				)
				cell_intensity_z.append(single_cell_intensity_z)
		feature_matrix_z_pieces.append(cell_intensity_z)
	
	feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
	for c in range(1, 11):
		model = KMeans(n_clusters=c, random_state=3).fit(feature_matrix_z)
		labels = model.labels_.astype(int)
		label_list.append(labels)
	return label_list


def cell_uniformity(mask, channels, label_list):
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	feature_matrix_pieces = []
	feature_matrix_z_pieces = []
	for i in range(n):
		channel = channels[i]
		channel_z = ss.fit_transform(channel)
		cell_intensity = []
		cell_intensity_z = []
		for j in range(cell_coord_num):
			cell_size_current = len(cell_coord[j][0])
			if cell_size_current != 0:
			# if True:
				single_cell_intensity = np.sum(channel[tuple(cell_coord[j])]) / cell_size_current
				single_cell_intensity_z = (
						np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
				)
				cell_intensity.append(single_cell_intensity)
				cell_intensity_z.append(single_cell_intensity_z)
		feature_matrix_pieces.append(cell_intensity)
		feature_matrix_z_pieces.append(cell_intensity_z)
	
	feature_matrix = np.vstack(feature_matrix_pieces).T
	feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
	CV = []
	fraction = []
	silhouette = []
	
	for c in range(1, 11):
		labels = label_list[c - 1]
		CV_current = []
		fraction_current = []
		if c == 1:
			silhouette.append(1)
		else:
			silhouette.append(silhouette_score(feature_matrix_z, labels))
		for i in range(c):
			cluster_feature_matrix = feature_matrix[np.where(labels == i)[0], :]
			cluster_feature_matrix_z = feature_matrix_z[np.where(labels == i)[0], :]
			CV_current.append(cell_uniformity_CV(cluster_feature_matrix))
			fraction_current.append(cell_uniformity_fraction(cluster_feature_matrix_z))
		CV.append(weighted_by_cluster(CV_current, labels))
		fraction.append(weighted_by_cluster(fraction_current, labels))
	return CV, fraction, silhouette[1:]


def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]


def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary


def flatten_dict(input_dict):
	local_list = []
	for key, value in input_dict.items():
		if type(value) == dict:
			local_list.extend(flatten_dict(value))
		else:
			local_list.append(value)
	return local_list



def get_quality_score(features, model):
	ss = model[0]
	pca = model[1]
	features_scaled = ss.transform(features)
	pc1 = pca.transform(features_scaled)[0, 0]
	pc2 = pca.transform(features_scaled)[0, 1]
	score = (
			pca.transform(features_scaled)[0, 0] * pca.explained_variance_ratio_[0]
			+ pca.transform(features_scaled)[0, 1] * pca.explained_variance_ratio_[1]
	)
	return score, pc1, pc2

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask, mode='inner')
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
	a = set((tuple(i) for i in cell_arr))
	b = set((tuple(i) for i in cell_membrane_arr))
	c = set((tuple(i) for i in nuclear_arr))
	d = a - b
	mismatch_pixel_num = len(list(c - d))
	mismatch_fraction = len(list(c - d)) / len(list(c))
	if not mismatch_repair:
		if mismatch_pixel_num == 0:
			return np.array(list(a)), np.array(list(c)), 0
		else:
			return False, False, False
	else:
		if mismatch_pixel_num < len(c):
			return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
		else:
			return False, False, False

def get_mask(cell_list, mask_shape):
	mask = np.zeros(mask_shape)
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask


def get_matched_masks(cell_mask, nuclear_mask):
	cell_membrane_mask = get_boundary(cell_mask)
	cell_coords = get_indices_sparse(cell_mask)[1:]
	nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
	cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]
	cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
	cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
	nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
	cell_matched_index_list = []
	nucleus_matched_index_list = []
	cell_matched_list = []
	nucleus_matched_list = []
	
	repaired_num = 0
	for i in range(len(cell_coords)):
		if len(cell_coords[i]) != 0:
			current_cell_coords = cell_coords[i]
			nuclear_search_num = np.unique(list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords)))
			best_mismatch_fraction = 1
			whole_cell_best = []
			for j in nuclear_search_num:
				# print(j)
				if j != 0:
					if (j-1 not in nucleus_matched_index_list) and (i not in cell_matched_index_list):
						whole_cell, nucleus, mismatch_fraction = get_matched_cells(cell_coords[i], cell_membrane_coords[i], nucleus_coords[j-1], mismatch_repair=1)
						if type(whole_cell) != bool:
							if mismatch_fraction < best_mismatch_fraction:
								best_mismatch_fraction = mismatch_fraction
								whole_cell_best = whole_cell
								nucleus_best = nucleus
								i_ind = i
								j_ind = j-1
			if best_mismatch_fraction < 1 and best_mismatch_fraction > 0:
				repaired_num += 1
			
			if len(whole_cell_best) > 0:
				cell_matched_list.append(whole_cell_best)
				nucleus_matched_list.append(nucleus_best)
				cell_matched_index_list.append(i_ind)
				nucleus_matched_index_list.append(j_ind)
	cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, nuclear_mask.shape)
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
	return cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask

def get_evaluation_ten_metrics(img_dir, mask_dir):
	print('Calculating single-method metrics...')
	# get compartment masks

	cell_mask = pickle.load(bz2.BZ2File(mask_dir, 'r'))

	metric_mask = np.expand_dims(cell_mask, axis=0)
	nucleus = imread(join(img_dir, 'nucleus.tif'))
	cytoplasm = imread(join(img_dir, 'cytoplasm.tif'))
	membrane = imread(join(img_dir, 'membrane.tif'))
	img = np.dstack((nucleus, cytoplasm, membrane))
	img = np.moveaxis(img, -1, 0)

	
	# separate image foreground background
	if not os.path.exists(join(img_dir, 'img_binary.pickle')):
		img_thresholded = sum(thresholding(img[c]) for c in range(img.shape[0]))

		img_binary = foreground_separation(img_thresholded)
		img_binary = np.sign(img_binary)
		fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode="L").convert("1")
		fg_bg_image.save(join(img_dir, 'img_binary.png'))
		pickle.dump(img_binary, bz2.BZ2File(join(img_dir, 'img_binary.pickle'), 'wb'))
	else:
		img_binary = pickle.load(bz2.BZ2File(join(img_dir, 'img_binary.pickle'), 'rb'))
	
	# set mask channel names
	channel_names = [
		"Matched Cell"
	]
	metrics = {}

	img_channels = get_image_channels(img_dir)

	for channel in range(metric_mask.shape[0]):
		current_mask = metric_mask[channel].astype(int)
		mask_binary = np.sign(current_mask)
		metrics[channel_names[channel]] = {}
		if channel_names[channel] == "Matched Cell":
			
			if img_dir.find('CODEX') != -1:
				pixel_size = 0.37745 ** 2
			elif img_dir.find('IMC') != -1:
				pixel_size = 1
			else:
				pixel_size = 1
			
			pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
			micron_num = pixel_size * pixel_num
			
			# calculate number of cell per 100 squared micron
			cell_num = len(np.unique(current_mask)) - 1
			
			cell_num_normalized = cell_num / micron_num * 100
			
			# calculate the standard deviation of cell size
			cell_size_std = cell_size_uniformity(current_mask)
			
			# get coverage metrics
			foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
				img_binary, mask_binary
			)
			
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
			metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
					foreground_CV + 1
			)
			metrics[channel_names[channel]][
				"FractionOfFirstPCForegroundOutsideCells"
			] = foreground_PCA
			
			# get cell type labels
			cell_type_labels = cell_type(current_mask, img_channels)
		
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
	PCA_model = pickle.load(open(join('pca_10_metrics.pickle'), "rb"))
	
	quality_score, pc1, pc2 = get_quality_score(metrics_flat, PCA_model)
	metrics["QualityScore"] = quality_score
	metrics["PC1"] = pc1
	metrics["PC2"] = pc2
	
	return metrics

def expert_annotation_analysis(tile_name):
	cwd = os.getcwd()
	mask_dir = join(cwd, tile_name)
	method_list = ['Cellpose-0.0.3.1', 'Cellpose-0.6.1', 'CellProfiler', 'Cellsegm', 'CellX', 'DeepCell-0.6.0_membrane',
	               'DeepCell-0.6.0_cytoplasm', 'DeepCell-0.9.0_membrane', 'DeepCell-0.9.0_cytoplasm', 'AICS_classic',
	               'Voronoi', 'expert1', 'expert2']
	for method in method_list:
		cell_matched_masks = join(mask_dir, 'mask_' + method + '.pickle')
		metrics = get_evaluation_ten_metrics(mask_dir, cell_matched_masks)
		# save QC json file
		
		with open(join(mask_dir, 'metrics_' + method + '.json'), 'w') as f:
			json.dump(metrics, f)

	