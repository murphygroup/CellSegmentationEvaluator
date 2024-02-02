import re
import xml.etree.ElementTree as ET
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import json
from aicsimageio import AICSImage
import numpy as np
import xmltodict
import pandas as pd
#from PIL import Image
from pint import Quantity, UnitRegistry
from scipy.sparse import csr_matrix
from scipy.stats import variation
from skimage.filters import threshold_mean, threshold_otsu
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import tifffile
import xml.etree.ElementTree as ET
import warnings
#from cellnuclearmatching import get_matched_masks, get_matched_fraction

warnings.filterwarnings('ignore')

"""
Package functions that evaluate a single cell segmentation mask for a single image
Authors: Haoran Chen and Ted Zhang and Robert F. Murphy
Version: 1.4 December 11, 2023 R.F.Murphy
        repair nuclear masks outside cell masks and mismatched cells and nuclei
         1.5 January 18, 2024 R.F.Murphy
        add CSE3D as simpler function for 3D evaluation
"""

schema_url_pattern = re.compile(r"\{(.+)\}OME")


class NumpyEncoder(json.JSONEncoder):
	"""Custom encoder for numpy data types"""

	def default(self, obj):
		if isinstance(
				obj,
				(
						np.int_,
						np.intc,
						np.intp,
						np.int8,
						np.int16,
						np.int32,
						np.int64,
						np.uint8,
						np.uint16,
						np.uint32,
						np.uint64,
				),
		):

			return int(obj)

		elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
			return float(obj)

		elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
			return {"real": obj.real, "imag": obj.imag}

		elif isinstance(obj, (np.ndarray,)):
			return obj.tolist()

		elif isinstance(obj, (np.bool_)):
			return bool(obj)

		elif isinstance(obj, (np.void)):
			return None

		return json.JSONEncoder.default(self, obj)


def fraction(img_bi, mask_bi):
	foreground_all = np.sum(img_bi)
	if len(img_bi.shape) == 3:
		background_all = img_bi.shape[0] * img_bi.shape[1] * img_bi.shape[2] - foreground_all
	elif len(img_bi.shape) == 2:
		background_all = img_bi.shape[0] * img_bi.shape[1] - foreground_all
	mask_all = np.sum(mask_bi)
	background = len(np.where(mask_bi - img_bi == 1)[0])
	foreground = np.sum(mask_bi * img_bi)
	if background_all == 0:
		background_fraction = 0
	else:
		background_fraction = background / background_all
	foreground_fraction = foreground / foreground_all
	mask_fraction = foreground / mask_all
	# print(foreground_fraction, background_fraction, mask_fraction)
	return foreground_fraction, background_fraction, mask_fraction



def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)


def thresholding(img):
	threshold = threshold_mean(img.astype(np.int64))
	threshold = threshold_mean(img)
	img_thre = img > threshold
	img_thre = img_thre * 1
	#print('in thresholding')
	#print(type(img_thre))
	#print(np.amin(img_thre),np.amax(img_thre))
	return img_thre


def foreground_separation(img_thre,disksizes,areasizes):
	import matplotlib.pyplot as plt
	import uuid
	#print(type(img_thre))
	#print(np.sum(img_thre),np.amin(img_thre),np.amax(img_thre))
	contour_ref = img_thre.copy()
	#fname=str(uuid.uuid4())
	for dsize in disksizes:
		img_thre = closing(img_thre, disk(dsize))
		img_thre = -img_thre + 1
		#print(dsize,np.sum(img_thre),np.amin(img_thre),np.amax(img_thre))
		#plt.figure()
		#plt.imshow(img_thre,vmin=np.amin(img_thre),vmax=np.amax(img_thre))
		#plt.savefig(fname+"d"+str(dsize)+".png")
		#plt.close()

	img_thre = area_closing(img_thre, areasizes[0], connectivity=2)
	#print(np.sum(img_thre),np.amin(img_thre),np.amax(img_thre))
	#plt.figure()
	#plt.imshow(img_thre,vmin=np.amin(img_thre),vmax=np.amax(img_thre))
	#plt.savefig(fname+"c0.png")
	#plt.close()

	contour_ref = contour_ref.astype(float)
	img_thre = img_thre.astype(float)
	img_binary = MorphGAC(
		-contour_ref + 1, 5, -img_thre + 1, smoothing=1, balloon=0.8, threshold=0.5
	)
	#print(np.sum(img_binary),np.amin(img_binary),np.amax(img_binary))
	#plt.figure()
	#plt.imshow(img_binary,vmin=np.amin(img_binary),vmax=np.amax(img_binary))
	#plt.savefig(fname+"b0.png")
	#plt.close()
	img_binary = area_closing(img_binary, areasizes[1], connectivity=2)
	img_binary = -img_binary + 1
	#print(np.sum(img_binary),np.amin(img_binary),np.amax(img_binary))
	#plt.figure()
	#plt.imshow(img_binary,vmin=np.amin(img_binary),vmax=np.amax(img_binary))
	#plt.savefig(fname+"c1.png")
	#plt.close()
	return img_binary


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
	# check for 2D or 3D
	if len(channels.shape) > 3:
		_, z, x, y = channels.shape
		for i in range(n):
			channel = channels[i]
			ss = StandardScaler()
			channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
			intensity = channel_z[tuple(loc.T)]
			feature_matrix_pieces.append(intensity)

	else:
		for i in range(n):
			channel = channels[i]
			ss = StandardScaler()
			channel_z = ss.fit_transform(channel.copy())
			intensity = channel_z[tuple(loc.T)]
			feature_matrix_pieces.append(intensity)

	feature_matrix = np.vstack(feature_matrix_pieces)
	#print(feature_matrix)
	pca = PCA(n_components=1)
	model = pca.fit(feature_matrix.T)
	fraction = model.explained_variance_ratio_[0]
	return fraction


def foreground_uniformity(img_bi, mask, channels):

	foreground_loc = np.argwhere((img_bi - mask) == 1)
	CV = uniformity_CV(foreground_loc, channels)
	fraction = uniformity_fraction(foreground_loc, channels)

	#not clear why this sampling approach is used
	#foreground_pixel_num = foreground_loc.shape[0]
	#foreground_loc_fraction = 1
	#while foreground_loc_fraction > 0:
	#	try:
	#		np.random.seed(3)
	#		foreground_loc_sampled = foreground_loc[
	#		                         np.random.randint(
	#			                         foreground_pixel_num,
	#			                         size=round(foreground_pixel_num * foreground_loc_fraction),
	#		                         ),
	#		                         :,
	#		                         ]
	#		fraction = uniformity_fraction(foreground_loc_sampled, channels)
	#		break
	#	except:
	#		foreground_loc_fraction = foreground_loc_fraction / 2
	#		print(foreground_loc_fraction)
	return CV, fraction


def background_uniformity(img_bi, channels):
	#print("In background_uniformity")

	background_loc = np.argwhere(img_bi == 0)
	CV = uniformity_CV(background_loc, channels)
	fraction = uniformity_fraction(background_loc_sampled, channels)

	#not clear why this sampling approach is used
	#background_pixel_num = background_loc.shape[0]
	#background_loc_fraction = 1
	#while background_loc_fraction > 0:
	#	try:
	#		background_loc_sampled = background_loc[
	#		                         np.random.randint(
	#			                         background_pixel_num,
	#			                         size=round(background_pixel_num * background_loc_fraction),
	#		                         ),
	#		                         :,
	#		                         ]
	#		fraction = uniformity_fraction(background_loc_sampled, channels)
	#		break
	#	except:
	#		background_loc_fraction = background_loc_fraction / 2
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
	cell_coord = get_indices_pandas(mask)[1:]
	# cell_coord_num = len(cell_coord)
	cell_sizes = []
	for i in cell_coord.index:
		cell_size_current = len(cell_coord[i][0])
		if cell_size_current != 0:
			cell_sizes.append(cell_size_current)
	cell_sizes = np.expand_dims(np.array(cell_sizes), 1)
	cell_size_std = np.std(cell_sizes)
	cell_size_mean = np.mean(cell_sizes)
	cell_size_CV = cell_size_std / cell_size_mean
	return cell_size_CV, cell_sizes.T[0].tolist(), cell_size_std


def cell_type(mask, channels):
	n = len(channels)
	#cell_coord = get_indices_sparse(mask)[1:]
	cell_coord = get_indices_pandas(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	feature_matrix_z_pieces = []
	#print('sum(mask)=',sum(mask))

	# check 2D or 3D
	if len(channels.shape) > 3:
		for i in range(n):
			channel = channels[i]
			z, x, y = channel.shape
			channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
			cell_intensity_z = []
			for j in cell_coord.index:
				cell_size_current = len(cell_coord[j][0])
				if cell_size_current != 0:
					single_cell_intensity_z = (
							np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
					)
					cell_intensity_z.append(single_cell_intensity_z)
			#print("i,sum(cell_intensity_z)=",i,sum(cell_intensity_z))
			feature_matrix_z_pieces.append(cell_intensity_z)

	else:
		for i in range(n):
			channel = channels[i]
			channel_z = ss.fit_transform(channel)
			cell_intensity_z = []
			for j in cell_coord.index:
				cell_size_current = len(cell_coord[j][0])
				if cell_size_current != 0:
					single_cell_intensity_z = (
							np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
					)
					cell_intensity_z.append(single_cell_intensity_z)
			feature_matrix_z_pieces.append(cell_intensity_z)

	feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
	#print('Feature_matrix_z.shape=',feature_matrix_z.shape)
	label_list = []
	label_list.append(np.zeros(cell_coord_num,dtype=int))
	for c in range(2, 11):
		model = KMeans(n_clusters=c, random_state=777).fit(feature_matrix_z)
		label_list.append(model.labels_.astype(int))
		#print(c,[sum(model.labels_.astype(int)==k) for k in range(c)])
	return label_list


def cell_uniformity(mask, channels, label_list):
	n = len(channels)
	#cell_coord = get_indices_sparse(mask)[1:]
	cell_coord = get_indices_pandas(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	feature_matrix_pieces = []
	feature_matrix_z_pieces = []

	if len(channels.shape) > 3:
		_, z, x, y = channels.shape
		for i in range(n):
			channel = channels[i]
			channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
			cell_intensity = []
			cell_intensity_z = []
			for j in cell_coord.index:
				cell_size_current = len(cell_coord[j][0])
				if cell_size_current != 0:
					single_cell_intensity = np.sum(channel[tuple(cell_coord[j])]) / cell_size_current
					single_cell_intensity_z = (
							np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
					)
					cell_intensity.append(single_cell_intensity)
					cell_intensity_z.append(single_cell_intensity_z)
			feature_matrix_pieces.append(cell_intensity)
			feature_matrix_z_pieces.append(cell_intensity_z)

	else:
		for i in range(n):
			channel = channels[i]
			channel_z = ss.fit_transform(channel)
			cell_intensity = []
			cell_intensity_z = []
			for j in cell_coord.index:
				cell_size_current = len(cell_coord[j][0])
				if cell_size_current != 0:
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
	#print('In cell_uniformity: sum(feature_matrix_z)=',sum(feature_matrix_z))
	CV = []
	fraction = []
	silhouette = []

	for c in range(1, 11):
		labels = label_list[c - 1]
		#print(sum(labels))
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
		#print(CV)
		fraction.append(weighted_by_cluster(fraction_current, labels))
		#print(fraction)
	return CV, fraction, silhouette #was silhouette[1:]


def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size))


def get_indices_sparse(data):
	data = data.astype(np.uint64)
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)

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


def get_schema_url(ome_xml_root_node: ET.Element) -> str:
	if m == schema_url_pattern.match(ome_xml_root_node.tag):
		return m.group(1)
	raise ValueError(f"Couldn't extract schema URL from tag name {ome_xml_root_node.tag}")


def get_quality_score(features, model):
	ss = model[0]
	pca = model[1]
	features_scaled = ss.transform(features)
	#print(features_scaled)
	ftransformed=np.dot(features_scaled, np.transpose(pca.components_))
	#score = np.exp(
	#		pca.transform(features_scaled)[0, 0] * pca.explained_variance_ratio_[0]
	#		+ pca.transform(features_scaled)[0, 1] * pca.explained_variance_ratio_[1]
	score = np.exp(
			ftransformed[0, 0] * pca.explained_variance_ratio_[0]
			+ ftransformed[0, 1] * pca.explained_variance_ratio_[1]
	)
	return score


def get_physical_dimension_func(
		dimensions: int,
) -> Callable[[AICSImage], Tuple[UnitRegistry, Quantity]]:
	dimension_names = "XYZ"


	def physical_dimension_func(img: AICSImage) -> Tuple[UnitRegistry, Quantity]:
		"""
        Returns area of each pixel (if dimensions == 2) or volume of each
        voxel (if dimensions == 3) as a pint.Quantity. Also returns the
        unit registry associated with these dimensions, with a 'cell' unit
        added to the defaults
        """
		reg = UnitRegistry()
		reg.define("cell = []")

		# aicsimageio parses the OME-XML metadata when loading an image,
		# and uses that metadata to populate various data structures in
		# the AICSImage object. The AICSImage.metadata.to_xml() function
		# constructs a new OME-XML string from that metadata, so anything
		# ignored by aicsimageio won't be present in that XML document.
		# Unfortunately, current aicsimageio ignores physical size units,
		# so we have to parse the original XML ourselves:
		
		# Read OME-TIFF metadata
		#with tifffile.TiffFile(file_path) as tif:
		#	metadata = tif.ome_metadata
		# Parse the XML metadata
		#root = ET.fromstring(metadata)
		#physize = np.zeros(3)
		#phyunit = np.zeros(3)
		#for elem in root.iter():
		#	if 'PhysicalSizeX' in elem.attrib:
		#		physize[0] = elem.get('PhysicalSizeX')
		#	if 'PhysicalSizeY' in elem.attrib:
		#		physize[1] = elem.get('PhysicalSizeY')
		#	if 'PhysicalSizeZ' in elem.attrib:
		#		physize[2] = elem.get('PhysicalSizeZ')
		#	if 'PhysicalSizeXUnit' in elem.attrib:
		#		phyunit[0] = elem.get('PhysicalSizeXUnit')
		#	if 'PhysicalSizeYUnit' in elem.attrib:
		#		phyunit[1] = elem.get('PhysicalSizeYUnit')
		#	if 'PhysicalSizeZUnit' in elem.attrib:
		#		phyunit[2] = elem.get('PhysicalSizeZUnit')

		physize=img.physical_pixel_sizes
		#breakpoint()

		print('Assuming OME pixel sizes are in microns...')
		sizes: List[Quantity] = []
		for idim in range(dimensions):
		#	unit = reg[phyunit[idim]]
			unit = reg["um"]
			sizes.append(physize[idim] * unit)

		size: Quantity = math.prod(sizes)
		return reg, size

	return physical_dimension_func


get_voxel_volume = get_physical_dimension_func(3)
get_pixel_area = get_physical_dimension_func(2)

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
		mask[tuple(cell_list[cell_num].T)] = cell_num+1
	return mask

def get_matched_fraction(repair_mask, mask, cell_matched_mask, nuclear_mask):
	if repair_mask == 'repaired_matched_mask':
		fraction_matched_cells = 1
	elif repair_mask == 'nonrepaired_matched_mask':
		#print(mask.shape,cell_matched_mask.shape,nuclear_mask.shape)
		matched_cell_num = len(np.unique(cell_matched_mask))
		total_cell_num = len(np.unique(mask))
		total_nuclei_num = len(np.unique(nuclear_mask))
		mismatched_cell_num = total_cell_num - matched_cell_num
		mismatched_nuclei_num = total_nuclei_num - matched_cell_num
		#print(matched_cell_num, total_cell_num, total_nuclei_num, mismatched_cell_num, mismatched_nuclei_num)
		fraction_matched_cells = matched_cell_num / (mismatched_cell_num + mismatched_nuclei_num + matched_cell_num)
	return fraction_matched_cells

def get_matched_masks(cell_mask, nuclear_mask):
	cell_membrane_mask = get_boundary(cell_mask)
	#cell_coords = get_indices_sparse(cell_mask)[1:]
	#nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
	cell_coords = get_indices_pandas(cell_mask)[1:]
	nucleus_coords = get_indices_pandas(nuclear_mask)[1:]
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
			else:
				print('Skipped cell#'+str(i))

	if repaired_num>0:
		print(str(repaired_num)+' cells repaired out of '+str(len(cell_coords)))

	cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, nuclear_mask.shape)
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
	return cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask

def getPCAmodel(model_name):
	model=[StandardScaler(),PCA(n_components=2)]
	model0=model[0]
	model1=model[1]
	if model_name == "2Dv1.5":
		model0.mean_=[ 0.24670323,  0.22761458,  0.77619797,  0.81844031,  0.13839755,
			0.52565821,  0.40882963,  0.47843721,  0.6369653 ,  0.48012444,
			  -0.03933201,  0.65830348,  0.47392855,  0.02017161]
		model0.scale_=[0.50052366, 0.29282845, 0.37274236, 0.326852  , 0.06640284,
			0.46528876, 0.23297468, 0.29162715, 0.26499123, 0.25330019,
			 0.45668672, 0.27270701, 0.2439245 , 0.47084118]
		model1.components_=[[ 0.09924839,  0.0969933 ,  0.25437275,  0.31304311,  0.26480891,
			0.14801793,  0.2932504 ,  0.28673252,  0.31817261,  0.29997623,
			0.28617296,  0.31957945,  0.30848302,  0.29561606],
			[ 0.49432047,  0.5929928 , -0.29784797, -0.06378808, -0.09963801,
			0.44910386,  0.02220557, -0.00938788, -0.14312166, -0.09237176,
			  0.15979289, -0.13594118, -0.08102384,  0.13937814]]
		model1.singular_values_=[1069.85908092,  534.61500756]
		model1.explained_variance_ratio_=[0.57297763, 0.14307601]
	elif model_name == "3Dv1.6":
		model0.mean_=[ 5.79344261e-03,  6.91975581e-02,  3.95266833e-01,  3.65272118e-01,
			2.70054340e-01,  4.77099213e+03,  9.63112779e-02,  8.61340231e-02,
			1.84069788e-01,  1.34796667e-01, -5.28896342e-01,  2.09814369e-01,
			  1.58985097e-01, -4.96024885e-01]
		model0.scale_=[1.47099458e-02, 1.49741276e-01, 4.71796152e-01, 4.42309133e-01,
			3.37986699e-01, 3.80289832e+04, 1.25475312e-01, 1.04676499e-01,
			2.20854911e-01, 1.68592374e-01, 5.57651212e-01, 2.50649799e-01,
			 2.00462506e-01, 5.96769403e-01]
		model1.mean_=[ 4.84232867e-16, -5.18574512e-16,  2.88218915e-16,  1.20713232e-15,
			2.15458536e-16, -1.43246996e-16,  3.00105201e-15,  2.79406128e-15,
			8.92569132e-16,  8.59325166e-16, -2.75987645e-16,  4.63220172e-16,
			1.24978496e-15, -2.31484637e-15]
		model1.components_=[[ 0.15646297,  0.18304149,  0.29587865,  0.29445736,  0.28630437,
			0.04908817,  0.27458276,  0.2928372 ,  0.29850688,  0.2865367 ,
			0.30095653,  0.2994107 ,  0.28362345,  0.30060469],
			[ 0.70344937,  0.6319194 , -0.14975031, -0.06015515, -0.06198289,
			0.03562027,  0.10157161, -0.12779778, -0.04083857, -0.11973617,
			  -0.05268667, -0.04197647, -0.15141267, -0.06756526]]
		model1.singular_values_=[88.06313101, 30.37030114]
		model1.explained_variance_ratio_=[0.78239659, 0.0930544 ]
	elif model_name == "3Dculturedcellsv1.6":
		model0.mean_=[1.87680467e-02, 1.65578528e-01, 4.75498775e-01, 5.03442214e-01,
			2.99368925e-01, 6.61534373e+02, 4.38856080e-01, 2.40569948e-01,
			4.69383885e-01, 3.67081756e-01, 1.61904762e-02, 4.74288433e-01,
			 3.70368074e-01, 1.61904762e-02]
		model0.scale_=[2.02220925e-02, 1.85113308e-01, 4.73147281e-01, 4.95468665e-01,
			2.96992111e-01, 7.27900905e+02, 4.32658646e-01, 2.41196403e-01,
			4.62366241e-01, 3.69999461e-01, 9.99868926e-01, 4.66966299e-01,
			 3.73630574e-01, 9.99868926e-01]
		model1.components_=[[ 0.25492073,  0.25048273,  0.26923895,  0.27287267,  0.270074  ,
			0.25091006,  0.27221739,  0.26973058,  0.27227579,  0.26915604,
			0.27298452,  0.27251879,  0.26960217,  0.27298452],
			[ 0.21807862,  0.66549775, -0.2430748 , -0.1368889 , -0.20021327,
			0.53078893, -0.14515681, -0.04521881, -0.16191717, -0.02627936,
			  -0.12894984, -0.15190535,  0.05961473, -0.12894984]]
		model1.singular_values_=[167.26530579,  26.91730773]
		model1.explained_variance_ratio_=[0.95162185, 0.02464427]
	else:
		print("Unrecognized model name")
	model[0]=model0
	model[1]=model1
	return model

def single_method_eval(img, mask, PCA_model, output_dir: Path, bz=0, unit='nanometer', pixelsizex=377.5,
                       pixelsizey=377.5):
	print("Calculating evaluation metrics v1.5 for", img["name"])
	# get best z slice for future use
	bestz = bz

	#print(mask["data"].shape)
	# get compartment masks
	#old code
	#matched_mask = np.squeeze(mask["data"][0, :, bestz, :, :])
	#print(matched_mask.shape)
	#cell_matched_mask = matched_mask[0]
	#nuclear_matched_mask = matched_mask[1]
	#cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
	#print(cell_outside_nucleus_mask.shape)
	#print(len(np.unique(np.ndarray.flatten(cell_outside_nucleus_mask))))
	#new code with corrected matching
	cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask = get_matched_masks(mask["data"][:,0,bestz,:,:],mask["data"][:,1,bestz,:,:])
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
			thresholding(np.squeeze(img["data"][0, c, bestz, :, :]))
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
				matched_fraction = get_matched_fraction('nonrepaired_matched_mask', np.squeeze(mask["data"][:,0,bestz,:,:]), cell_matched_mask, np.squeeze(mask["data"][:,1,bestz,:,:]))
				#print('Matched fraction='+str(matched_fraction))
			try:
				units, pixel_size = get_pixel_area(img["img"])
			except:
				reg = UnitRegistry()
				reg.define("cell = []")
				units = reg(unit)
				sizes = [pixelsizex * units, pixelsizey * units]
				# print(sizes)
				units = reg
				# pixel_size = math.prod(sizes)
				pixel_size = sizes[0] * sizes[1]
			pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
			total_area = pixel_size * pixel_num
			# calculate number of cell per 100 squared micron
			cell_num = units("cell") * len(np.unique(current_mask)) - 1
			cells_per_area = cell_num / total_area
			units.define("hundred_square_micron = micrometer ** 2 * 100")
			cell_num_normalized = cells_per_area.to("cell / hundred_square_micron")
			# calculate the standard deviation of cell size

			_, _, cell_size_std = cell_size_uniformity(current_mask)

			# get coverage metrics
			foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
				img_binary, mask_binary
			)

			img_channels = np.squeeze(img["data"][0, :, bestz, :, :])

			foreground_CV, foreground_PCA = foreground_uniformity(
				img_binary, mask_binary, img_channels
			)
			# background_CV, background_PCA = background_uniformity(img_binary, img_channels)
			metrics[channel_names[channel]][
				"NumberOfCellsPer100SquareMicrons"
			] = cell_num_normalized.magnitude
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
			img_channels = np.squeeze(img["data"][0, :, bestz, :, :])
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
