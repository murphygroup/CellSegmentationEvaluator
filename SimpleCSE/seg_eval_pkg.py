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
from PIL import Image
from pint import Quantity, UnitRegistry
from scipy.sparse import csr_matrix
from scipy.stats import variation
from skimage.filters import threshold_mean, threshold_otsu
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
from cellnuclearmatching import get_matched_masks, get_matched_fraction

warnings.filterwarnings('ignore')

"""
Package functions that evaluate a single cell segmentation mask for a single image
Authors: Haoran Chen and Ted Zhang and Robert F. Murphy
Version: 1.4 December 11, 2023 R.F.Murphy
        repair nuclear masks outside cell masks and mismatched cells and nuclei
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


# def uniformity_fraction(loc, channels) -> float:
#     n = len(channels)
#     feature_matrix_pieces = []
#     for i in range(n):
#         channel = channels[i]
#         ss = StandardScaler()
#         z, x, y = channel.shape
#         channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
#         intensity = channel_z[tuple(loc.T)]
#         feature_matrix_pieces.append(intensity)
#     feature_matrix = np.vstack(feature_matrix_pieces)
#     pca = PCA(n_components=1)
#     model = pca.fit(feature_matrix.T)
#     fraction = model.explained_variance_ratio_[0]
#     return fraction


# def foreground_uniformity(img_bi, mask, channels):
#     foreground_loc = np.argwhere((img_bi - mask) == 1)
#     CV = uniformity_CV(foreground_loc, channels)
#     fraction = uniformity_fraction(foreground_loc, channels)
#     return CV, fraction


# def background_uniformity(img_bi, channels):
#     background_loc = np.argwhere(img_bi == 0)
#     CV = uniformity_CV(background_loc, channels)
#     fraction = uniformity_fraction(background_loc, channels)
#     return CV, fraction

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)


def cell_type(mask, channels):
	label_list = []
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	feature_matrix_z_pieces = []
	for i in range(n):
		channel = channels[i]
		z, x, y = channel.shape
		channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
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
		model = KMeans(n_clusters=c).fit(feature_matrix_z)
		label_list.append(model.labels_.astype(int))
	return label_list


#
# def cell_uniformity(mask, channels, label_list):
#     n = len(channels)
#     cell_coord = get_indices_sparse(mask)[1:]
#     cell_coord_num = len(cell_coord)
#     ss = StandardScaler()
#     feature_matrix_pieces = []
#     feature_matrix_z_pieces = []
#     for i in range(n):
#         channel = channels[i]
#         z, x, y = channel.shape
#         channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
#         cell_intensity = []
#         cell_intensity_z = []
#         for j in range(cell_coord_num):
#             cell_size_current = len(cell_coord[j][0])
#             if cell_size_current != 0:
#                 single_cell_intensity = np.sum(channel[tuple(cell_coord[j])]) / cell_size_current
#                 single_cell_intensity_z = (
#                     np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
#                 )
#                 cell_intensity.append(single_cell_intensity)
#                 cell_intensity_z.append(single_cell_intensity_z)
#         feature_matrix_pieces.append(cell_intensity)
#         feature_matrix_z_pieces.append(cell_intensity_z)
#
#     feature_matrix = np.vstack(feature_matrix_pieces).T
#     feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
#     CV = []
#     fraction = []
#     silhouette = []
#
#     for c in range(1, 11):
#         labels = label_list[c - 1]
#         CV_current = []
#         fraction_current = []
#         if c == 1:
#             silhouette.append(1)
#         else:
#             silhouette.append(silhouette_score(feature_matrix_z, labels))
#         for i in range(c):
#             cluster_feature_matrix = feature_matrix[np.where(labels == i)[0], :]
#             cluster_feature_matrix_z = feature_matrix_z[np.where(labels == i)[0], :]
#             CV_current.append(cell_uniformity_CV(cluster_feature_matrix))
#             fraction_current.append(cell_uniformity_fraction(cluster_feature_matrix_z))
#         CV.append(weighted_by_cluster(CV_current, labels))
#         fraction.append(weighted_by_cluster(fraction_current, labels))
#     return CV, fraction, silhouette[1:]


def thresholding(img):
	threshold = threshold_mean(img.astype(np.int64))
	img_thre = img > threshold
	img_thre = img_thre * 1
	return img_thre


def foreground_separation(img_thre):
	contour_ref = img_thre.copy()
	img_thre = closing(img_thre, disk(1))
	
	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(2))
	img_thre = -img_thre + 1
	
	img_thre = closing(img_thre, disk(20))
	
	img_thre = -img_thre + 1
	img_thre = closing(img_thre, disk(10))
	img_thre = -img_thre + 1
	
	img_thre = area_closing(img_thre, 20000, connectivity=2)
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
	pca = PCA(n_components=1)
	model = pca.fit(feature_matrix.T)
	fraction = model.explained_variance_ratio_[0]
	return fraction


def foreground_uniformity(img_bi, mask, channels):
	# foreground_loc = np.argwhere((img_bi - mask) == 1)
	# CV = uniformity_CV(foreground_loc, channels)
	# fraction = uniformity_fraction(foreground_loc, channels)
	
	# foreground_pixel_num = foreground_loc.shape[0]
	# foreground_loc_fraction = 1
	# while foreground_loc_fraction > 0:
	#    try:
	#        foreground_loc_sampled = foreground_loc[
	#            np.random.randint(
	#                foreground_pixel_num,
	#                size=round(foreground_pixel_num * foreground_loc_fraction),
	#            ),
	#            :,
	#        ]
	#        fraction = uniformity_fraction(foreground_loc_sampled, channels)
	#        break
	#    except:
	#        foreground_loc_fraction = foreground_loc_fraction / 2
	# return CV, fraction
	
	foreground_loc = np.argwhere((img_bi - mask) == 1)
	CV = uniformity_CV(foreground_loc, channels)
	foreground_pixel_num = foreground_loc.shape[0]
	foreground_loc_fraction = 1
	while foreground_loc_fraction > 0:
		try:
			np.random.seed(3)
			foreground_loc_sampled = foreground_loc[
			                         np.random.randint(
				                         foreground_pixel_num,
				                         size=round(foreground_pixel_num * foreground_loc_fraction),
			                         ),
			                         :,
			                         ]
			fraction = uniformity_fraction(foreground_loc_sampled, channels)
			break
		except:
			foreground_loc_fraction = foreground_loc_fraction / 2
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
	label_list = []
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
	cell_coord_num = len(cell_coord)
	ss = StandardScaler()
	feature_matrix_z_pieces = []
	
	# check 2D or 3D
	if len(channels.shape) > 3:
		for i in range(n):
			_, z, x, y = channels.shape
			channel = channels[i]
			channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
			cell_intensity_z = []
			for j in range(cell_coord_num):
				cell_size_current = len(cell_coord[j][0])
				if cell_size_current != 0:
					single_cell_intensity_z = (
							np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
					)
					cell_intensity_z.append(single_cell_intensity_z)
			feature_matrix_z_pieces.append(cell_intensity_z)
	
	else:
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
		model = KMeans(n_clusters=c).fit(feature_matrix_z)
		label_list.append(model.labels_.astype(int))
	return label_list


def cell_uniformity(mask, channels, label_list):
	n = len(channels)
	cell_coord = get_indices_sparse(mask)[1:]
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
			for j in range(cell_coord_num):
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
			for j in range(cell_coord_num):
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
	return csr_matrix((cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size))


def get_indices_sparse(data):
	data = data.astype(np.uint64)
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


def get_schema_url(ome_xml_root_node: ET.Element) -> str:
	if m == schema_url_pattern.match(ome_xml_root_node.tag):
		return m.group(1)
	raise ValueError(f"Couldn't extract schema URL from tag name {ome_xml_root_node.tag}")


def get_quality_score(features, model):
	ss = model[0]
	pca = model[1]
	features_scaled = ss.transform(features)
	score = np.exp(
			pca.transform(features_scaled)[0, 0] * pca.explained_variance_ratio_[0]
			+ pca.transform(features_scaled)[0, 1] * pca.explained_variance_ratio_[1]
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
		# root = ET.fromstring(img.xarray_dask_data.unprocessed[270])
		# schema_url = get_schema_url(root)
		# pixel_node_attrib = root.findall(f".//{{{schema_url}}}Pixels")[0].attrib
		
		sizes: List[Quantity] = []
		for _, dimension in zip(range(dimensions), dimension_names):
			unit = reg[pixel_node_attrib[f"PhysicalSize{dimension}Unit"]]
			# unit = reg["nm"]
			value = float(pixel_node_attrib[f"PhysicalSize{dimension}"])
			sizes.append(value * unit)
		
		size: Quantity = math.prod(sizes)
		return reg, size
	
	return physical_dimension_func


get_voxel_volume = get_physical_dimension_func(3)
get_pixel_area = get_physical_dimension_func(2)


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
	img_binary = foreground_separation(img_thresholded)
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
	try:
		quality_score = get_quality_score(metrics_flat, PCA_model)
	except:
		quality_score = float('nan')
	metrics["QualityScore"] = quality_score
	
	# return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
	return metrics


def single_method_eval_3D(img, mask, PCA_model, output_dir: Path, unit='nm', pixelsizex=1000, pixelsizey=1000,
                          pixelsizez=2000) -> Tuple[Dict[str, Any], float, float]:
	print("Calculating evaluation metrics v1.6 for", img['name'])
	
	# get compartment masks
	matched_mask = mask["data"][0, :, :, :, :]
	cell_matched_mask = matched_mask[0]
	nuclear_matched_mask = matched_mask[1]
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
	
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
	img_thresholded = sum(thresholding(img["data"][0, c, :, :, :]) for c in thresholding_channels)
	if not seg_channel_provided:
		img_thresholded[img_thresholded <= round(img["data"].shape[1] * 0.2)] = 0
	img_binary_pieces = []
	for z in range(img_thresholded.shape[0]):
		img_binary_pieces.append(foreground_separation(img_thresholded[z]))
	img_binary = np.stack(img_binary_pieces, axis=0)
	img_binary = np.sign(img_binary)
	# background_voxel_num = np.argwhere(img_binary == 0).shape[0]
	# fraction_background = background_voxel_num / (
	#         img_binary.shape[0] * img_binary.shape[1] * img_binary.shape[2]
	# )
	
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
		img_channels = img["data"][0, :, :, :, :]
		if channel_names[channel] == "Matched Cell":
			
			try:
				units, voxel_size = get_voxel_volume(img["img"])
			except:
				reg = UnitRegistry()
				reg.define("cell = []")
				units = reg(unit)
				sizes = [pixelsizex * units, pixelsizey * units, pixelsizez * units]
				# print(sizes)
				units = reg
				# voxel_size = math.prod(sizes)
				voxel_size = sizes[0] * sizes[1] * sizes[2]
			
			voxel_num = mask_binary.shape[0] * mask_binary.shape[1] * mask_binary.shape[2]
			total_volume = voxel_size * voxel_num
			
			# TODO: match 3D cell and nuclei and calculate the fraction of match, assume cell and nuclei are matched for now
			
			# calculate number of cell per 100 cubic micron
			cell_num = units("cell") * len(np.unique(current_mask)) - 1
			
			cells_per_volume = cell_num / total_volume
			units.define("hundred_cubic_micron = micrometer ** 3 * 100")
			cell_num_normalized = cells_per_volume.to("cell / hundred_cubic_micron")
			
			# calculate the coefficient of variation of cell sizes
			cell_size_CV, cell_sizes_voxels, cell_size_std = cell_size_uniformity(current_mask)
			# print(cell_size_CV, cell_sizes_voxels, cell_size_std)
			
			# calculate the weighted average of cell sizes
			cell_sizes_microns = [size * voxel_size for size in cell_sizes_voxels]
			weighted_avg_microns = sum(size * size for size in cell_sizes_microns) / sum(cell_sizes_microns)
			weighted_avg_microns = weighted_avg_microns.to("micrometer ** 3")
			
			# get coverage metrics
			foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
				img_binary, mask_binary
			)
			
			foreground_CV, foreground_PCA = foreground_uniformity(
				img_binary, mask_binary, img_channels
			)
			metrics[channel_names[channel]][
				"NumberOfCellsPer100CubicMicrons"
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
			metrics[channel_names[channel]]["1/(CVOfCellSize+1)"] = 1 / (
					cell_size_CV + 1
			)
			
			metrics[channel_names[channel]][
				"WeightedAvgCellSizeinCubicMicrons"
			] = weighted_avg_microns.magnitude
			
			metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
					foreground_CV + 1
			)
			metrics[channel_names[channel]][
				"FractionOfFirstPCForegroundOutsideCells"
			] = foreground_PCA
			
			cell_type_labels = cell_type(current_mask, img_channels)
		else:
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
				"AvgSilhouetteOver1~10NumberOfClusters"
			] = avg_cell_silhouette
	
	# generate quality score
	# print(metrics)
	metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
	try:
		quality_score = get_quality_score(metrics_flat, PCA_model)
	except:
		quality_score = float('nan')
	metrics["QualityScore"] = quality_score
	
	return metrics
