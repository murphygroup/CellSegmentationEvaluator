from skimage.io import imread
from skimage.io import imsave
from skimage.external.tifffile import TiffFile
import numpy as np
from os.path import join
import os
import argparse
import cv2 as cv
from typing import List
import tifffile
import matplotlib.pyplot as plt

def laplacian_variance(img):
	return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))

def find_best_z_plane_id(img_stack):
	lap_vars_per_z_plane = []
	for z in range(img_stack.shape[0]):
		lap_vars_per_z_plane.append(laplacian_variance(img_stack[z,...]))
	max_var = max(lap_vars_per_z_plane)
	max_var_id = lap_vars_per_z_plane.index(max_var)
	return max_var_id


def get_IMC_channel_names(img_dir):
	metadata = AICSImage(img_dir).metadata
	import xmltodict
	metadata_dict = xmltodict.parse(metadata)
	channel_names = []
	channel_num = len(metadata_dict['ome:OME']['ome:Image']['ome:Pixels']['ome:Channel'])
	for i in range(channel_num):
		channel_names.append(metadata_dict['ome:OME']['ome:Image']['ome:Pixels']['ome:Channel'][i]['@Name'].split('_')[0])
	return channel_names

def get_CODEX_channel_names(img_dir):
	with tifffile.TiffFile(img_dir) as tif:
		tif_tags = {}
		for tag in tif.pages[0].tags.values():
			name, value = tag.name, tag.value
			tif_tags[name] = value
	description = tif_tags['ImageDescription']
	name_list = list()
	for i in range(50):
		channel_num = "Channel:0:" + str(i)
		channel_anchor = description.find(channel_num)
		channel_str = description[channel_anchor:channel_anchor + 80]
		name_anchor = channel_str.find("Name")
		name_str = channel_str[name_anchor+6:name_anchor + 20]
		channel_name = name_str[:name_str.find('"')]
		if len(channel_name) > 0:
			name_list.append(channel_name)
	return name_list
	
def get_input_channels(img_dir, input_channel_idx=None):
	image = imread(img_dir)
	if input_channel_idx is not None:
		imsave(join(img_dir, 'nucleus.tif'), image[input_channel_idx[0]])
		imsave(join(img_dir, 'cytoplasm.tif'), image[input_channel_idx[1]])
		imsave(join(img_dir, 'membrane.tif'), image[input_channel_idx[2]])
	else:
		if img_dir.find('CODEX') != -1:
			modality = 'CODEX'
		elif img_dir.index('IMC') != -1:
			modality = 'IMC'
		if modality == 'CODEX':
			do_shared_channels = True
			if not os.path.exists(join(img_dir, join(img_dir, 'nucleus.tif'))):
				channel_names = get_CODEX_channel_names(join(img_dir))
				if image.shape[0] == 47:
					nucleus = channel_names.index('HOECHST1')
					cytoplasm = channel_names.index('cytokeratin')
					membrane = channel_names.index('CD45')
				else:
					nucleus = channel_names.index('DAPI-02')
					cytoplasm = channel_names.index('CD107a')
					membrane = channel_names.index('E-CAD')
				if do_shared_channels == True:
					shared_channels = [channel_names.index('CD11c'), channel_names.index('CD21'), channel_names.index('CD4'), channel_names.index('CD8'), channel_names.index('Ki67')]
				slice = find_best_z_plane_id(image[nucleus, ...])
				imsave(join(img_dir, 'nucleus.tif'), image[nucleus, slice])
				imsave(join(img_dir, 'cytoplasm.tif'), image[cytoplasm, slice])
				imsave(join(img_dir, 'membrane.tif'), image[membrane, slice])
				if not os.path.exists(join(img_dir, 'channels')):
					os.makedirs(join(img_dir, 'channels'))
				if do_shared_channels == True:
					for j in range(len(shared_channels)):
						imsave(join(img_dir, 'channels',  str(j) + '.tif'), image[shared_channels[j], slice])
				else:
					for j in range(image.shape[0]):
						imsave(join(img_dir, 'channels',  str(j) + '.tif'), image[j, slice])
		elif modality == 'IMC':
			if not os.path.exists(join(img_dir, 'nucleus.tif')):
				channel_names = get_IMC_channel_names(img_dir)
				print(channel_names)
				try:
					nucleus = channel_names.index('191Ir')
				except:
					nucleus = channel_names.index('Histone')
				cytoplasm = channel_names.index('SMA')
				membrane = channel_names.index('HLA-ABC')
				imsave(join(img_dir, 'nucleus.tif'), image[nucleus])
				imsave(join(img_dir, 'cytoplasm.tif'), image[cytoplasm])
				imsave(join(img_dir, 'membrane.tif'), image[membrane])
				if not os.path.exists(join(img_dir, 'channels')):
					os.makedirs(join(img_dir, 'channels'))
				for j in range(image.shape[0]):
					imsave(join(img_dir, 'channels',  str(j) + '.tif'), image[j])
		