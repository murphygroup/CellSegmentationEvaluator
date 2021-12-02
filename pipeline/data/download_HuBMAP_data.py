import numpy as np
from os.path import join
import os
import pandas as pd
import glob
import cv2 as cv
import json
from skimage.io import imread
import gdown
import sys
import requests

def make_dir(save_dir):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
def laplacian_variance(img):
	return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))


def extend_str_zero(string, target):
	while len(string) != target:
		string = '0' + string
	return string

def get_raw_tif(TMC, tile_idx, part, idx, img_id, tissue_id, save_dir, tile_name, max_var_id=None):
	file_structure_img = file_structure[img_id][tissue_id]
	folder_names = list(file_structure_img.keys())
	cyc = str(idx // 4 + 1)
	if TMC == 'Florida':
		cyc = extend_str_zero(cyc, 3)
		folder_name_partial = 'cyc' + str(cyc) + '_reg001'
		folder_name = [i for i in folder_names if i.startswith(folder_name_partial)][0]
	else:
		folder_name = 'Cyc' + str(cyc) + '_' + 'reg' + str(R)
	
	ch = str(idx % 4)
	if ch == '0':
		ch = '4'
	tile_idx = extend_str_zero(tile_idx, 5)
	result_dir = join(save_dir, tile_name)
	make_dir(result_dir)
	if part == 'nucleus':
		for z in range(30):
			tif_file_name = '1_' + tile_idx + '_Z' + extend_str_zero(str(z), 3) + '_CH' + ch + '.tif'
			try:
				download_path = join('https://d5ebc.a567.data.globus.org', img_id, tissue_id, folder_name, tif_file_name)
				os.system('wget -P ' +  result_dir + ' ' + download_path)
			except:
				pass
			
		slice_list = sorted(glob.glob(join(result_dir, '*CH*.tif')))
		if len(slice_list) > 0:
			lap_vars_per_z_plane = []
			for slice in slice_list:
				try:
					lap_vars_per_z_plane.append(laplacian_variance(imread(slice)))
				except:
					lap_vars_per_z_plane.append(0)
			max_var = max(lap_vars_per_z_plane)
			max_var_id = str(lap_vars_per_z_plane.index(max_var)+1)
			max_var_id = extend_str_zero(max_var_id, 3)
			os.system('mv ' + join(result_dir, '*Z' + max_var_id + '*CH*.tif') + ' ' + join(result_dir, part + '.tif'))
			os.system('rm ' + join(result_dir, '*CH*tif'))
			return max_var_id
		else:
			os.system('rm -rf ' + result_dir)
			return False
	else:
		tif_file_name = '1_' + tile_idx + '_Z' + extend_str_zero(str(max_var_id), 3) + '_CH' + ch + '.tif'
		download_path = join('https://d5ebc.a567.data.globus.org', img_id, tissue_id, folder_name, tif_file_name)
		os.system('wget -P ' + result_dir + ' ' + download_path)
		
		# if TMC == 'Florida':
		# 	os.system('sshpass -p Eric3504039453?? scp hchenc@hive.psc.edu:/hive/hubmap/data/public/' + img_id + '/src*/cyc' + cyc + '*/1_' + tile_idx + '_Z' + max_var_id + '_CH' + ch + '.tif ' + result_dir)
		# else:
		# 	os.system('sshpass -p Eric3504039453?? scp hchenc@hive.psc.edu:/hive/hubmap/data/public/' + img_id + '/*mc1/Cyc' + cyc + '_reg' + str(R) + '/1_' + tile_idx + '_Z' + max_var_id + '_CH' + ch + '.tif ' + result_dir)
		#
		os.system('chmod 777 *tif')
		os.system('mv ' + join(result_dir, '*Z' + max_var_id + '*CH*.tif') + ' ' + join(result_dir, part + '.tif'))


def get_CODEX_data(output_dir):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	checklist = np.loadtxt(join(current_dir, 'CODEX_img_list.txt'), dtype=str)
	global file_structure
	file_structure = json.load(open(join(current_dir, 'CODEX_file_structure.json'), 'r'))
	for dataset in checklist:
		img_name = dataset[0]
		img_id = dataset[1]
		tissue_id = dataset[2]
		R_num = int(dataset[4])
		X_num = int(dataset[5])
		Y_num = int(dataset[6])
		if img_name == 'HBM433.MQRQ.278' or img_name == 'HBM988.SNDW.698':
			TMC = 'Standford'
		else:
			TMC = 'Florida'
		save_dir = join(output_dir, img_name)
		make_dir(save_dir)
		os.system('wget -P ' + save_dir + ' ' + join('https://d5ebc.a567.data.globus.org', img_id, tissue_id, 'channelnames_report.csv'))
		channel_info = pd.read_csv(glob.glob(join(save_dir, '*.csv'))[0], header=None).iloc[:, 0].values.tolist()
		if TMC == 'Florida':
			nucleus_idx = channel_info.index('DAPI-02') + 1
			cytoplasm_idx = channel_info.index('CD107a') + 1
			membrane_idx = channel_info.index('E-CAD') + 1
		else:
			nucleus_idx = channel_info.index('HOECHST1') + 1
			cytoplasm_idx = channel_info.index('cytokeratin') + 1
			membrane_idx = channel_info.index('CD45') + 1
		shared_channels = [channel_info.index('CD11c') + 1, channel_info.index('CD21') + 1, channel_info.index('CD4') + 1, channel_info.index('CD8') + 1, channel_info.index('Ki67') + 1]
		# channel_names = ['CD107a', 'CD11c', 'CD20', 'CD21', 'CD3e', 'CD4', 'CD45RO', 'CD8', 'DAPI-02', 'E-CAD', 'Ki67']
		# shared_channels = []
		# for channel_name in channel_names:
		# 	shared_channels.append(channel_info.index(channel_name) + 1)
		# 	print(channel_info)
		# shared_channels = [channel_info.index('CD11c') + 1, channel_info.index('CD21') + 1, channel_info.index('CD4') + 1, channel_info.index('CD8') + 1, channel_info.index('Ki67') + 1]
		global R
		R = 1
		while R <= R_num:
			Y = 2
			while Y <= Y_num and Y < 10:
				X = 2
				while X <= X_num and X < 10:
					if Y % 2 == 1:
						r_idx = str(X + (Y-1) * X_num)
					else:
						r_idx = str(X_num - (X-1) + (Y-1) * X_num)
					tile_name = 'R00' + str(R) + '_X00' + str(X) + '_Y00' + str(Y)
					if not os.path.exists(join(save_dir, tile_name, 'nucleus.tif')):
						z_slice_id = get_raw_tif(TMC, r_idx, 'nucleus', nucleus_idx, img_id, tissue_id, save_dir, tile_name)
						if z_slice_id:
							get_raw_tif(TMC, r_idx, 'cytoplasm', cytoplasm_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							get_raw_tif(TMC, r_idx, 'membrane', membrane_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							for channel_idx in shared_channels:
								# get_raw_tif(r_idx, 'CH_' + channel_info[channel_idx-1], channel_idx, id, save_dir, tile_name, z_slice_id)
								get_raw_tif(TMC, r_idx, 'CH_' + str(channel_idx), channel_idx, img_id, tissue_id, save_dir, tile_name, z_slice_id)
							channel_dir = join(save_dir, tile_name, 'channels')
							if not os.path.exists(channel_dir):
								os.makedirs(channel_dir)
							os.system('mv ' + join(save_dir, tile_name, 'CH*tif') + ' ' + channel_dir)
					X = X + 2
				Y = Y + 2
			R = R + 1

def get_IMC_data(output_dir):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	checklist = np.loadtxt(join(current_dir, 'IMC_img_list.txt'), dtype=str)
	for dataset in checklist:
		img_name = dataset[0]
		img_id = dataset[1]
		tissue_id = dataset[3]
		tif_file_name = dataset[4]
		save_dir = join(output_dir, img_name)
		make_dir(save_dir)
		os.system('wget -P ' + save_dir + ' ' + join('https://d5ebc.a567.data.globus.org', img_id, 'ometiff', tissue_id, tif_file_name))


def get_HuBMAP_data(modality, output_dir):
	if modality == 'CODEX':
		get_CODEX_data(output_dir + '/CODEX')
	elif modality == 'IMC':
		get_IMC_data(output_dir + '/IMC')
	elif modality == 'all':
		get_CODEX_data(output_dir + '/CODEX')
		get_IMC_data(output_dir + '/IMC')

#
# def download_file_from_google_drive(id, destination):
# 	def get_confirm_token(response):
# 		for key, value in response.cookies.items():
# 			if key.startswith('download_warning'):
# 				return value
#
# 		return None
#
# 	def save_response_content(response, destination):
# 		CHUNK_SIZE = 32768
#
# 		with open(destination, "wb") as f:
# 			for chunk in response.iter_content(CHUNK_SIZE):
# 				if chunk: # filter out keep-alive new chunks
# 					f.write(chunk)
#
# 	URL = "https://docs.google.com/uc?export=download"
#
# 	session = requests.Session()
#
# 	response = session.get(URL, params = { 'id' : id }, stream = True)
# 	token = get_confirm_token(response)
#
# 	if token:
# 		params = { 'id' : id, 'confirm' : token }
# 		response = session.get(URL, params = params, stream = True)
#
# 	save_response_content(response, destination)




def get_segmentation_masks(output_dir):
	file_id = 'https://drive.google.com/u/0/uc?id=1YL0BKkOwoKe7vLU2CcRp1JLXR-lw3hcI&export=download'
	destination = output_dir + '/masks.zip'
	gdown.download(file_id, destination)
	os.system('unzip ' + destination + ' -d ' + output_dir)
	os.system('rm ' + destination)