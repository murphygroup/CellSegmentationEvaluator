import json
import os
from os.path import join
from pipeline.data.download_HuBMAP_data import *
from pipeline.segmentation.methods.installation.install_all_methods import *

if __name__ == '__main__':
	config = {}
	package_utilization = input('How would you like to utilize this package?\n'
	              '1. Reproduce all results of the paper.\n'
	              '2. Seek the most suitable segmentation method for one or a batch of multichannel images.\n'
	              '3. Obtain the segmentation quality score on given segmentation masks.\n'
	              'Enter 1 or 2 or 3:\n')
	if package_utilization == '1':
		config['HuBMAP_data'] = 1
		config['evaluation'] = 1
		download = input('Would you like to\n'
		              '1. Download the raw images from HuBMAP portal and run the whole pipeline (only CODEX and IMC, Cell DIVE and MIBI are not yet public)\n'
		              '2. Download the segmentation masks and skip the segmentation step\n'
		              'Enter 1 or 2:\n')
		if download == '1':
			img_path = input('Enter the absolute path you would like to download the images to:\n')
			print('Downloading...')
			download_modality = input('Which imaging modality?\n'
			                          'Enter CODEX or IMC or all\n')
			get_HuBMAP_data(download_modality, img_path + '/HuBMAP_images')
			config['segmentation'] = 1
			config['img_dir'] = img_path + '/HuBMAP_images'
			print('Installing segmentation methods...')
			install_segmentation_methods()
			noise = input('Run image perturbations? y or n')
			if noise == 'y':
				noise_kind = input('Gaussian or downsampling?')
				if noise_kind == 'Gaussian':
					config['noise'] = 'Gaussian'
				elif noise_kind == 'downsampling':
					config['noise'] = 'downsampling'
			elif noise == 'n':
				config['noise'] = None
		elif download == '2':
			mask_path = input('Enter the absolute path you would like to download the segmentation masks to:\n')
			get_segmentation_masks(mask_path)
			config['segmentation'] = 0
			config['mask_dir'] = mask_path + '/HuBMAP_segmentation_masks'
	elif package_utilization == '2':
		config['HuBMAP_data'] = 0
		config['segmentation'] = 1
		config['noise'] = None
		config['input_channel_indices'] = {}
		value = input('Please provide the c-axis index of nuclear channel for segmentation\n')
		config['input_channel_indices']['nucleus'] = value
		value = input('Please provide the c-axis index of cytoplasmic channel for segmentation\n')
		config['input_channel_indices']['cytoplasm'] = value
		value = input('Please provide the c-axis index of cell membrane channel for segmentation\n')
		config['input_channel_indices']['membrane'] = value
		config['evaluation'] = 1
		print('Installing segmentation methods...')
		install_segmentation_methods()
	elif package_utilization == '3':
		config['HuBMAP_data'] = 0
		config['segmentation'] = 0
		config['evaluation'] = 1
		config['noise'] = None
		img_path = input('Enter the absolute path of your image folder (image has to be TIF or OME-TIFF format):\n')
		config['img_dir'] = img_path
		mask_path = input('Enter the absolute path of your segmentation masks folder (masks have to be TIF or OME-TIFF format):\n')
		config['mask_dir'] = mask_path
	with open("config.json", "w") as f:
		json.dump(config, f)
	print('Configuration file generated.')