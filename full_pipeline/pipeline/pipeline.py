import faulthandler
from argparse import ArgumentParser
from subprocess import CalledProcessError, check_output
from typing import Optional
from pathlib import Path
import json
import glob
from .utils import *
from .segmentation.run_segmentation import *
from .evaluation.run_evaluation import *
from .data.download_HuBMAP_data import *
from .evaluation.run_pairwise_evaluation import *

def main(config_dir):
	
	# Read configuration file
	with open(config_dir, 'r') as f:
		config = json.load(f)
	if config['segmentation'] == 1:
		img_dirs = glob.glob(join(config['img_dir'], '**', '*.tif*'), recursive=True)
		img_dirs = np.unique([os.path.dirname(x) for x in img_dirs])
		if config['HuBMAP_data'] == 1:
			img_dirs = [x for x in img_dirs if x.find('channels') == -1]
	else:
		img_dirs = [config['img_dir']]

	# Main pipeline
	for img_dir in img_dirs:
		seg_dir_list = [img_dir]
		if config['noise'] != None:
			if img_dir.find('CODEX') != -1:
				modality = 'CODEX'
			elif img_dir.index('IMC') != -1:
				modality = 'IMC'
			# image perturbation
			if config['noise'] == 'Gaussian':
				if modality == 'CODEX':
					gaussian_noise_interval = 500
				elif modality == 'IMC':
					modality = 5
				perturbed_img_num = 3
				get_gaussian_perturbed_image(img_dir, gaussian_noise_interval, perturbed_img_num)
				for i in range(perturbed_img_num+1):
					seg_dir_list.append(join(img_dir, 'random_gaussian_' + str(i)))
			elif config['noise'] == 'downsampling':
				get_downsampled_image(img_dir)
				for i in [30, 50, 70]:
					seg_dir_list.append(join(img_dir, 'downsampling' + str(i)))
		
		# segmentation
		if config['segmentation'] == 0 and config['evaluation'] == 1:
			evaluation(img_dir, config['mask_dir'], config['segmentation'])
		elif config['segmentation'] == 1:
			for seg_dir in seg_dir_list:
				mask_dirs = segmentation(seg_dir, config)
				for mask_dir in mask_dirs:
					# evaluation
					if config['evaluation'] == 1:
						evaluation(seg_dir, mask_dir, config['segmentation'])
						pairwise_evaluation(mask_dir, config['segmentation'])

		
def argparse_wrapper():
	# arguments
	p = ArgumentParser()
	p.add_argument('config_dir', type=Path)
	args = p.parse_args()
	main(args.config_dir)