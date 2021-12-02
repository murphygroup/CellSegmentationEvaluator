import os
import numpy as np
from os.path import join
from .input_channels import get_input_channels
from .noise_maker import *
from .get_cellular_compartments import *
from .get_masks import *

def get_segmentation_masks(input_dir):
	methods = ['DeepCell-0.9.0_cytoplasm', 'DeepCell-0.9.0_membrane', 'Deepcell-0.6.0_cytoplasm', 'DeepCell-0.6.0_membrane', 'Cellpose-0.0.3.1', 'Cellpose-0.6.1', 'Voronoi', 'AICS_classic', 'CellProfiler', 'Cellsegm', 'CellX']
	# methods = ['DeepCell-0.9.0_cytoplasm']
	repair_types = ['nonrepaired', 'repaired']
	for method in methods:
		mask_generating(input_dir, method)
		for repair_type in repair_types:
			mask_repairing(input_dir, method, repair_type)

def segmentation(img_dir, config):
	if img_dir.find('CODEX') == -1:
		if config['HuBMAP_data'] == 0:
			input_channel_index = config['input_channel_indices']
			get_input_channels(img_dir, input_channel_index)
		elif config['HuBMAP_data'] == 1:
			get_input_channels(img_dir)
	return [join(img_dir, 'repaired_matched_mask'), join(img_dir, 'nonrepaired_matched_mask')]



