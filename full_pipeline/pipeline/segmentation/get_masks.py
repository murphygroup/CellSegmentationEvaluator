import os
from os.path import join
def mask_generating(input_dir, method):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	os.system(join(dir_path, 'methods', 'run_' + method + '.sh ' + input_dir))