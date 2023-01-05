import os
from os.path import join
def install_segmentation_methods():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	# print(dir_path)
	os.system('bash ' + join(dir_path, 'install_AICS_classic.sh') + ' ' + dir_path)
	os.system('bash ' + join(dir_path, 'install_Cellpose-0.0.3.1.sh') + ' ' + dir_path)
	os.system('bash ' + join(dir_path, 'install_Cellpose-0.6.1.sh') + ' ' + dir_path)
	os.system('bash ' + join(dir_path, 'install_CellProfiler.sh') + ' ' + dir_path)
	os.system('bash ' + join(dir_path, 'install_CellSegm.sh') + ' ' + join(os.path.dirname(dir_path), 'cellsegm'))
	os.system('bash ' + join(dir_path, 'install_CellX.sh') + ' ' + join(os.path.dirname(dir_path), 'CellX'))
	os.system('bash ' + join(dir_path, 'install_DeepCell-0.6.0.sh') + ' ' + dir_path)
	os.system('bash ' + join(dir_path, 'install_DeepCell-0.9.0.sh') + ' ' + dir_path)