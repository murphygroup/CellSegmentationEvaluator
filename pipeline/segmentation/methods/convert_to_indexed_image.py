from skimage.io import imread
import sys
import numpy as np
from os.path import join
from skimage.color import rgb2gray
import bz2
import pickle
# from skimage.measure import label
import matplotlib.pyplot as plt
from scipy.ndimage import label

def convert_to_indexed(img):
	unique_cell = np.unique(img)
	n_cell = len(np.unique(img))
	for i in range(n_cell):
		img[np.where(img == unique_cell[i])] = -i
	return -img

if __name__ == '__main__':
	file_dir = sys.argv[1]
	try:
		img = np.load(join(file_dir, 'mask_' + sys.argv[2] + '.npy'))
	except:
		img = imread(join(file_dir, 'mask_' + sys.argv[2] + '.png'), as_gray=True)
	if sys.argv[2] == 'CellX' or 'cellsegm':
		img = label(img)[0]
	
	else:
		img = convert_to_indexed(img)
	save_dir = bz2.BZ2File(join(file_dir, 'mask_' + sys.argv[2] + '.pickle'), 'wb')
	pickle.dump(img.astype('uint16'), save_dir)
