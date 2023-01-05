import os
import sys
import numpy as np
from os.path import join
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries
import itertools

def get_compartments_diff(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	diff = np.array(list(a - b))
	return diff


def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
	a = set((tuple(i) for i in cell_arr))
	b = set((tuple(i) for i in cell_membrane_arr))
	c = set((tuple(i) for i in nuclear_arr))
	d = a - b
	# mismatch_pixel = list(c - d)
	# match_pixel_num = len(list(d & c))
	mismatch_pixel_num = len(list(c - d))
	# print(mismatch_pixel_num)
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


def append_coord(rlabel_mask, indices, maxvalue):
	masked_imgs_coord = [[[], []] for i in range(maxvalue)]
	for i in range(0, len(rlabel_mask)):
		masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
		masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])
	return masked_imgs_coord

def unravel_indices(labeled_mask, maxvalue):
	rlabel_mask = labeled_mask.reshape(-1)
	indices = np.arange(len(rlabel_mask))
	indices = np.unravel_index(indices, (labeled_mask.shape[0], labeled_mask.shape[1]))
	masked_imgs_coord = append_coord(rlabel_mask, indices, maxvalue)
	masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))
	return masked_imgs_coord

def get_coordinates(mask):
	print("Getting cell coordinates...")
	cell_num = np.unique(mask)
	maxvalue = len(cell_num)
	channel_coords = unravel_indices(mask, maxvalue)
	return channel_coords

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
					  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def show_plt(mask):
	plt.imshow(mask)
	plt.show()
	plt.clf()

def list_remove(c_list, indexes):
	for index in sorted(indexes, reverse=True):
		del c_list[index]
	return c_list

def filter_cells(coords, mask):
	# completely mismatches
	no_cells = []
	for i in range(len(coords)):
		if np.sum(mask[coords[i]]) == 0:
			no_cells.append(i)
	new_coords = list_remove(coords.copy(), no_cells)
	return new_coords



def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask, mode='inner')
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def get_mask(cell_list, mask_shape):
	mask = np.zeros(mask_shape)
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask

def mask_repairing(file_dir, method, repaired):
	# print(repaired)
	if repaired == 'repaired':
		result_dir = join(file_dir, 'repaired_matched_mask')
		mismatch_repair = True
	elif repaired == 'nonrepaired':
		result_dir = join(file_dir, 'nonrepaired_matched_mask')
		mismatch_repair = False
		
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	import bz2
	import pickle
	methods_with_nuc = ['Voronoi', 'DeepCell-0.6.0_membrane', 'DeepCell-0.9.0_membrane', 'DeepCell-0.6.0_cytoplasm', 'DeepCell-0.9.0_cytoplasm', 'Cellpose-0.0.3.1', 'Cellpose-0.6.1']
	if method in methods_with_nuc:
		try:
			nuclear_mask_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_' + method + '.pickle'), 'rb')
			nuclear_mask = pickle.load(nuclear_mask_dir)
		except:
			nuclear_mask = np.load(join(file_dir, 'nuclear_mask_' + method + '.npy'))
			nuclear_mask_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_' + method + '.pickle'), 'wb')
			pickle.dump(nuclear_mask, nuclear_mask_dir)
			os.system('rm ' + join(file_dir, 'nuclear_mask_' + method + '.npy'))
	else:
		nuclear_mask_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_artificial.pickle'), 'rb')
		nuclear_mask = pickle.load(nuclear_mask_dir)
	
	try:
		whole_cell_mask_dir = bz2.BZ2File(join(file_dir, 'mask_' + method + '.pickle'), 'rb')
		whole_cell_mask = pickle.load(whole_cell_mask_dir)
	except:
		whole_cell_mask = np.load(join(file_dir, 'mask_' + method + '.npy'))
		whole_cell_mask_dir = bz2.BZ2File(join(file_dir, 'mask_' + method + '.pickle'), 'wb')
		pickle.dump(whole_cell_mask, whole_cell_mask_dir)
		os.system('rm ' + join(file_dir, 'mask_' + method + '.npy'))




	cell_membrane_mask = get_boundary(whole_cell_mask)
	
	cell_coords = get_indices_sparse(whole_cell_mask)[1:]
	nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
	cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]

	cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
	cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
	nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
	
	cell_matched_index_list = []
	nucleus_matched_index_list = []
	cell_matched_list = []
	nucleus_matched_list = []
	

		
	for i in range(len(cell_coords)):
		if len(cell_coords[i]) != 0:
			current_cell_coords = cell_coords[i]
			nuclear_search_num = np.unique(list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords)))
			best_mismatch_fraction = 1
			whole_cell_best = []
			for j in nuclear_search_num:
				if j != 0:
					if (j-1 not in nucleus_matched_index_list) and (i not in cell_matched_index_list):
						whole_cell, nucleus, mismatch_fraction = get_matched_cells(cell_coords[i], cell_membrane_coords[i], nucleus_coords[j-1], mismatch_repair=mismatch_repair)
						if type(whole_cell) != bool:
							if mismatch_fraction < best_mismatch_fraction:
								best_mismatch_fraction = mismatch_fraction
								whole_cell_best = whole_cell
								nucleus_best = nucleus
								i_ind = i
								j_ind = j-1
			if len(whole_cell_best) > 0:
				cell_matched_list.append(whole_cell_best)
				nucleus_matched_list.append(nucleus_best)
				cell_matched_index_list.append(i_ind)
				nucleus_matched_index_list.append(j_ind)

		
	

	
	cell_matched_mask = get_mask(cell_matched_list, whole_cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, whole_cell_mask.shape)
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask

	
	import bz2
	import pickle
	dir = bz2.BZ2File(join(result_dir, 'cell_matched_mask_' + method + '.pickle'), 'wb')
	pickle.dump(cell_matched_mask, dir)
	dir = bz2.BZ2File(join(result_dir, 'nuclear_matched_mask_' + method + '.pickle'), 'wb')
	pickle.dump(nuclear_matched_mask, dir)
	dir = bz2.BZ2File(join(result_dir, 'cell_outside_nucleus_matched_mask_' + method + '.pickle'), 'wb')
	pickle.dump(cell_outside_nucleus_mask, dir)


	imsave(join(result_dir, 'cell_matched_mask_' + method + '.png'), cell_matched_mask)
	imsave(join(result_dir, 'nuclear_matched_mask_' + method + '.png'), nuclear_matched_mask)
	imsave(join(result_dir, 'cell_outside_nucleus_matched_mask_' + method + '.png'), cell_outside_nucleus_mask)

