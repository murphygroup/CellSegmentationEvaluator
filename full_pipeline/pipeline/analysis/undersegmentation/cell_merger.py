import numpy as np
from os.path import join
import pickle
import bz2
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries
import random
import os
import glob
from skimage.morphology import binary_closing, binary_opening, disk

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_neighboring_cells(mask, coord, cell_index):
	x_max = np.max(coord[0])
	x_min = np.min(coord[0])
	y_max = np.max(coord[1])
	y_min = np.min(coord[1])
	cropped_cell_mask = mask[max(x_min-2, 0):min(x_max+3, mask.shape[0]), max(y_min-2, 0):min(y_max+3, mask.shape[1])]
	cropped_cell_mask_single_cell = cropped_cell_mask.copy()
	cropped_cell_mask_single_cell[cropped_cell_mask_single_cell != cell_index] = 0
	mask_boundary = find_boundaries(cropped_cell_mask_single_cell, mode='outer')
	mask_boundary_indexed = np.unique(get_indexed_mask(cropped_cell_mask, mask_boundary))
	mask_boundary_indexed = np.delete(mask_boundary_indexed, np.where(mask_boundary_indexed == 0))
	mask_boundary_indexed = np.delete(mask_boundary_indexed, np.where(mask_boundary_indexed == cell_index))
	return mask_boundary_indexed


def cut_cell(x_coord, y_coord):
	if len(np.unique(x_coord)) > len(np.unique(y_coord)):
		x_cut = np.round(np.average(x_coord)).astype(int)
		cell_1_coord = (x_coord[x_coord < x_cut], y_coord[x_coord < x_cut])
		cell_1_boundary_coord = (x_coord[x_coord == (x_cut-1)], y_coord[x_coord == (x_cut-1)])
		cell_2_coord = (x_coord[x_coord >= x_cut], y_coord[x_coord >= x_cut])
		cell_2_boundary_coord = (x_coord[x_coord == x_cut], y_coord[x_coord == x_cut])
	else:
		y_cut = np.round(np.average(y_coord)).astype(int)
		cell_1_coord = (x_coord[y_coord < y_cut], y_coord[y_coord < y_cut])
		cell_1_boundary_coord = (x_coord[y_coord == (y_cut-1)], y_coord[y_coord == (y_cut-1)])
		cell_2_coord = (x_coord[y_coord >= y_cut], y_coord[y_coord >= y_cut])
		cell_2_boundary_coord = (x_coord[y_coord == y_cut], y_coord[y_coord == y_cut])
		
	return cell_1_coord, cell_2_coord, cell_1_boundary_coord, cell_2_boundary_coord

def get_mask_new_indices(mask):
	new_mask = np.zeros(mask.shape)
	cell_index = 1
	cell_coords_new = get_indices_sparse(mask)[1:]
	for i in range(len(cell_coords_new)):
		if len(cell_coords_new[i][0]) != 0:
			new_mask[cell_coords_new[i]] = cell_index
			cell_index += 1
	return new_mask.astype(np.int64)


def splitter(mask, nucleus, cell_outside_nucleus):
	max_mask_index = np.max(mask)
	cell_coords = get_indices_sparse(mask)[1:]
	for i in random.sample(range(0, len(cell_coords)), 1):
		if len(cell_coords[i][0]) != 0:
			x_coord = cell_coords[i][0]
			y_coord = cell_coords[i][1]
			cut_cell_1_coord, cut_cell_2_coord, boundary_cell_1_coord, boundary_cell_2_coord = cut_cell(x_coord, y_coord)
			if (sum(np.sign(nucleus[cut_cell_2_coord])) != sum(np.sign(nucleus[cell_coords[i]]))) and (sum(np.sign(nucleus[cut_cell_1_coord])) != sum(np.sign(nucleus[cell_coords[i]]))):
				max_mask_index += 1
				mask[cut_cell_2_coord] = max_mask_index
				nucleus[cut_cell_2_coord] = np.sign(nucleus[cut_cell_2_coord]) * max_mask_index
				nucleus[boundary_cell_1_coord] = 0
				nucleus[boundary_cell_2_coord] = 0
				cell_outside_nucleus[cut_cell_2_coord] = np.sign(cell_outside_nucleus[cut_cell_2_coord]) * max_mask_index
				cell_outside_nucleus[boundary_cell_1_coord] = i
				cell_outside_nucleus[boundary_cell_2_coord] = np.sign(cell_outside_nucleus[boundary_cell_2_coord]) * max_mask_index
	return mask, nucleus, cell_outside_nucleus


def merger(mask, nucleus, cell_outside_nucleus, percentage):
	cell_coords = get_indices_sparse(mask)[1:]
	nucleus_coords = get_indices_sparse(nucleus)[1:]
	cell_outside_nucleus_coords = get_indices_sparse(cell_outside_nucleus)[1:]
	merged_cell_list = []
	merged_cell_list_to = []
	for i in range(len(cell_coords)):
		if len(cell_coords[i][0]) != 0 and (i+1) not in merged_cell_list:
			cell_coord = cell_coords[i]
			neighboring_cell_indices = get_neighboring_cells(mask, cell_coord, i+1)
			if len(neighboring_cell_indices) != 0:
				for j in neighboring_cell_indices:
					merge_index = np.random.choice(np.arange(0, 2), p=[1-percentage, percentage])
					if j not in merged_cell_list and merge_index == 1:
						mask[cell_coords[j-1]] = i+1
						nucleus[nucleus_coords[j-1]] = i+1
						cell_outside_nucleus[cell_outside_nucleus_coords[j-1]] = i+1
						merged_cell_list.append(j)
						merged_cell_list.append(i+1)
						merged_cell_list_to.append(i+1)
						break
	return mask, nucleus, cell_outside_nucleus, merged_cell_list_to

def merge_nuclei(cell_mask, nuclear_mask, cell_outside_nucleus_mask, cell_list):
	nucleus_coords_all = get_indices_sparse(nuclear_mask)[1:]
	cell_outside_nucleus_coords_all = get_indices_sparse(cell_outside_nucleus_mask)[1:]
	for cell_index in cell_list:
		if len(nucleus_coords_all[cell_index-1][0]) != 0:
			nucleus_coord = nucleus_coords_all[cell_index-1]
			cell_outside_nucleus_coord = cell_outside_nucleus_coords_all[cell_index-1]
			x_max = np.max(nucleus_coord[0])
			x_min = np.min(nucleus_coord[0])
			y_max = np.max(nucleus_coord[1])
			y_min = np.min(nucleus_coord[1])
			single_nucleus_mask = nuclear_mask[max(x_min-5, 0):min(x_max+6, nuclear_mask.shape[0]), max(y_min-5, 0):min(y_max+6, nuclear_mask.shape[1])]
			single_nucleus_mask_original = single_nucleus_mask.copy()
			single_nucleus_mask[single_nucleus_mask != cell_index] = 0
			single_nucleus_mask_inverted = -np.sign(single_nucleus_mask)+1
			single_nucleus_mask_inverted_opened = binary_opening(single_nucleus_mask_inverted, selem=disk(3)) * 1
			single_nucleus_mask_opened = -single_nucleus_mask_inverted_opened + 1
			
			single_nucleus_mask_final = np.sign(single_nucleus_mask + single_nucleus_mask_opened) * cell_index
			single_nucleus_mask_original[np.where(single_nucleus_mask_final == cell_index)] = cell_index
			nuclear_mask[max(x_min-5, 0):min(x_max+6, nuclear_mask.shape[0]), max(y_min-5, 0):min(y_max+6, nuclear_mask.shape[1])] = single_nucleus_mask_original
			
			single_nucleus_mask = nuclear_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])]
			single_cell_outside_nucleus_mask = cell_outside_nucleus_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])]
			single_cell_outside_nucleus_mask[np.where(single_nucleus_mask != 0)] = 0

			cell_outside_nucleus_mask[max(x_min, 0):min(x_max+1, nuclear_mask.shape[0]), max(y_min, 0):min(y_max+1, nuclear_mask.shape[1])] = single_cell_outside_nucleus_mask

			
			
	return nuclear_mask, cell_outside_nucleus_mask

def merging_cells():
	print('merging cells...')
	np.random.seed(3)
	dataset_list = sorted(glob.glob('/CODEX/HBM**'))
	for dataset in dataset_list:
	
		data_dir = join(dataset, 'R001_X004_Y004', 'random_gaussian_0')
		cell_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_matched_mask_DeepCell-0.9.0-membrane.pickle'), 'r')).astype(np.int64)
		nuclear_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'nuclear_matched_mask_DeepCell-0.9.0-membrane.pickle'), 'r')).astype(np.int64)
		cell_outside_nucleus_mask = pickle.load(bz2.BZ2File(join(data_dir, 'repaired_mask', 'cell_outside_nucleus_matched_mask_DeepCell-0.9.0-membrane.pickle'), 'r')).astype(np.int64)
		for i in np.linspace(0.25,1,4):
			merged_cell_mask, merged_nuclear_mask, merged_cell_outside_nucleus_mask, merged_cell_list = merger(cell_mask.copy(), nuclear_mask.copy(), cell_outside_nucleus_mask.copy(), i)
			merged_nuclear_mask, merged_cell_outside_nucleus_mask = merge_nuclei(merged_cell_mask, merged_nuclear_mask.copy(), merged_cell_outside_nucleus_mask.copy(), merged_cell_list)
			if not os.path.exists(join(data_dir, 'merged_mask')):
				os.makedirs(join(data_dir, 'merged_mask'))
			
			merged_cell_mask = get_mask_new_indices(merged_cell_mask)
			merged_nuclear_mask = get_mask_new_indices(merged_nuclear_mask)
			merged_cell_outside_nucleus_mask = get_mask_new_indices(merged_cell_outside_nucleus_mask)
			
			pickle.dump(merged_cell_mask, bz2.BZ2File(join(data_dir, 'merged_mask', 'cell_matched_mask_DeepCell-0.9.0-membrane_' + str(i) + '.pickle'), 'w'))
			pickle.dump(merged_nuclear_mask, bz2.BZ2File(join(data_dir, 'merged_mask', 'nuclear_matched_mask_DeepCell-0.9.0-membrane_' + str(i) + '.pickle'), 'w'))
			pickle.dump(merged_cell_outside_nucleus_mask, bz2.BZ2File(join(data_dir, 'merged_mask', 'cell_outside_nucleus_matched_mask_DeepCell-0.9.0-membrane_' + str(i) + '.pickle'), 'w'))
		
		
