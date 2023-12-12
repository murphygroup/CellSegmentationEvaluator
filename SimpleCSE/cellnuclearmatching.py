import numpy as np
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries


"""
Package functions for matching cells and nuclei
Author: Haoran Chen and R.F.Murphy
Version 1.4 December 11, 2023 R.F.Murphy
"""

def get_matched_fraction(repair_mask, mask, cell_matched_mask, nuclear_mask):
	if repair_mask == 'repaired_matched_mask':
		fraction_matched_cells = 1
	elif repair_mask == 'nonrepaired_matched_mask':
		#print(mask.shape,cell_matched_mask.shape,nuclear_mask.shape)
		matched_cell_num = len(np.unique(cell_matched_mask))
		total_cell_num = len(np.unique(mask))
		total_nuclei_num = len(np.unique(nuclear_mask))
		mismatched_cell_num = total_cell_num - matched_cell_num
		mismatched_nuclei_num = total_nuclei_num - matched_cell_num
		#print(matched_cell_num, total_cell_num, total_nuclei_num, mismatched_cell_num, mismatched_nuclei_num)
		fraction_matched_cells = matched_cell_num / (mismatched_cell_num + mismatched_nuclei_num + matched_cell_num)
	return fraction_matched_cells

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]


def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary


def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask, mode='inner')
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
	a = set((tuple(i) for i in cell_arr))
	b = set((tuple(i) for i in cell_membrane_arr))
	c = set((tuple(i) for i in nuclear_arr))
	d = a - b
	mismatch_pixel_num = len(list(c - d))
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

def get_mask(cell_list, mask_shape):
	mask = np.zeros(mask_shape)
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num+1
	return mask


def get_matched_masks(cell_mask, nuclear_mask):
	cell_membrane_mask = get_boundary(cell_mask)
	cell_coords = get_indices_sparse(cell_mask)[1:]
	nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
	cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]
	cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
	cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
	nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
	cell_matched_index_list = []
	nucleus_matched_index_list = []
	cell_matched_list = []
	nucleus_matched_list = []
	
	repaired_num = 0
	for i in range(len(cell_coords)):
		if len(cell_coords[i]) != 0:
			current_cell_coords = cell_coords[i]
			nuclear_search_num = np.unique(list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords)))
			best_mismatch_fraction = 1
			whole_cell_best = []
			for j in nuclear_search_num:
				# print(j)
				if j != 0:
					if (j-1 not in nucleus_matched_index_list) and (i not in cell_matched_index_list):
						whole_cell, nucleus, mismatch_fraction = get_matched_cells(cell_coords[i], cell_membrane_coords[i], nucleus_coords[j-1], mismatch_repair=1)
						if type(whole_cell) != bool:
							if mismatch_fraction < best_mismatch_fraction:
								best_mismatch_fraction = mismatch_fraction
								whole_cell_best = whole_cell
								nucleus_best = nucleus
								i_ind = i
								j_ind = j-1
			if best_mismatch_fraction < 1 and best_mismatch_fraction > 0:
				repaired_num += 1
			
			if len(whole_cell_best) > 0:
				cell_matched_list.append(whole_cell_best)
				nucleus_matched_list.append(nucleus_best)
				cell_matched_index_list.append(i_ind)
				nucleus_matched_index_list.append(j_ind)
			else:
				print('Skipped cell#'+str(i))

	if repaired_num>0:
		print(str(repaired_num)+' cells repaired out of '+str(len(cell_coords)))

	cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, nuclear_mask.shape)
	cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
	return cell_matched_mask, nuclear_matched_mask, cell_outside_nucleus_mask
