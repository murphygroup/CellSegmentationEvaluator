import numpy as np
import os
from os.path import join
from sklearn.metrics import jaccard_score as JI
from scipy.integrate import simps
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import bz2
import pickle
import glob

def DC(im1, im2):
	im1 = np.asarray(im1).astype(np.bool)
	im2 = np.asarray(im2).astype(np.bool)
	
	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
	
	# Compute Dice coefficient
	intersection = np.logical_and(im1, im2)
	
	return 2. * intersection.sum() / (im1.sum() + im2.sum())

def search_overlap(ref_im, que_im, index):
	ref_cell_loc = np.argwhere(ref_im == index)
	que_cell_index = []
	for i in range(ref_cell_loc.shape[0]):
		que_cell_index.append(que_im[ref_cell_loc[i][0], ref_cell_loc[i][1]])
	que_cell_index = np.unique(que_cell_index)
	que_cell_index = np.delete(que_cell_index, np.where(que_cell_index == 0))
	if len(que_cell_index) == 0:
		return 0, 0, 0
	else:
		jaccard = []
		overlap = []
		fraction = []
		dice = []
		for i in range(len(que_cell_index)):
			que_cell_loc = np.argwhere(que_im == que_cell_index[i])
			locs = np.vstack((ref_cell_loc, que_cell_loc))
			ref_cell_binary = []
			que_cell_binary = []
			cell_size = 0
			for loc in range(locs.shape[0]):
				ref_boolean = ref_im[locs[loc][0], locs[loc][1]] == index
				que_boolean = que_im[locs[loc][0], locs[loc][1]] == que_cell_index[i]
				if ref_boolean:
					ref_cell_binary.append(1)
				else:
					ref_cell_binary.append(0)
				if que_boolean:
					que_cell_binary.append(1)
				else:
					que_cell_binary.append(0)
				if ref_boolean and que_boolean:
					cell_size += 1
			fraction.append(cell_size * 0.5 / max(len(ref_cell_loc), len(que_cell_loc)))
			overlap.append((index, que_cell_index[i]))
			jaccard.append(JI(ref_cell_binary, que_cell_binary))
			dice.append(DC(ref_cell_binary, que_cell_binary))
		if len(overlap) != 0:
			return overlap, fraction, jaccard, dice
		else:
			return 0, 0, 0



def hausdorff_distance(image0, image1):
	a_points = np.transpose(np.nonzero(image0))
	b_points = np.transpose(np.nonzero(image1))
	
	# Handle empty sets properly:
	# - if both sets are empty, return zero
	# - if only one set is empty, return infinity
	if len(a_points) == 0:
		return 0 if len(b_points) == 0 else np.inf
	elif len(b_points) == 0:
		return np.inf
	
	return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
	           max(cKDTree(b_points).query(a_points, k=1)[0]))

def HD(ref_cell_loc, que_cell_loc, img_ref, img_que):
	# ref_cell_loc = np.argwhere(img_ref == ind1)
	# que_cell_loc = np.argwhere(img_que == ind2)
	# print(ref_cell_loc)
	x_min = min(min(ref_cell_loc[:, 0]), min(que_cell_loc[:, 0]))
	x_max = max(max(ref_cell_loc[:, 0]), max(que_cell_loc[:, 0]))
	y_min = min(min(ref_cell_loc[:, 1]), min(que_cell_loc[:, 1]))
	y_max = max(max(ref_cell_loc[:, 1]), max(que_cell_loc[:, 1]))
	a = img_ref[x_min:x_max, y_min:y_max]
	b = img_que[x_min:x_max, y_min:y_max]
	return hausdorff_distance(a, b)

def get_error_matrix(cell_list, img_ref, img_que, coord1, coord2):
	TP = 0
	FP = 0
	FN = 0
	ind_2d = np.zeros((img_ref.shape[0], img_ref.shape[1]), dtype=int)
	for i in range(len(cell_list)):
		# ref_cell_loc = np.argwhere(img_ref == cell_list[i][0])
		# que_cell_loc = np.argwhere(img_que == cell_list[i][1])
		# que_cell_loc = np.argwhere(img_ref == cell_list[i][0])
		# ref_cell_loc = np.argwhere(img_que == cell_list[i][1])
		ref_cell_loc = coord1[cell_list[i, 0]-1]
		que_cell_loc = coord2[cell_list[i, 1]-1]
		for j in range(len(ref_cell_loc)):
			if ref_cell_loc[j].tolist() in que_cell_loc.tolist():
				TP += 1
			else:
				FN += 1
			ind_2d[ref_cell_loc[j][0], ref_cell_loc[j][1]] = 1
		for j in range(len(que_cell_loc)):
			if ind_2d[que_cell_loc[j][0], que_cell_loc[j][1]] == 0:
				FP += 1
				ind_2d[que_cell_loc[j][0], que_cell_loc[j][1]] = 1
	TN = ind_2d.shape[0] * ind_2d.shape[1] - np.sum(ind_2d)
	return TP, FP, TN, FN

def get_mask(cell_list, mask_shape):
	mask = np.zeros((mask_shape))
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask

def get_error_matrix2(img_ref, coord1, coord2):
	new_mask1 = np.sign(get_mask(coord1, img_ref.shape))
	new_mask2 = np.sign(get_mask(coord2, img_ref.shape))
	TP = np.sum((new_mask1 + new_mask2 == 2) * 1)
	FP = np.sum((new_mask1 - new_mask2 == 1) * 1)
	FN = np.sum((new_mask2 - new_mask1 == 1) * 1)
	TN = np.sum((new_mask1 + new_mask2 == 0) * 1)
	
	return TP, FP, TN, FN

def get_mutual_info_var_of_info(TP, FP, TN, FN):
	n = TP + FP + TN + FN
	p_Sg1 = (TP + FN) / n
	p_Sg2 = (TN + FN) / n
	p_St1 = (TP + FP) / n
	p_St2 = (TN + FP) / n
	p_S11 = TP / n
	p_S21 = FN / n
	p_S12 = FP / n
	p_S22 = TN / n
	H_Sg = -(p_Sg1 * np.log(p_Sg1) + p_Sg2 * np.log(p_Sg2))
	H_St = -(p_St1 * np.log(p_St1) + p_St2 * np.log(p_St2))
	H_Sgt = - p_S11 * np.log(p_S11) - p_S12 * np.log(p_S12) \
	        - p_S21 * np.log(p_S21) - p_S22 * np.log(p_S22)
	MI = H_Sg + H_St - H_Sgt
	VOI = H_Sg + H_St - 2 * MI
	# print((H_Sg, H_St, H_Sgt, MI, VOI))
	return MI, VOI

def get_cohen_kappa(TP, FP, TN, FN):
	n = TP + FP + TN + FN
	fa = TP + TN
	fc = ((TN + FN) * (TN + FP) + (FP + TP) * (FN + TP)) / n
	return (fa - fc) / (n - fc)

def auc(TP, FP, TN, FN):
	return 1 - 0.5 * (FP / (FP + TN) + FN / (FN + TP))


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


def get_mask(cell_list, mask_shape):
	mask = np.zeros((mask_shape))
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask


def get_intersection(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	overlap = list(a & b)
	if len(overlap) != 0:
		return len(overlap)
	else:
		return False

def get_union(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	union = list(a | b)
	if len(union) != 0:
		return len(union)
	else:
		return False

def get_matched_list(coord1, coord2, coord2_mask):
	matched_list = np.empty((0, 2), int)
	overlap_list = []
	union_list = []
	ref_area_cell = []
	que_area_cell = []
	for coord1_idx in range(len(coord1)):
		current_coord1 = coord1[coord1_idx]
		coord2_search_num = np.unique(list(map(lambda x: coord2_mask[tuple(x)], current_coord1)))
		overlap_pixel_max = 0
		coord2_idx_max = 0
		for coord2_idx in coord2_search_num:
			if coord2_idx != 0:
				intersection = get_intersection(coord1[coord1_idx], coord2[coord2_idx-1])
				if type(intersection) != bool:
					overlap_portion = intersection
					if overlap_portion > overlap_pixel_max:
						overlap_pixel_max = overlap_portion
						coord2_idx_max = coord2_idx-1
		if coord2_idx_max != 0:
			matched_list = np.vstack((matched_list, (coord1_idx, coord2_idx_max)))
			overlap_list.append(overlap_pixel_max)
			union_list.append(get_union(coord1[coord1_idx], coord2[coord2_idx_max]))
			ref_area_cell.append(len(coord1[coord1_idx]))
			que_area_cell.append(len(coord2[coord2_idx_max]))
	return matched_list, np.array(overlap_list), np.array(union_list), np.array(ref_area_cell), np.array(que_area_cell)

def get_matched_cells_JI_DC_TAUC(img_ref, img_que, ref_cell_coords, que_cell_coords):
	ref_que_matched_list, ref_que_overlap_list, ref_que_union_list, ref_cell_list, que_cell_list = get_matched_list(ref_cell_coords, que_cell_coords, img_que)
	que_ref_matched_list, _, _, _, _ = get_matched_list(que_cell_coords, ref_cell_coords, img_ref)
	temp = que_ref_matched_list[:, 0].copy()
	que_ref_matched_list[:, 0] = que_ref_matched_list[:, 1].copy()
	que_ref_matched_list[:, 1] = temp
	
	matched_index = np.where((ref_que_matched_list == que_ref_matched_list[:, None]).all(-1))[1]
	matched_list_final_all = ref_que_matched_list[matched_index]
	ref_que_overlap_list_final_all = ref_que_overlap_list[matched_index]
	ref_que_union_list_final_all = ref_que_union_list[matched_index]
	ref_cell_list_final_all = ref_cell_list[matched_index]
	que_cell_list_final_all = que_cell_list[matched_index]
	ref_que_max_cell_list_final_all = np.max(np.vstack((ref_cell_list_final_all, que_cell_list_final_all)), axis=0)
	
	overlapped_range = np.arange(0,1.01,0.01)
	overlapped_normalized = []
	
	for threshold in overlapped_range:
		ref_que_overlap_list_current_index = ref_que_overlap_list_final_all > (ref_que_max_cell_list_final_all * threshold)
		ref_que_overlap_list_current = ref_que_overlap_list_final_all[ref_que_overlap_list_current_index]
		overlapped_normalized.append(len(ref_que_overlap_list_current)
		                             / len(ref_que_overlap_list_final_all))
	
	area = simps(overlapped_normalized, dx=0.01)
	
	ref_que_overlap_list_threshold_index = ref_que_overlap_list_final_all > (ref_que_max_cell_list_final_all * 0.5)
	ref_que_overlap_list_final_threshold = ref_que_overlap_list_final_all[ref_que_overlap_list_threshold_index]
	ref_que_union_list_final_threshold = ref_que_union_list_final_all[ref_que_overlap_list_threshold_index]
	ref_cell_list_final_threshold = ref_cell_list_final_all[ref_que_overlap_list_threshold_index]
	que_cell_list_final_threshold = que_cell_list_final_all[ref_que_overlap_list_threshold_index]
	ji_score_avg = np.average(ref_que_overlap_list_final_threshold / ref_que_union_list_final_threshold)
	dice_score_avg = np.average(2 * ref_que_overlap_list_final_threshold / (ref_cell_list_final_threshold + que_cell_list_final_threshold))
	matched_list_final_threshold = matched_list_final_all[ref_que_overlap_list_threshold_index]
	return matched_list_final_threshold+1, ji_score_avg, dice_score_avg, area

def get_hausdorff_dist(overlapped_cell, m1, m2, coord1, coord2):
	# calculate Hausdorff Distance between overlapped cells
	hausdorff = []
	for i in range(len(overlapped_cell)):
		hausdorff.append(HD(coord1[overlapped_cell[i][0]-1], coord2[overlapped_cell[i][1]-1], m1, m2))
	hausdorff_avg = np.average(hausdorff)
	return hausdorff_avg

def get_bi_consist_error(img_ref, img_que, coord1, coord2):
	
	# BCE
	A1 = list(map(lambda x:len(x), coord1))
	A2 = list(map(lambda x:len(x), coord2))
	A1.insert(0, np.argwhere(img_ref == 0).shape[0])
	A2.insert(0, np.argwhere(img_que == 0).shape[0])
	
	E1 = []
	E2 = []
	pixel_num = 0
	for i in range(img_que.shape[0]):
		for j in range(img_que.shape[1]):
			R1 = A1[img_ref[i, j]]
			R2 = A2[img_que[i, j]]
			E1.append(abs(R1-R2) / R1)
			E2.append(abs(R2-R1) / R2)
			pixel_num += 1
	BCE = max(np.sum(E1), np.sum(E2)) / pixel_num
	return BCE

def get_volume_similarity(TP, FP, TN, FN):
	return  1 - abs(FN-FP) / (2*TP + FP + FN)

def pairwise_evaluation(mask_dir, segmentation):
	if segmentation != 1:
		original_masks = glob.glob(join(mask_dir, '**', '*.tif*'), recursive=True)
		method_num = len(original_masks)
		metric_num = 10
		pairwise_metric_matrix_pairwise = np.zeros((metric_num, method_num, method_num))
		for i in range(method_num - 1):
			for j in range(i+1, method_num):
				mask1 = original_masks[i]
				mask2 = original_masks[j]
				mask1_cell_coords = get_indices_sparse(mask1)[1:]
				mask2_cell_coords = get_indices_sparse(mask2)[1:]
				mask1_cell_coords = list(map(lambda x: np.array(x).T, mask1_cell_coords))
				mask2_cell_coords = list(map(lambda x: np.array(x).T, mask2_cell_coords))
				# cell difference, jaccard index, dice coef, tauc
				matched_cell, ji, dc, tauc = get_matched_cells_JI_DC_TAUC(mask1, mask2, mask1_cell_coords, mask2_cell_coords)
				# hausdorff distance
				hd = get_hausdorff_dist(matched_cell, mask1, mask2, mask1_cell_coords, mask2_cell_coords)
				# binary consistency error
				bce = get_bi_consist_error(mask1, mask2, mask1_cell_coords, mask2_cell_coords)
				# error matrix
				TP, FP, TN, FN = get_error_matrix2(mask1, mask1_cell_coords, mask2_cell_coords)
				# mutual information & variation of information
				mi, voi = get_mutual_info_var_of_info(TP, FP, TN, FN)
				# Cohens Kappa
				kap = get_cohen_kappa(TP, FP, TN, FN)
				# Volumetric similarity
				vs = get_volume_similarity(TP, FP, TN, FN)
				num = matched_cell.shape[0]
				metrics = [num, ji, dc, tauc, hd, bce, mi, voi, kap, vs]
				pairwise_metric_matrix_pairwise[:, i, j] = metrics
		
		pickle.dump(pairwise_metric_matrix_pairwise, bz2.BZ2File(join(mask_dir, 'pairwise_metrics_' + os.path.basename(original_masks[i]) + '.json'), 'wb'))
	








