from skimage.io import imread, imsave
from skimage import measure
from skimage import draw
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_mean, threshold_local
from scipy.sparse import csr_matrix
from scipy.spatial import Voronoi
import sys
from os.path import join
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import matlab.engine
import scipy.ndimage
import numpy as np


def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def segmentation(img):
	# fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
	# plt.show()
	# image = scipy.ndimage.gaussian_filter(img,0.5)
	# image = img
	# th = threshold_otsu(image)
	# binary_local = image > th
	# plt.imshow(binary_local)
	# plt.show()
	# block_size = 201
	# local_thresh = threshold_local(image, block_size, offset=50)    # thresh_function = cv2.THRESH_BINARY_INV if param_invert else cv2.THRESH_BINARY
	# binary_local = image > local_thresh
	# plt.imshow(binary_local)
	# plt.show()
	# plt.clf()
	# plt.close()
	# thresh = cv2.adaptiveThreshold(image, 65535, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2)
	# kernel = np.ones((param_dilation,param_dilation), np.uint8)
	# img_dilation = cv2.dilate(thresh, kernel, iterations=2)
	# kernel = np.ones((param_erode,param_erode), np.uint8)
	# img_erode = cv2.erode(img_dilation,kernel, iterations=1)
	# img_open = area_opening(img_erode, area_threshold=param_area)
	image = scipy.ndimage.gaussian_filter(img, 0.05)
	th = threshold_otsu(image)
	image_binary = image > th
	image_indexed = measure.label(image_binary)
	
	return image_indexed

def get_centroid(img):
	coords = get_indices_sparse(img)
	centroids = list(map(lambda x: np.average(x, axis=1), coords))
	return centroids

def get_voronoi(centroids):
	vor = Voronoi(centroids)
	return vor

def voronoi_finite_polygons_2d(vor, radius=None):
	"""
	Reconstruct infinite voronoi regions in a 2D diagram to finite
	regions.

	Parameters
	----------
	vor : Voronoi
		Input diagram
	radius : float, optional
		Distance to 'points at infinity'.

	Returns
	-------
	regions : list of tuples
		Indices of vertices in each revised Voronoi regions.
	vertices : list of tuples
		Coordinates for revised Voronoi vertices. Same as coordinates
		of input vertices, with 'points at infinity' appended to the
		end.

	"""
	
	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")
	
	new_regions = []
	new_vertices = vor.vertices.tolist()
	
	center = vor.points.mean(axis=0)
	if radius is None:
		radius = vor.points.ptp().max()
	
	# Construct a map containing all ridges for a given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))
	
	# Reconstruct infinite regions
	for p1, region in enumerate(vor.point_region):
		vertices = vor.regions[region]
		
		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue
		
		# reconstruct a non-finite region
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]
		
		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				# finite ridge: already in the region
				continue
			
			# Compute the missing endpoint of an infinite ridge
			
			t = vor.points[p2] - vor.points[p1] # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal
			
			midpoint = vor.points[[p1, p2]].mean(axis=0)
			direction = np.sign(np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius
			
			new_region.append(len(new_vertices))
			new_vertices.append(far_point.tolist())
		
		# sort region counterclockwise
		vs = np.asarray([new_vertices[v] for v in new_region])
		c = vs.mean(axis=0)
		angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
		new_region = np.array(new_region)[np.argsort(angles)]
		
		# finish
		new_regions.append(new_region.tolist())
	
	return new_regions, np.asarray(new_vertices)

# def vorarr(regions, vertices, width, height, dpi=100):
# 	fig = plt.Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
# 	canvas = FigureCanvas(fig)
# 	ax = fig.add_axes([0,0,1,1])
#
# 	# colorize
# 	index = 0
# 	for region in regions:
# 		polygon = vertices[region]
# 		ax.fill(*zip(*polygon), alpha=0.4)
# 		index += 1
#
# 	# ax.plot(nuclear_centroids[:,0], nuclear_centroids[:,1], 'ko')
# 	ax.set_xlim(voronoi.min_bound[0] - 0.1, voronoi.max_bound[0] + 0.1)
# 	ax.set_ylim(voronoi.min_bound[1] - 0.1, voronoi.max_bound[1] + 0.1)
# 	canvas.draw()
# 	return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
#

def remove_non_nuclei(img):
	coords = get_indices_sparse(img)[1:]
	for i in range(len(coords)):
		if len(coords[i][0]) < 30 or len(coords[i][0]) > 500:
			img[coords[i]] = 0
	return img
	
if __name__ == '__main__':
	# image = imread('/data/HuBMAP/CODEX/HBM696.DSTM.687/R001_X004_Y002/nucleus.tif')
	file_dir = sys.argv[1]
	image = imread(join(file_dir, 'nucleus.tif'))
	nuclear_mask_indexed_image = segmentation(image)
	print(len(np.unique(nuclear_mask_indexed_image)))
	nuclear_mask_indexed_image_update = measure.label(remove_non_nuclei(nuclear_mask_indexed_image.copy()))
	# nuclear_mask_indexed_image_update = measure.label(nuclear_mask_indexed_image.copy())
	# nuclear_mask_indexed_image_update = nuclear_mask_indexed_image
	nuclear_centroids = get_centroid(nuclear_mask_indexed_image_update)
	voronoi = get_voronoi(nuclear_centroids)
	regions, vertices = voronoi_finite_polygons_2d(voronoi)
	# regions = voronoi.regions
	# vertices = voronoi.vertices
	cell_mask_indexed_image = np.zeros(image.shape)
	index = 1
	for region in regions:
		if len(region) != 0:
			polygon = vertices[region].astype(int).T
			polygon_area = draw.polygon(polygon[0].clip(min=0, max=image.shape[0]), polygon[1].clip(min=0, max=image.shape[1]))
			cell_mask_indexed_image[polygon_area] = index
			index += 1
	
	cell_mask_indexed_image = cell_mask_indexed_image.astype(int)
	# cell_mask_indexed_image[np.where(cell_mask_indexed_image == 0)] = np.max(cell_mask_indexed_image) + 1
	# plt.imshow(cell_mask_indexed_image)
	# plt.show()
	import bz2
	import pickle
	mask_dir = bz2.BZ2File(join(file_dir, 'mask_Voronoi.pickle'), 'wb')
	pickle.dump(cell_mask_indexed_image, mask_dir)
	imsave(join(file_dir, 'mask_Voronoi.png'), cell_mask_indexed_image)
	# plt.imshow(nuclear_mask_indexed_image_update)
	print(len(np.unique(nuclear_mask_indexed_image_update)))
	nuclear_dir = bz2.BZ2File(join(file_dir, 'nuclear_mask_Voronoi.pickle'), 'wb')
	# print(cell_mask_indexed_image[490:510,490:510])
	pickle.dump(nuclear_mask_indexed_image_update, nuclear_dir)
	imsave(join(file_dir, 'nuclear_mask_Voronoi.png'), nuclear_mask_indexed_image_update)
	# plt.show()
	