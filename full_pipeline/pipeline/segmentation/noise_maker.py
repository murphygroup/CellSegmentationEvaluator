import numpy as np
import os
import cv2
from skimage.io import imread
from skimage.io import imsave
from os.path import join
import sys
import matplotlib.pyplot as plt
import argparse



def add_noise(noise_typ, image, sigma):
	if noise_typ == "gauss":
		row, col, ch = image.shape
		mean = 0
		gauss = np.random.normal(mean, sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row, col, ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		          for i in image.shape]
		out[coords] = 1
		
		# Pepper mode
		num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
		          for i in image.shape]
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ == "speckle":
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy
	
def get_gaussian_perturbed_image(img_dir, noise_num, noise_interval):
	for i in range(noise_num+1):
		gaussian_dir = join(img_dir, 'random_gaussian_' + str(i))
		if not os.path.exists(join(gaussian_dir, 'membrane.tif')):
			try:
				os.makedirs(gaussian_dir)
			except:
				pass
		for img_name in ['nucleus', 'cytoplasm', 'membrane']:
			img = imread(join(img_dir, img_name + '.tif'))
			# w = int(sys.argv[4])
			# h = int(sys.argv[5])
			# center = [img.shape[0] / 2, img.shape[1] / 2]
			# x = center[1] - w/2
			# y = center[0] - h/2
			# img = img[int(y):int(y+h), int(x):int(x+w)]
			img = np.expand_dims(img, axis=-1)
			img_noisy = add_noise('gauss', img, i * noise_interval)
			img_noisy = np.squeeze(img_noisy, axis=-1)
			img_noisy[np.where(img_noisy < 0)] = 0
			img_noisy[np.where(img_noisy > 65535)] = 65535
			img_noisy = img_noisy.astype('uint16')
			imsave(join(gaussian_dir, img_name + '.tif'), img_noisy)
			
def get_downsampled_image(img_dir):
	for i in [30, 50, 70]:
		downsample_dir = join(img_dir, 'downsampling_' + str(i))
		try:
			os.makedirs(downsample_dir)
		except:
			pass
		if not os.path.exists(join(downsample_dir, 'membrane.tif')):
			for img_name in ['nucleus', 'cytoplasm', 'membrane']:
				img = imread(join(img_dir, img_name + '.tif'))
				x = img.shape[0]
				y = img.shape[1]
				img_downsampled = cv2.resize(img, (int(y * i / 100), int(x * i / 100)), interpolation=cv2.INTER_AREA)
				img_downsampled[np.where(img_downsampled < 0)] = 0
				img_downsampled[np.where(img_downsampled > 65535)] = 65535
				img_downsampled = img_downsampled.astype('uint16')
				imsave(join(downsample_dir, img_name + '.tif'), img_downsampled)
		channel_dir = join(img_dir, 'channels_downsampling_' + str(i))
		if (not os.path.exists(channel_dir)) or (len(os.listdir(channel_dir)) == 0):
			os.system('rm -rf ' + channel_dir)
			os.makedirs(channel_dir)
			channels = glob.glob(join(img_dir, "channels", '*.tif'))
			n = len(channels)
			# print(channels)
			for c in range(n):
				channel = imread(channels[c])
				x = channel.shape[0]
				y = channel.shape[1]
				channel_downsampled = cv2.resize(channel, (int(y * i / 100), int(x * i / 100)), interpolation=cv2.INTER_AREA)
				imsave(join(channel_dir, str(c) + '.tif'), channel_downsampled)