import json
import math
import time
from bisect import bisect
from collections import defaultdict
from itertools import chain, combinations, product
from os import walk
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from aicsimageio import AICSImage

import matplotlib
import matplotlib.cm
import matplotlib.colors
import numba as nb
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from frozendict import frozendict
import re
import os
INTEGER_PATTERN = re.compile(r"(\d+)")
FILENAMES_TO_IGNORE = frozenset({".DS_Store"})

def make_dir(save_dir):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

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


def try_parse_int(value: str) -> Union[int, str]:
	if value.isdigit():
		return int(value)
	return value

def alphanum_sort_key(path: Path) -> Sequence[Union[int, str]]:
	"""
	By: Matt Ruffalo
	Produces a sort key for file names, alternating strings and integers.
	Always [string, (integer, string)+] in quasi-regex notation.
	>>> alphanum_sort_key(Path('s1 1 t.tiff'))
	['s', 1, ' ', 1, ' t.tiff']
	>>> alphanum_sort_key(Path('0_4_reg001'))
	['', 0, '_', 4, '_reg', 1, '']
	"""
	return [try_parse_int(c) for c in INTEGER_PATTERN.split(path.name)]

def is_number(val: Any) -> Any:
	try:
		val = int(val)
	except ValueError:
		try:
			val = float(val)
		except ValueError:
			val = val
	return val

def check_output_dir(path: Path, options: Dict):
	if Path(path).is_dir():
		if options.get("debug"):
			print("Output directory exists")
	else:
		path.mkdir()
		if options.get("debug"):
			print("Output directory created")
			


def get_version():
	# Import things in this function to keep the package namespace clean
	from pathlib import Path
	
	package_directory = Path(__file__).parent
	with open(package_directory / "version.txt") as f:
		return f.read().strip()


