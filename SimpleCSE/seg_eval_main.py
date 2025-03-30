from pathlib import Path
import pickle
from read_and_eval_seg import read_and_eval_seg
import argparse

"""
MAIN PROGRAM TO CALCULATE SEGMENTATION EVALUATION STATISTICS FOR A
FOR A SINGLE IMAGE AND MASK
Author: Robert F. Murphy and Haoran Chen and Ted Zhang
"""
    
parser = argparse.ArgumentParser(description='image and mask paths')
parser.add_argument('img_path', type=Path)
parser.add_argument('mask_path', type=Path)
args = parser.parse_args()

img_path = args.img_path
mask_path = args.mask_path

output_directory = Path('results')
if not output_directory.exists():
    output_directory.mkdir()

# can edit this program to specify a PCA model, or let read_and_eval_seg use the default based on whether image is 2D or 3D
#PCA_model = pickle.load(open( "2D_PCA_model.pickle", "rb" ))
                         #or
#PCA_model = pickle.load( open( "3D_PCA_model.pickle", "rb" ))
PCA_model = []

seg_metrics = read_and_eval_seg (img_path, mask_path, PCA_model, output_directory)

print(seg_metrics)
