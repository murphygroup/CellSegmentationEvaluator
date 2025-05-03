from pathlib import Path
#import pickle
from read_and_eval_seg import read_and_eval_seg
import argparse
from os.path import isdir, isfile, join, split
import glob
import pandas as pd

"""
MAIN PROGRAM TO CALCULATE SEGMENTATION EVALUATION STATISTICS FOR A
FOR A SINGLE IMAGE AND MASK
Author: Robert F. Murphy and Haoran Chen and Ted Zhang
Version 1.5.15 R.F.Murphy April 22, 2025
        Allow folder names for images and masks
        save all metrics into a csv file
"""
print('CellSegmentationEvaluator (SimpleCSE) v1.5.15')
    
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

allmetrics = []

if isdir(img_path):
    imgs = glob.glob(join(img_path, '*.tif*'))
    if not isdir(mask_path):
        print("Image path and mask path must both be folders or both be files.")
        exit()
elif isfile(img_path):
    imgs = img_path
else:
    print("Invalid image path")
    exit()

for ifile in imgs:
    # use same filename for both image and mask
    mfile = join(mask_path, split(ifile)[1])
    seg_metrics = read_and_eval_seg (ifile, mfile, PCA_model, output_directory)
    allmetrics.append(seg_metrics)

if len(imgs) > 1:
    print("Saving all metrics to CSV file in 'results' folder")
    df=pd.DataFrame(allmetrics)
    df["Filename"] = imgs
    df.to_csv(output_directory / "segmentation_metrics.csv", header=True, index=False)
else:
    print(seg_metrics)
