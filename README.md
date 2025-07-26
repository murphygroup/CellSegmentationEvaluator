# Cell Segmentation Evaluator: evaluation of cell segmentation methods without reference segmentations
Haoran Chen and Robert F. Murphy and Ted Zhang\
Carnegie Mellon University\
V1.5.19 July 26, 2025

This package implements an approach for cell segmentation evaluation (CSE) that does not rely upon comparison to annotations from humans. For this, we defined a series of segmentation quality metrics that can be applied to multichannel images. The metrics are designed mainly for tissue images assuming that (1) there are multiple channels, and (2) there are multiple cell types that are expected to differ in their expression values for the channels. We calculated these metrics for 11 previously-described segmentation methods applied to 2D images from 4 multiplexed microscope modalities covering 5 tissues. Using principal component analysis to combine the metrics, we defined an overall cell segmentation quality score. The individual metrics and the quality score are returned.

The package supports both 2D and 3D images. It can be installed using

```bash
pip install CellSegmentationEvaluator
```

Then import the desired function, e.g., 
```bash
from CellSegmentationEvaluator import single_method_eval, single_method_eval3D, CSE3D
```

There are three main functions to evaluate an image and associated masks.  For providing inputs as AICSImage structures, use 
```bash
single_method_eval(imgpath, maskpath) #for 2D
single_method_eval3D(imgpath, maskpath) #for 3D
```
Or for providing 3D images and masks as nd-arrays, use
```bash
CSE3D(img, mask, [PCAmodel], [threshimage], [voxel_size])
```
where PCAmodel defaults to "3Dv1.6", threshimage defaults to the same of all channels, and the voxel size is in cubic micrometers and defaults to 1.

Reference:

Chen, Haoran, and Robert F. Murphy. "Evaluation of cell segmentation methods without reference segmentations." Molecular Biology of the Cell 34.6 (2023): ar50. https://doi.org/10.1091/mbc.E22-08-0364

Chen, Haoran, and Robert F. Murphy. "3DCellComposer: A versatile pipeline utilizing 2D cell segmentation methods for 3D cell segmentation" (2023)

## Repository contents
This repository contains two implementations of the cell segmentation metrics as well as example images for testing. The first implementation ("SimpleCSE") calculates metrics and quality scores for one or more images and corresponding cell segmentation masks.  The second ("full_pipeline") runs different cell segmentation programs on a given multichannel image and evaluates the resulting segmentations using the metrics.  There is also a PyPI package described above.

It was tested on Python >=3.8 under Ubuntu 18.04 LTS.

**Important:** This repository uses Git LFS for tracking large files. Please ensure you have [Git LFS](https://git-lfs.github.com/) installed on your machine before cloning or pulling.

## Input files

Two input files are required, one containing a multichannel image (e.g., CODEX image) and the other containing the segmentation mask image to evaluate.  The mask image should contain channels for a cell mask and a nucleus mask (in that order).  Other mask channels may be present but will be ignored.  Each channel should contain an indexed image, in which pixels contain the integer number of the cell that that pixel belongs to.  Note that nuclear masks should not extend beyond cell masks; such nuclear masks will be truncated.  Note also that cell and/or nuclear masks that are not matched will be ignored but reflected in the 'FractionOfMatchedCellsAndNuclei' metric.


## SimpleCSE

This folder contains a simplified version of the CSE that calculates the metrics and quality score given a multichannel image and a corresponding cell mask.  

It is provided as a Python main program and as an example Jupyter Notebook ("SegEvalExample.ipynb").

The multichannel image should be in a format readable by AICSimageio (e.g., OME-TIFF).  The masks should be in a similar format with an indexed image for cell masks in the first channel and an indexed image (with corresponding indices) for nuclear masks in the second channel.

The output is a JSON file with the metrics and the scores.

## Execution from the command line
### Step 1:
Download the "SimpleCSE" folder and change your default directory ("cd SimpleCSE") to that folder
### Step 2:
Run
```bash
pip install -r requirements.txt 
```
### Step 3
Run
```bash
python seg_eval_main.py [img_dir_path] [mask_dir_path]
```
The "example_data" folder contains example 2D CODEX and 3D IMC images and their corresponding cell masks.  To run them, use

```bash
python seg_eval_main.py example_data/imgs/2D_CODEX.ome.tiff example_data/masks/2D_CODEX.ome.tiff
```

```bash
python seg_eval_main.py example_data/imgs/3D_IMC.ome.tiff example_data/masks/3D_IMC.ome.tiff
```

##full CSE pipeline

This implementation can be used to seek the most suitable segmentation method for one or a batch of multichannel images.

## Execution: find most suitable segmentation method for multichannel images
### Step 1:
Download "full_pipeline" folder and change your default directory to that folder
### Step 2: Setup environment
```bash
pip install -r requirements.txt 
```

### Step 3: Setup configuration file
```bash
python generate_config.py 
```
This step generates config.json which contains all configurational parameters and options in order to run the evaluation pipeline. The user will be asked a series of questions about how they would like to utilize this package. The answers will be stored in config.json for the pipeline to read. Meanwhile, all necessary data, software and dependencies will also be automatically downloaded or installed based on user responses.

### Step 4: Run the pipeline
```bash
python run_pipeline.py configuration_file_path
```
This step runs the evaluation pipeline given the generated configuration file.  


## Documentation 

For a more detailed introduction to segmentation quality metrics among other image quality metrics, please see
[HuBMAP Image Quality Control Metrics](http://hubmap.scs.cmu.edu/wp-content/uploads/2021/09/HuBMAP-Image-Quality-Control-Metrics-v1.5.pdf)

## Changes from v1.4 to v1.5

Reconcile differences with the 3D cell segmentation evaluation code in 3DCellComposer.  Convert seg_eval_pkg.py to a PyPI package named CellSegmentationEvaluator and import that package for use in SimpleCSE.  Add new function CSE3D as an alternative to "single_method_eval_3D" so that inputs can be nd-arrays rather than AICSImage structures.

## Changes from v1.5 to v1.5.12

Correct setup.py in pip_package and requirements.txt in SimpleCSE.
Create results folder in SimpleCSE if doesn't exist.

## Changes in v1.5.13

Handle case where feature mean of pixels outside cells but in foreground is zero
Remove pint due to deprecation of getitem

## Changes in v1.5.14

Further correction to case where channel mean of pixels outside cells but in foreground is zero

## Change in v1.5.15

Modify pip package; Change SimpleCSE to use proper CellSegmentationEvaluator package

## Changes in v1.5.16

Modify pip package; Remove unnecessary pint import from functions.py

Modify SimpleCSE/seg_eval_main.py to
    allow folder names for images and masks
    save all metrics into a csv file

## Changes in v1.5.17

Modify pip package: rewrite get_voxel_volume and get_pixel_area and eliminate default pixel sizes

## Changes in v1.5.18/v1.5.19

Modify pip package: force import of get_matched_masks and flatten_dict; fix pixel size checking

Modify SimpleCSE to fix handling of single filename; use OME-TIFF to get pixel sizes or require user to specify if missing

## Contact

Robert F. Murphy - murphy@cmu.edu\
