# Cell Segmentation Evaluator: evaluation of cell segmentation methods without reference segmentations
Haoran Chen and Robert F. Murphy and Ted Zhang\
Carnegie Mellon University\
V1.2 Jan 05, 2023

## Cell Segmentation Approach
This package implements an approach for cell segmentation evaluation (CSE) without relying upon comparison to results from humans. 
For this, we defined a series of segmentation quality metrics that can be applied to multichannel fluorescence images. 
We calculated these metrics for 11 previously-described segmentation methods applied to 2D images from 4 multiplexed microscope modalities covering 5 tissues. 
Using principal component analysis to combine the metrics we defined an overall cell segmentation quality score.
We also defined similar metrics and quality score for 3D images.

Reference: Haoran Chen and Robert F. Murphy (2022) Evaluation of cell segmentation methods without reference segmentations. Molecular Biology of the Cell.  https://doi.org/10.1091/mbc.E22-08-0364

## Package contents
This package contains two implementations of the cell segmentation metrics as well as example images for testing.  The first implementation ("SimpleCSE") just calculates metrics and quality score for one or more images and corresponding cell segmentation masks.  
It was tested on python >=3.8 under Ubuntu 18.04 LTS.

## SimpleCSE

This folder contains a simplified version of the CSE that calculate the metrics and quality score given a multichannel image and a corresponding cell mask.  

It is provided as a python main program and as an example Jupyter Notebook ("SegEvalExample.ipynb").

The multichannel image should be in a format readable by AICSimageio (e.g., OME TIFF).  The masks should be in a similar format with an indexed image for cell masks in the first channel and an indexed image (with corresponding indices) for nuclear masks in the second channel.

The output is a JSON file with the metrics and the scores.

## Execution from command line
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
python seg_eval_main.py --img-dir [img_dir_path] --mask-dir [mask_dir_path]
```
The "example_data" folder contains example 2D CODEX and 3D IMC images and their corresponding cell masks.  To run them, download the example data use  

python seg_eval_main.py --img-dir [example_data/imgs/2D_CODEX.ome.tiff] --mask-dir [example_data/masks/2D_CODEX.ome.tiff]

python seg_eval_main.py --img-dir [example_data/imgs/3D_IMC.ome.tiff] --mask-dir [example_data/masks/3D_IMC.ome.tiff]


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

## Segmentation masks of HuBMAP images
All segmentation masks and evaluation results we generated for the paper can be download from the Google Drive folder below:

https://drive.google.com/drive/folders/14tw4qrXWTt2eg64zOpYjE9cZ7OFo1i6b?usp=sharing

The segmentation masks will be automatically downloaded if you choose to reproduce all results of the paper at step 1 above.

## Documentation 

For more detailed introduction of segmentation quality metrics among other image quality metrics, please see
[HuBMAP Image Quality Control Metrics](http://hubmap.scs.cmu.edu/wp-content/uploads/2021/09/HuBMAP-Image-Quality-Control-Metrics-v1.5.pdf)


## Contact

Robert F. Murphy - murphy@cmu.edu\
Haoran Chen - hrchen@cmu.edu
