# Cell Segmentation Evaluator: evaluation of cell segmentation methods without reference segmentations
Haoran Chen and Robert F. Murphy\
Carnegie Mellon University\
V1.0.2 Aug 24, 2022

## Description of idea
We present here an approach that seeks to evaluate cell segmentation methods without relying upon comparison to results from humans. 
For this, we defined a series of segmentation quality metrics that can be applied to multichannel fluorescence images. 
We calculated these metrics for 11 previously-described segmentation methods applied to datasets from 4 multiplexed microscope modalities covering 5 tissues. 
Using principal component analysis to combine the metrics we defined an overall cell segmentation quality score and ranked the segmentation methods.

## Use of package
This package allows users to
1. Reproduce all results that we generated for the paper on multichannel images from HuBMAP public portal. 
2. Seek the most suitable segmentation method for one or a batch of multichannel images.
3. Obtain the segmentation quality score on given segmentation masks. 

The package was tested under Ubuntu 18.04 LTS.


## Execution
### Step 0: Setup environment
```bash
pip install -r requirements.txt 
```

### Step 1: Setup configuration file
```bash
python generate_config.py 
```
This step generates config.json which contains all configurational parameters and options in order to run the evaluation pipeline. The users will be asked a series of questions about how they would like to utilize this package. The answers will be stored in config.json for the pipeline to read. Meanwhile, all necessay data, softwares and dependencies will also be automatically downloaded or installed based on users' responses.

### Step 2: Run the pipeline
```bash
python run_pipeline.py configuration_file_path
```
This step runs the evaluation pipeline given the generated configuration file.  

## Segmentation masks of HuBMAP images
All segmentation masks and evaluation results we generated for the paper can be download from Google Drive folder below:

https://drive.google.com/drive/folders/14tw4qrXWTt2eg64zOpYjE9cZ7OFo1i6b?usp=sharing

The segmentation masks would be automatically downloaded if you choose to reproduce all results of the paper at step 1 above.

## Documentation 

For more detailed introduction of segmentation quality metrics among other image quality metrics, please see
[HuBMAP Image Quality Control Metrics](http://hubmap.scs.cmu.edu/wp-content/uploads/2021/09/HuBMAP-Image-Quality-Control-Metrics-v1.5.pdf)

## Citation
If you find our package useful in your research, please cite our bioRxiv paper:
> Chen, Haoran, and Robert F. Murphy. "Evaluation of cell segmentation methods without reference segmentations." bioRxiv (2022): 2021-09.

## Contact

Robert F. Murphy - murphy@cmu.edu\
Haoran Chen - hrchen@cmu.edu

