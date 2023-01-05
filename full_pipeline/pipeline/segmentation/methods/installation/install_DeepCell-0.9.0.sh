#!/bin/bash
conda create -n DeepCell-0.9.0 python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DeepCell-0.9.0
pip install deepcell==0.9.0
pip install scikit-image==0.18.1
pip install imagecodecs
conda deactivate
#conda env create -f $1/DeepCell-0.9.0.yml