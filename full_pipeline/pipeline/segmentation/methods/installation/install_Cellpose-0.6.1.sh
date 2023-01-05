#!/bin/bash
conda create -n Cellpose-0.6.1 python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Cellpose-0.6.1
pip install cellpose==0.6.1
pip install mxnet==1.8.0
pip install scikit-image==0.18.1
pip install imagecodecs
conda deactivate
#conda env create -f $1/Cellpose-0.6.1.yml
