#!/bin/bash

#conda create -n CellProfiler python=3.8
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate CellProfiler
#python3.8 -m pip install cellprofiler==4.0.5
#conda deactivate
conda env create -f $1/CellProfiler.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate CellProfiler
pip install numpy==1.20.0
conda deactivate
cp $1/displaydataonimage.py ~/anaconda3/envs/CellProfiler/lib/python3.8/site-packages/cellprofiler/modules
