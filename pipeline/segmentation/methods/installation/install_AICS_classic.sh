#!/bin/bash
conda create -n AICS_classic python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AICS_classic
git clone https://github.com/AllenCell/aics-segmentation.git $1
cd $1/aics-segmentation
git checkout 810ad95
pip install numpy
pip install itkwidgets==0.14.0
pip install -e .[all]
pip install scikit-image==0.18.3
pip install aicssegmentation
cd $1
rm -rf aics-segmentation
conda deactivate
#conda env create -f $1/AICS_classic.yml
