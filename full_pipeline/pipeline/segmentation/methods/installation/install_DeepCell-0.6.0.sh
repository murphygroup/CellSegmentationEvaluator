#!/bin/bash
conda create -n DeepCell-0.6.0 python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DeepCell-0.6.0
git clone https://github.com/vanvalenlab/deepcell-tf.git $1/deepcell-tf
cd $1/deepcell-tf
git checkout ad2b019
pip install deepcell_toolbox==0.8.0
python setup.py install
cd $1
rm -rf deepcell-tf
pip install scikit-image==0.18.2
pip install pandas==0.23.3
pip install scikit-learn==0.20.0
pip install scikit-image==0.14.5
pip install imagecodecs
pip install h5py==2.10.0
pip install google_pasta==0.1.6
pip install grpcio==1.8.6
pip install opt-einsum==2.3.2
pip install numpy==1.20.0
pip install pycocotools==2.0.0
pip install tensorboard==1.15.0
pip install tensorflow-estimator==1.15.1
pip install --upgrade google-api-python-client
pip install absl-py
pip install wrapt
pip install gast==0.2.2
pip install astor
pip install termcolor==1.1.0
conda deactivate
#conda env create -f $1/DeepCell-0.6.0.yml