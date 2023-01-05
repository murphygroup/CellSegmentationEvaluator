#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DeepCell-0.9.0
python $BASEDIR/wrapper/DeepCell-0.9.0_membrane.py $1
conda deactivate

