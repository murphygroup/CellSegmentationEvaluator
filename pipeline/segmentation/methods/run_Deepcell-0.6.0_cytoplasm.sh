#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DeepCell-0.6.0
python $BASEDIR/wrapper/DeepCell-0.6.0_cytoplasm.py $1
conda deactivate
