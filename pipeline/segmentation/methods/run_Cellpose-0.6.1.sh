#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Cellpose-0.6.1
python $BASEDIR/wrapper/Cellpose-0.6.1.py $1
conda deactivate

