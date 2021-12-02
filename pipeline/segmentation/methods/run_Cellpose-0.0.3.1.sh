#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Cellpose-0.0.3.1
python $BASEDIR/wrapper/Cellpose-0.0.3.1.py $1
conda deactivate