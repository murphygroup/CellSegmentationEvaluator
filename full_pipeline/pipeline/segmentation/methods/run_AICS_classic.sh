#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AICS_classic
python $BASEDIR/wrapper/AICS_classic.py $1
conda deactivate


