#!/bin/bash
BASEDIR=$(dirname "$0")
source ~/anaconda3/etc/profile.d/conda.sh
conda activate CellProfiler
bash $BASEDIR/get_CellProfiler_cppipe.sh $1 $2
echo $BASEDIR
cellprofiler -r -c -p $1/CellProfiler_config.cppipe -i $1 -o $1
conda deactivate
python $BASEDIR/convert_to_indexed_image.py $1 CellProfiler


