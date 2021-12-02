#!/bin/bash
BASEDIR=$(dirname "$0")
cp $BASEDIR/installation/run_Cellsegm.m $BASEDIR/CellSegm/examples
cd $BASEDIR/CellSegm/examples
matlab -nodesktop -r "run_cellsegm $1; quit"
cd ../../
python $BASEDIR/convert_to_indexed_image.py $1 cellsegm
