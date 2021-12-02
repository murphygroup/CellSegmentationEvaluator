#!/bin/bash
BASEDIR=$(dirname "$0")
cp $BASEDIR/installation/run_CellX.m $BASEDIR/CellX
cd $BASEDIR/CellX
matlab -nodesktop -r "run_CellX $1; quit"
rm $1/seeding_00001.png
rm $1/membrane.tif_control.png
rm $1/final_contour1.png
mv $1/final_mask1.png mask_CellX.png
python $BASEDIR/convert_to_indexed_image.py $1 CellX
