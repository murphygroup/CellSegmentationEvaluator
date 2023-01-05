parentdir="$(dirname "$1")"
git clone https://github.com/ehodneland/cellsegm.git $parentdir/CellSegm
cp $parentdir/installation/run_cellsegm.m $parentdir/CellSegm/example