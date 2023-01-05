parentdir="$(dirname "$1")"
git clone https://gitlab.com/csb.ethz/CellX.git $parentdir/CellX
cp $parentdir/installation/run_CellX.m $parentdir/CellX
cp $parentdir/installation/get_xml.sh $parentdir/CellX
