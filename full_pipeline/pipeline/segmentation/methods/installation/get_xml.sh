num1=$((10 * $2 / 100))
num2=$((20 * $2 / 100))
num3=$((50 * $2 / 100))
cat << EOF > $1/CellX_config.xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<CellXConfiguration creator="CellXGui" timestamp="2013_09_11__08_31_40">
<CellXParam name="membraneIntensityProfile">
<ArrayElement value="0.06"/>
<ArrayElement value="0.05"/>
<ArrayElement value="0.04"/>
<ArrayElement value="0.03"/>
<ArrayElement value="0.02"/>
<ArrayElement value="0.01"/>
<ArrayElement value="-0.01"/>
<ArrayElement value="-0.03"/>
<ArrayElement value="-0.02"/>
<ArrayElement value="-0.04"/>
<ArrayElement value="-0.05"/>
<ArrayElement value="-0.06"/>

</CellXParam>
<CellXParam name="membraneLocation" value="5"/>
<CellXParam name="membraneWidth" value="2"/>
<CellXParam name="maximumCellLength" value="$num3"/>
<CellXParam name="seedRadiusLimit">
<ArrayElement value="$num1"/>
<ArrayElement value="$num2"/>
</CellXParam>
<CellXParam name="isHoughTransformOnCLAHE" value="1"/>
<CellXParam name="seedSensitivity" value="0.05"/>
<CellXParam name="isGraphCutOnCLAHE" value="1"/>
<CellXParam name="claheClipLimit" value="0.01"/>
<CellXParam name="claheBlockSize" value="100"/>
<CellXParam name="maximumMinorAxisLengthHoughRadiusRatio" value="1.7"/>
<CellXParam name="requiredFractionOfAcceptedMembranePixels" value="0.85"/>
<CellXParam name="overlapMergeThreshold" value="0.8"/>
<CellXParam name="overlapResolveThreshold" value="0.2"/>
<CellXParam name="nuclearVolumeFraction" value="0.07"/>
<CellXParam name="intensityBrightAreaPercentage" value="0.3"/>
<CellXParam name="spatialMotionFactor" value="50"/>
<CellXParam name="maxGapConnectionDistance" value="3"/>
</CellXConfiguration>
EOF

