num1=$((5 * $2 / 100))
num2=$((20 * $2 / 100))
cat << EOF > $1/CellProfiler_config.cppipe
CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
DateRevision:407
GitHash:
ModuleCount:8
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:['To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "[\\\\\\\\/]\\\\.")

Metadata:[module_num:2|svn_version:'Unknown'|variable_revision_number:6|show_window:False|notes:['The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression to extract from file name:^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])
    Regular expression to extract from folder name:(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:Elsewhere...|
    Match file and image metadata:[]
    Use case insensitive matching?:No
    Metadata file name:
    Does cached metadata exist?:No

NamesAndTypes:[module_num:3|svn_version:'Unknown'|variable_revision_number:8|show_window:False|notes:['nucleus:', 'membrane: ']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:Images matching rules
    Select the image type:Grayscale image
    Name to assign these images:nucleus
    Match metadata:[]
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:2
    Single images count:0
    Maximum intensity:65536
    Process as 3D?:No
    Relative pixel spacing in X:1.0
    Relative pixel spacing in Y:1.0
    Relative pixel spacing in Z:1.0
    Select the rule criteria:and (file does contain "nucleus.tif")
    Name to assign these images:nucleus
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Maximum intensity:65536
    Select the rule criteria:and (file does contain "membrane.tif")
    Name to assign these images:membrane
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Maximum intensity:65536

Groups:[module_num:4|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:['']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

IdentifyPrimaryObjects:[module_num:5|svn_version:'Unknown'|variable_revision_number:14|show_window:True|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:nucleus
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):$num1,$num2
    Discard objects outside the diameter range?:Yes
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7.0
    Speed up by using lower-resolution image to find local maxima?:Yes
    Fill holes in identified objects?:After declumping only
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:5000
    Display accepted local maxima?:No
    Select maxima color:Blue
    Use advanced settings?:Yes
    Threshold setting version:12
    Threshold strategy:Global
    Thresholding method:Minimum Cross-Entropy
    Threshold smoothing scale:1.3488
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Log transform before thresholding?:No
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:50
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2.0
    Thresholding method:Otsu

IdentifySecondaryObjects:[module_num:6|svn_version:'Unknown'|variable_revision_number:10|show_window:True|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:Nuclei
    Name the objects to be identified:Cells
    Select the method to identify the secondary objects:Propagation
    Select the input image:membrane
    Number of pixels by which to expand the primary objects:20
    Regularization factor:0.05
    Discard secondary objects touching the border of the image?:No
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredNuclei
    Fill holes in identified objects?:Yes
    Threshold setting version:12
    Threshold strategy:Global
    Thresholding method:Minimum Cross-Entropy
    Threshold smoothing scale:0.0
    Threshold correction factor:0.8
    Lower and upper bounds on threshold:0.0,1.0
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Three classes
    Log transform before thresholding?:No
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:50
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2.0
    Thresholding method:Otsu

DisplayDataOnImage:[module_num:7|svn_version:'Unknown'|variable_revision_number:6|show_window:True|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Display object or image measurements?:Object
    Select the input objects:Cells
    Measurement to display:Number_Object_Number
    Select the image on which to display the measurements:membrane
    Text color:red
    Name the output image that has the measurements displayed:DisplayImage
    Font size (points):10
    Number of decimals:2
    Image elements to save:Image
    Annotation offset (in pixels):0
    Display mode:Color
    Color map:gray
    Display background image?:No
    Color map scale:Use this image's measurement range
    Color map range:0.0,1.0

SaveImages:[module_num:8|svn_version:'Unknown'|variable_revision_number:15|show_window:True|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:DisplayImage
    Select method for constructing file names:Single name
    Select image name for file prefix:membrane
    Enter single file name:mask_CellProfiler
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:png
    Output file location:Default Output Folder|
    Image bit depth:16-bit integer
    Overwrite existing files without warning?:Yes
    When to save:Every cycle
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:No
    Base image folder:Elsewhere...|
    How to save the series:T (Time)
EOF
