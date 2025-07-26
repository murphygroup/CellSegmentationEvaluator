#from pathlib import Path
import json
from aicsimageio import AICSImage
import numpy as np
#from PIL import Image
from CellSegmentationEvaluator.single_method_eval import single_method_eval
from CellSegmentationEvaluator.single_method_eval_3D import single_method_eval_3D
from os.path import split
#import pickle
#import xmltodict
import tifffile
import xml.etree.ElementTree as ET


"""
MAIN FUNCTION TO CALCULATE SEGMENTATION EVALUATION STATISTICS FOR A
FOR A SINGLE IMAGE AND MASK
Author: Robert F. Murphy and Haoran Chen and Ted Zhang
Version: 1.5.14 April 3, 2025
        just remove unnecessary imports
Version: 1.5.14 April 22, 2025
        just remove redundant version print statement
Version: 1.5.18 June 27, 2025
        use OME-TIFF info to get pixel sizes
Version:1.5.19 July 26, 2025
        pass pixel sizes to single_method_eval_3D
"""

def extract_voxel_size_from_tiff(file_path):
    # Read OME-TIFF metadata
    with tifffile.TiffFile(file_path) as tif:
        metadata = tif.ome_metadata

    print(metadata)
    # Parse the XML metadata
    root = ET.fromstring(metadata)

    # Initialize sizes
    physical_size_x = physical_size_y = physical_size_z = None

    # Iterate over all elements to find the first instance with physical sizes
    for elem in root.iter():
        if 'PhysicalSizeX' in elem.attrib:
            physical_size_x = elem.get('PhysicalSizeX')
        if 'PhysicalSizeY' in elem.attrib:
            physical_size_y = elem.get('PhysicalSizeY')
        if 'PhysicalSizeZ' in elem.attrib:
            physical_size_z = elem.get('PhysicalSizeZ')
        if physical_size_x and physical_size_y and physical_size_z:
            break

    return (physical_size_x, physical_size_y, physical_size_z)


def read_and_eval_seg(img_path, mask_path, PCA_model, output_directory):

    aimg = AICSImage(img_path)
    physical_size_x, physical_size_y, physical_size_z = aimg.physical_pixel_sizes
    #print(xmltodict.parse(aimg.metadata.to_xml()))
    #physical_size_x, physical_size_y, physical_size_z=extract_voxel_size_from_tiff(img_path)
    #print(physical_size_x, physical_size_y, physical_size_z)
    img = {}
    iheadtail = split(img_path)
    img["path"] = iheadtail[0]
    img["name"] = iheadtail[1]
    img["img"]  = aimg
    img["data"] = aimg.get_image_data()
    img["pixelsizes"]=(physical_size_x, physical_size_y, physical_size_z)
    #print(img["data"].shape)

    amask = AICSImage(mask_path)
    mask = {}
    mheadtail = split(mask_path)
    mask["path"] = mheadtail[0]
    mask["name"] = mheadtail[1]
    mask["img"] = amask
    mask["data"] = amask.get_image_data()
    #print(mask["data"].shape)

    if not physical_size_x:
        print('File missing physical pixel sizes...')
        physical_size_x = float(input("Enter size of x pixels: "))
        physical_size_y = float(input("Enter size of y pixels: "))

    if img["data"].shape[2]==1:
        seg_metrics = single_method_eval(img, mask, PCA_model, output_directory, 0,0,physical_size_x, physical_size_y)
    else:
        if not physical_size_z:
            physical_size_z = float(input("Enter size of z pixels: "))
        seg_metrics = single_method_eval_3D(img, mask, PCA_model, output_directory,'um',physical_size_x,physical_size_y,physical_size_z)

    #print(seg_metrics)

    struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}
    with open(
             output_directory / (img["name"] + "-seg_eval.json"), "w"
    ) as json_file:
            #json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)
            json.dump(struct, json_file)

    return seg_metrics
