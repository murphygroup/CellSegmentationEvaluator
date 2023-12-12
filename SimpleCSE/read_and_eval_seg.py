from pathlib import Path
import json
from aicsimageio import AICSImage
import numpy as np
from PIL import Image
from seg_eval_pkg import *
from os.path import split
import pickle

"""
MAIN FUNCTION TO CALCULATE SEGMENTATION EVALUATION STATISTICS FOR A
FOR A SINGLE IMAGE AND MASK
Author: Robert F. Murphy and Haoran Chen and Ted Zhang
Version: 1.4 December 11, 2023
"""
    
def read_and_eval_seg(img_path, mask_path, PCA_model, output_directory):

    print('CellSegmentationEvaluator (SimpleCSE) v1.4')
    aimg = AICSImage(img_path)
    img = {}
    iheadtail = split(img_path)
    img["path"] = iheadtail[0]
    img["name"] = iheadtail[1]
    img["img"]  = aimg
    img["data"] = aimg.get_image_data()
    #print(img["data"].shape)

    amask = AICSImage(mask_path)
    mask = {}
    mheadtail = split(mask_path)
    mask["path"] = mheadtail[0]
    mask["name"] = mheadtail[1]
    mask["img"] = amask
    mask["data"] = amask.get_image_data()
    #print(mask["data"].shape)
    # check if the mask is 2D or 3D (could be 3D with only one slice non-zero)
    bestz = 0
    # if the mask is 3D, test whether only one slice has cells in it
    if mask["data"].shape[2] > 1:
        isum = []
        for iii in range(0,mask["data"].shape[2]):
            isum.append(np.sum(mask["data"][0,:,iii,:,:]))
        #print(isum)
        bestz = np.nonzero(isum)
        #print(bestz)
        # if only one nonzero, reduce mask and img to 2D
        if isinstance(bestz,int):
            mask['data'][0, :, iii, :, :] = mask['data'][0, :, bestz, :, :]
            img['data'][0, :, iii, :, :] = img['data'][0, :, bestz, :, :]

    if isinstance(bestz,int):
        if PCA_model==[]:
            try:
                PCA_model = pickle.load(open( "2D_PCA_model.pickle", "rb" ))
            except:
                print('2D PCA model file missing')
        seg_metrics = single_method_eval(img, mask, PCA_model, output_directory, bz=bestz)
    else:
        if PCA_model==[]:
            try:
                PCA_model = pickle.load( open( "3D_PCA_model.pickle", "rb" ))
            except:
                print('3D PCA model file missing')
        seg_metrics = single_method_eval_3D(img, mask, PCA_model, output_directory)

    #print(seg_metrics)

    struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}
    with open(
             output_directory / (img["name"] + "-seg_eval.json"), "w"
    ) as json_file:
            json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)

    return seg_metrics
