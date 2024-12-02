import os
import re
import math
import random
import statistics as stats
from os.path import exists
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath

import cv2
from aicsimageio import AICSImage
import tifffile
from bresenham import bresenham
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage.morphology import binary_erosion

import geopandas as gpd
from shapely.geometry import Polygon, LineString
from sklearn.neighbors import KDTree

import xlsxwriter
from openpyxl import Workbook, load_workbook

micron_to_pixel_scale = 3.0769

def im_adjust(I, thres=[1, 99, True], autoscale=None):
    # compute percentile: remove too big or too small values
    # thres: [thres_low, thres_high, percentile]
    if thres[2]:
        I_low, I_high = np.percentile(I.reshape(-1), thres[:2])
    else:
        I_low, I_high = thres[0], thres[1]
    # thresholding
    I[I > I_high] = I_high
    I[I < I_low] = I_low
    if autoscale is not None:
        # scale to 0-1
        I = (I.astype(float) - I_low) / (I_high - I_low)
        if autoscale == "uint8":
            # convert it to uint8
            I = (I * 255).astype(np.uint8)
    return I

def process_stack_czi_files(job_id, organized_files, storage_path):
    # Create a new dictionary to store processed files
    processed_files = {}
    
    # Process files for each request
    for req_name, files in organized_files.items():
        processed_files[req_name] = []
        
        # Sort files within each request if needed
        sorted_files = sorted(files, 
                            key=lambda x: (int(x['original_name'].split()[1].split('.')[0]), 
                                         int(x['original_name'].split()[1].split('.')[1])))
        
        for file_info in sorted_files:
            file_path = file_info['path']
            
            # Process only CZI files
            if not file_info['original_name'].endswith('.czi'):
                continue
                
            # Process the file
            img = AICSImage(file_path)
            
            # Process each channel
            channel_img = []
            for c in range(img.shape[1]):
                image = img.get_image_data("ZYX", C=c)
                image = np.max(image, axis=0)
                channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            
            # Stack and adjust image
            stacked_image = np.stack(channel_img, axis=-1)
            uint8_stacked = im_adjust(stacked_image, autoscale='uint8')
            
            # Save as TIFF
            base_filename = os.path.splitext(file_info['original_name'])[0]
            tiff_file_path = os.path.join(storage_path, f'{base_filename}.tiff')
            tifffile.imwrite(tiff_file_path, uint8_stacked)
            
            # Store processed file information
            processed_files[req_name].append({
                'path': tiff_file_path,
                'original_name': file_info['original_name']
            })
    
    return processed_files