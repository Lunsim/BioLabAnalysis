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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        logger.info(f"Starting processing for job {job_id}")
        processed_files = {}
        
        # Ensure storage path exists
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Process files for each requirement
        for req_name, files in organized_files.items():
            logger.info(f"Processing requirement: {req_name}")
            processed_files[req_name] = []
            
            for file_info in files:
                try:
                    file_path = file_info['path']
                    logger.info(f"Processing file: {file_info['original_name']}")
                    
                    # Skip non-CZI files
                    if not str(file_path).lower().endswith('.czi'):
                        logger.warning(f"Skipping non-CZI file: {file_path}")
                        continue
                    
                    # Process the file
                    img = AICSImage(file_path)
                    
                    # Process each channel
                    channel_img = []
                    for c in range(img.shape[1]):
                        # Get image data for channel
                        image = img.get_image_data("ZYX", C=c)
                        
                        # Maximum intensity projection
                        image = np.max(image, axis=0)
                        
                        # Rotate image
                        channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    
                    # Stack channels
                    stacked_image = np.stack(channel_img, axis=-1)
                    
                    # Adjust image
                    uint8_stacked = im_adjust(stacked_image, autoscale='uint8')
                    
                    # Generate output filename
                    base_filename = Path(file_info['original_name']).stem
                    tiff_file_path = storage_path / f'{base_filename}_stacked.tiff'
                    
                    # Save as TIFF
                    tifffile.imwrite(str(tiff_file_path), uint8_stacked)
                    logger.info(f"Saved processed file: {tiff_file_path}")
                    
                    # Store processed file information
                    processed_files[req_name].append({
                        'path': str(tiff_file_path),
                        'original_name': file_info['original_name'],
                        'channels': img.shape[1]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_info['original_name']}: {str(e)}")
                    # Continue with next file instead of failing entire job
                    continue
        
        logger.info(f"Completed processing for job {job_id}")
        return processed_files
        
    except Exception as e:
        logger.error(f"Error in process_stack_czi_files: {str(e)}")
        raise