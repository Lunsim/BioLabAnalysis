import pandas as pd
import xlsxwriter
from openpyxl import Workbook
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import numpy as np
import os
from aicsimageio import AICSImage
from bresenham import bresenham
import re
import shutil

from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, LineString
from scipy.ndimage.morphology import binary_erosion
from sklearn.neighbors import KDTree
from os.path import exists
import matplotlib.patches as patches
import matplotlib.path as mpath
import random
import math
import statistics as stats

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

def stack_czi(image_path, storage_path):
    filename = os.listdir(image_path)
    filenames_folder = []
    sorted_filenames = sorted(filename, key=lambda x: (int(x.split()[1].split('.')[0]), int(x.split()[1].split('.')[1])))
    a_animal = [0]
    for i in range(1, len(sorted_filenames)):
        prev_animal = int(sorted_filenames[i-1].split()[1].split('.')[0])
        curr_animal = int(sorted_filenames[i].split()[1].split('.')[0])
        if prev_animal != curr_animal:
            a_animal.append(i * 2)

    for filename in sorted_filenames:
        if filename.endswith('.czi'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            img = AICSImage(file_path)
            channel_img = []
            for c in range(img.shape[1]):
              image = img.get_image_data("ZYX", C=c)
              image = np.max(image, axis = 0)
              channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            stacked_image = np.stack(channel_img, axis=-1)
            uint8_stacked = im_adjust(stacked_image, autoscale = 'uint8')
            
            base_filename = os.path.splitext(filename)[0]
            tiff_file_path = os.path.join(storage_path, f'{base_filename}.tiff')
            filenames_folder.append(tiff_file_path)
            tifffile.imwrite(tiff_file_path, uint8_stacked)
    return filenames_folder
