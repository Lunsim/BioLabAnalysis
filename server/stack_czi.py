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

def stack_czi(images):
    image_list = []
    for img in images:
        img = AICSImage(file_path)
        image_data = img.get_image_data("YX")
        #uint8_stacked = im_adjust(image_data, autoscale = 'uint8')
        image_list.append(image_data)
    return image_list