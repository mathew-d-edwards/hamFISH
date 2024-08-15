""" After everything is processed, if we need to flip the entire sample, we can 
simple filp the images and the location of cells, instead of flipping all the original images.
"""

import glob
import math
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

### ----------------------------- func ----------------------------- ###
### ----------------------------- func ----------------------------- ###
### ----------------------------- func ----------------------------- ###

def rotate_arrayofpoints_90clockwise(height, apoint):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    angle = -1.5708
    ox, oy = [0, 0]
    px, py = apoint[:, 0], apoint[:, 1]
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    qy = qy + height
    return qx, qy


def rotate_arrayofpoints_90counterclockwise(width, apoint):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    angle = 1.5708
    ox, oy = [0, 0]
    px, py = apoint[:, 0], apoint[:, 1]
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    qx = qx + width
    return qx, qy


def rotate_arrayofpoints_180(width, height, apoint):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    angle = 3.142
    ox, oy = [0, 0]
    px, py = apoint[:, 0], apoint[:, 1]
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    qx = qx + height
    qy = qy + width
    return qx, qy




def rotate_dapi_cellpos(dapiPath, cellposPath, rotation_angle, flip_horizontal):
    
    # grab files
    dapi_files = glob.glob('{}/stitch_view_*_*_clear.tif'.format(dapiPath))
    cellpose_files = glob.glob('{}/stitch_view_*_*_clear_cp_masks.tif'.format(dapiPath))
    cell_position_files = glob.glob('{}/view_*_*_cell_positions.csv'.format(cellposPath))
    view_list = ['_'.join(Path(fp).stem.split('_')[2:4]) for fp in dapi_files]

    ## rotate dapi and cell masks files, as well the cell positions
    for view in view_list:
        ##
        dapifp = [fp for fp in dapi_files if view in fp][0]
        cellposefp = [fp for fp in cellpose_files if view in fp][0]
        cell_position_fp = [fp for fp in cell_position_files if view in fp][0]
        ## open files ##
        open_dapi = Image.open(dapifp)
        open_cellpose = Image.open(cellposefp)
        width, height = open_dapi.size
        cluster_loc = pd.read_csv(cell_position_fp)
        ## plot original
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(open_dapi, origin = 'lower')
        axes[0].scatter(cluster_loc['X'], cluster_loc['Y'], s = 5)
        axes[1].imshow(open_cellpose, vmax = 1, origin = 'lower')
        axes[1].scatter(cluster_loc['X'], cluster_loc['Y'], s = 5)
        plt.title('origin view ' + view + dapiPath.parts[-2])
        plt.show()
        
        ## rotate dapi
        rotated_dapi = open_dapi.rotate(rotation_angle, expand=True)
        rotated_cellpose = open_cellpose.rotate(rotation_angle, expand=True)
        if flip_horizontal == True:
            rotated_dapi = rotated_dapi.transpose(Image.FLIP_LEFT_RIGHT)
            rotated_cellpose = rotated_cellpose.transpose(Image.FLIP_LEFT_RIGHT)
        ## save
        rotated_dapi.save(dapifp.split('.tif')[0] + '_rotated.tif')
        rotated_cellpose.save(cellposefp.split('.tif')[0] + '_rotated.tif')
        ## plot rotated
        '''
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rotated_dapi, origin = 'lower')
        axes[1].imshow(rotated_cellpose, vmax = 1, origin = 'lower')
        plt.title('rotated view ' + view)
        plt.show()
        '''
        ###
        if rotation_angle == 90:
            qx, qy = rotate_arrayofpoints_90clockwise(width, np.array(cluster_loc[['X', 'Y']]))
        elif rotation_angle == 180:
            qx, qy = rotate_arrayofpoints_180(height, width , np.array(cluster_loc[['X', 'Y']]))
        elif rotation_angle == 270:
            qx, qy = rotate_arrayofpoints_90counterclockwise(height, np.array(cluster_loc[['X', 'Y']]))
        else:
            qx = cluster_loc['X']
            qy = cluster_loc['Y']
            print('do nothing')
        ##
        if flip_horizontal:
            qx = np.min([width, height]) - qx
        ## plot rotated
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rotated_dapi, origin = 'lower')
        axes[0].scatter(qx, qy, s = 5)
        axes[1].imshow(rotated_cellpose, vmax = 1, origin = 'lower')
        axes[1].scatter(qx, qy, s = 5)
        plt.suptitle('view: {}, exp: {}, rotated view and cell pos'.format(view, dapiPath.parts[-2]))
        plt.show()
        ###
        cluster_loc['X'] = qx
        cluster_loc['Y'] = qy
        cluster_loc.to_csv(cell_position_fp.split('.csv')[0] + '_rotated.csv')