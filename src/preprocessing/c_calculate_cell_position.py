# force X, Y output follow normal cartesian axis order
import os
import glob
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
from itertools import repeat
from typing import Union
import matplotlib.pyplot as plt



''' First run cellpose and
get cell position  '''

# legacy
def create_zscanlist(start, end, dnum):
    # Helper func :create a list of zscan given range and number of digit
    # the start and end will obey python rules, ie. [start, end)
    return ['zscan_{}'.format(str(x).zfill(dnum)) for x in range(start, end + 1)]


# legacy
def run_copy_dapi_experiment(experimentPath, dapiPath, zscanlist, bitlist):
    # copy dapi into one folder
    # prepare for cellpose
    if not os.path.exists(dapiPath):
        os.mkdir(dapiPath)
    # copy files
    for zscan in zscanlist:
        zscanPath = experimentPath + '/' + zscan
        dapi_file_list = glob.glob(zscanPath + '/hyb*_dapi.tif')
        for dapifile in dapi_file_list:
            copyfilename = "{}_{}".format(zscan, dapifile.split('/')[-1])
            shutil.copyfile(dapifile, dapiPath + '/' + copyfilename)
            print('Copied ', copyfilename)


def run_calculate_cellpos_experiment(imageSavePath, downstreamPath, view_list, filter = False, low = 0.5, high = 3, overwrite = False):
    # get cell pos for all views
    # also update the total cell position file
    print('Processing: ', imageSavePath)
    # create save dir
    if not os.path.exists(downstreamPath):
        os.mkdir(downstreamPath)
    # create save dir
    downstreamPath1 = downstreamPath / 'cell_position'
    if not os.path.exists(downstreamPath1):
        os.mkdir(downstreamPath1)
    # grab view file
    view_cellpose_fns = glob.glob(str(imageSavePath) + '/*_cp_masks.tif')
    view_cellpose_fns = [x for x in view_cellpose_fns if int(Path(x).stem.split('_')[2]) in [v[0] for v in view_list]]
    # start looping through view
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(run_calculate_cellpos_view,
                                view_cellpose_fns,
                                repeat(downstreamPath),
                                repeat(filter),
                                repeat(low),
                                repeat(high),
                                repeat(overwrite))
    # update all cell position file
    print('All Done!')
    return


def run_calculate_cellpos_view(view_cellpose_fn, downstreamPath, filter, low, high, overwrite):
    # get cell pos for one view
    # will be parallel between all hybs
    print('Find cell position for ', view_cellpose_fn)
    # create downstream path
    downstreamPath1 = downstreamPath / 'cell_position'
    #
    view = (int(Path(view_cellpose_fn).stem.split('_')[2]), int(Path(view_cellpose_fn).stem.split('_')[3]))
    outputfilename = downstreamPath1 / 'view_{}_{}_cell_positions.csv'.format(view[0], view[1])
    #
    if os.path.exists(outputfilename) & (overwrite == False):
        print('file exist and no overwrite permission, exiting')
        return
    #
    cp = np.array(Image.open(view_cellpose_fn))
    #cellCount = np.max(cp.flatten())
    cell_list = np.unique(cp.flatten())
    cell_list = [x for x in cell_list if x != 0]
    # loop though each cell and get cellsize, cell xy
    temp = []
    for cellid in cell_list:
        # Y, X followed numpy axis order
        Y, X = np.where(cp == cellid)
        temp.append([view, cellid, len(X), np.mean(X), np.mean(Y)])
        #print([view, cellid, len(X), np.mean(X), np.mean(Y)])
    # save everything
    cell_loc = pd.DataFrame(temp, columns = ['view', 'cellid', 'cellsize', 'X', 'Y'])
    cell_loc.sort_values(['view', 'cellid'])
    # filter
    if filter:
        cell_loc_filtered = filter_cell_loc(cell_loc, low, high)
        # save
        cell_loc_filtered.to_csv(outputfilename, index = False)
        # save original
        outputfilename1 = downstreamPath1 / 'view_{}_{}_cell_positions_original.csv'.format(view[0], view[1])
        cell_loc.to_csv(outputfilename1, index = False)
    else:
        cell_loc.to_csv(outputfilename, index = False)


def filter_cell_loc(cellloc_view, low, high):
    # filter out extra small and extra large cells
    # small == lower than low * sd, large == larger than high * sd
    # get mean and std
    m = np.mean(cellloc_view['cellsize'])
    s = np.std(cellloc_view['cellsize'])
    print('filter out cells that lower than', (m - low * s), 'higher than', (m + high * s))
    # grab it
    cellloc_view_filtered = cellloc_view[(cellloc_view['cellsize'] > (m - low * s)) &
                                (cellloc_view['cellsize'] < (m + high * s))]
    # save
    return cellloc_view_filtered


def plot_cell_pos(view, downstreamPath, imageSavePath):
    # incase need to locate individual cells
    # plot cellpose with cellid
    cellpose = imageSavePath / 'stitch_view_{}_{}_clear.tif'.format(view[0], view[1])
    cellpose = np.array(Image.open(cellpose))
    cell_position = downstreamPath / 'cell_position/view_{}_{}_cell_positions.csv'.format(view[0], view[1])
    cell_position = pd.read_csv(cell_position)
    outputpathname = imageSavePath / 'stitch_view_{}_{}_clear_labelled.png'.format(view[0], view[1])
    #
    plt.figure(figsize=(int(cellpose.shape[1] / 300), int(cellpose.shape[0] / 300)), dpi=300)
    plt.imshow(cellpose)
    for ind, row in cell_position.iterrows():
        plt.text(row['X'] - 20, row['Y'] + 20, str(row['cellid']), c = 'yellow', fontsize = 'xx-small')
    #
    plt.tight_layout()
    plt.savefig(outputpathname, dpi = 150)
    plt.close('all')
