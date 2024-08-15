# corrected shiftsx and shiftsy
# 20220113 by Julia
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
from itertools import repeat
from skimage.registration import phase_cross_correlation



def run_calculate_shifts_experiment(experimentPath, downstreamPath, view_list, hyb_bit_dict):
    # calculate shift for all the view
    # align all other hyb image to hyb1 image
    # do that for all the zscan all the hybs
    # select the best one
    #
    if not os.path.exists(downstreamPath):
        os.mkdir(downstreamPath)
    #
    downstreamPath1 = downstreamPath / 'computed_shifts'
    if not os.path.exists(downstreamPath1):
        os.mkdir(downstreamPath1)
    # loop through all the view and all the hybs
    hyblist = list(hyb_bit_dict.keys())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run_calculate_shifts_view,
                            repeat(experimentPath),
                            repeat(downstreamPath),
                            view_list,
                            repeat(hyblist))
    # concat shifts for different view into one single file
    files = glob.glob(str(downstreamPath1) + '/computed_shifts*.csv')
    temp = [pd.read_csv(fp) for fp in files]
    computed_shifts = pd.concat(temp)
    computed_shifts.sort_values(['zscan', 'hyb'])
    #computed_shifts.to_csv(downstreamPath + '/computed_shifts.csv', index = False)
    # for one view, select the best alignment with lowest error rate
    # parse into hyb view
    get_shift_hyb_view(downstreamPath)
    print('All done!')
    return


def run_calculate_shifts_view(experimentPath, downstreamPath, view, hyblist):
    # calculate shifts for one view all zscans all hybs
    # create save dir
    #downstreamPath1 = downstreamPath + '/computed_shifts'
    #
    print('Calculating shifts for ', view)
    zscanlist_all = os.listdir(experimentPath)
    zscanlist = [x for x in zscanlist_all if int(x.split('_')[1]) >= view[0] and int(x.split('_')[1]) <= view[1]]
    zscanlist.sort()
    #print(zscanlist)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run_calculate_shifts_zscan,
                        repeat(experimentPath),
                        repeat(downstreamPath),
                        zscanlist,
                        repeat(hyblist))
    # save result
    computed_shifts = pd.concat(list(results))
    downstreamPath1 = downstreamPath / 'computed_shifts'
    computed_shifts.to_csv(downstreamPath1 / 'computed_shifts_view_{}_{}.csv'.format(view[0], view[1]), index = False)
    return


def run_calculate_shifts_zscan(experimentPath, downstreamPath, zscan, hyblist):
    # calculate shifts for one zscan
    # for all hybs
    all_dapi_file = glob.glob(str(experimentPath) + '/*/*dapi.tif')
    zscan_dapi_file = [x for x in all_dapi_file if zscan in x]
    zscan_dapi_file_hyb = [x for x in zscan_dapi_file if Path(x).stem.split('_')[0] in hyblist]
    zscan_dapi_file_hyb.sort()
    # get everything
    original_im = [x for x in zscan_dapi_file if 'hyb1_' in x]
    original_image = np.array(Image.open(original_im[0]))
    computed_shifts = []
    #print(zscan_dapi_file_hyb)
    for hybim in zscan_dapi_file_hyb:
        hybimage = np.array(Image.open(hybim))
        #print(original_image, hybimage)
        shift, error, diffphase = phase_cross_correlation(original_image.astype('int8'), hybimage.astype('int8'))
        hybb = Path(hybim).stem.split('_')[0]
        print([zscan, hybb, int(shift[1]), int(shift[0]), error])
        # note that returned shift followed this order
        # Shift vector (in pixels) required to register moving_image with reference_image. Axis ordering is consistent with numpy (e.g. Z, Y, X)
        # therefore in order to get shiftsx and then y, we need to flip it
        computed_shifts.append([zscan,hybb, int(shift[1]), int(shift[0]), error])
    computed_shifts = pd.DataFrame(computed_shifts, columns = ['zscan', 'hyb', 'shiftx', 'shifty', 'error'])
    return computed_shifts



def get_shift_hyb_view(downstreamPath):
    # get shifts
    # should we compile them together?
    shift_hyb_list = []
    downstreamPath1 = downstreamPath / 'computed_shifts'
    shift_view_files = os.listdir(downstreamPath1)
    for shiftviewfp in shift_view_files:
        shift_view = pd.read_csv(downstreamPath1 / shiftviewfp)
        zscanlist = list(set(shift_view['zscan']))
        hyblist = list(set(shift_view['hyb']))
        view = '{}_{}'.format(shiftviewfp.split('.')[0].split('_')[-2], shiftviewfp.split('.')[0].split('_')[-1])
        for hyb in hyblist:
            shift_hyb = shift_view[shift_view['hyb'] == hyb]
            lowest_error = np.where(shift_hyb['error'] == np.min(shift_hyb['error']))[0][0]
            shiftsx = shift_hyb.iloc[lowest_error]['shiftx']
            shiftsy = shift_hyb.iloc[lowest_error]['shifty']
            shift_hyb_list.append([view, hyb, shiftsx, shiftsy])
    shift_hyb = pd.DataFrame(shift_hyb_list, columns = ['view', 'hyb', 'shiftsx', 'shiftsy'])
    shift_hyb.sort_values(['view', 'hyb'])
    shift_hyb.to_csv(downstreamPath / 'computed_hyb_shifts.csv', index = False)



