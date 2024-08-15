# corrected coordinate, anything with explict x, y, refers to cartesian coordinate
# row and col, is explicitly to table
# when stitching images, each image needs to be flipped vertically, along y axis
# width is horizontal width, height is vertical height (of an image)
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
from itertools import repeat
from typing import Union
from skimage.registration import phase_cross_correlation


def get_new_coord_experiment(coordinate_file, view_list, downstreamPath, dapiDir, coorSavePath, num_digit):
    #
    if not os.path.exists(coorSavePath):
        os.makedirs(coorSavePath)
    #
    with concurrent.futures.ProcessPoolExecutor(max_workers = 12) as executor:
        results = executor.map(get_new_coord,
                            repeat(coordinate_file),
                            view_list,
                            repeat(downstreamPath),
                            repeat(dapiDir),
                            repeat(coorSavePath),
                            repeat(num_digit))



def process_coord(coordinate, view):
    view_coor = coordinate[view[0]:view[1] + 1]
    new_coor = view_coor.copy()
    new_coor['x'] = view_coor['x'] - np.min(view_coor['x'])
    new_coor['y'] = view_coor['y'] - np.min(view_coor['y'])
    scale_factor = list(set(new_coor['x']))
    scale_factor.sort()
    new_coor = new_coor / scale_factor[1] * 2048
    return new_coor


def get_new_coord(coordinate_file, view, downstreamPath, dapiDir, coorSavePath, num_digit):
    # from the original coordinate csv, get new stitched version of coordinate
    # process one view at a time
    #
    # read in the original csv file
    # get the section of selected view
    coordinate = pd.read_csv(coordinate_file, header = None)
    coordinate.columns = ['x', 'y']
    coor = process_coord(coordinate, view)
    # get basic info from coor
    t = np.max(coor, axis = 0) / 2048
    rownum = int(np.round(t[0]) + 1)
    colnum = int(np.round(t[1]) + 1)
    # coor table is the table that specify the location of zscan
    # within the view, accommodate different the imaging sequence
    coor_table = pd.DataFrame(columns = range(int(colnum)), index = range(int(rownum)))
    for index, row in coor.iterrows():
        rowid = int(np.round(row[0] / 2048))
        colid = int(np.round(row[1] / 2048))
        coor_table.iloc[rowid, colid] = index
    # save
    # be aware that this table is not following cartesian coordinate
    coor_table.to_csv(coorSavePath / 'coor_table_view_{}_{}.csv'.format(view[0], view[1]))
    # create table for new stitched coordinate
    # create table for new stitched coordinate
    new_coord = pd.DataFrame(columns = range(int(colnum)), index = range(int(rownum)))
    # iterate though each zscan
    for colid in range(colnum):
        for rowid in range(rownum):
            print('Processing:', view, rowid, colid)
            # if it is the first image, no stitching required
            if rowid == 0 and colid == 0:
                new_coord.iloc[rowid, colid] = (0, 0)
            # otherwise
            else:
                # get current zscan
                zscanid = coor_table.iloc[rowid, colid]
                dapifilename = dapiDir / 'zscan_{}_hyb1_dapi_corrected.tif'.format(str(zscanid).zfill(num_digit))
                # import current image
                currentImage = np.array(Image.open(dapifilename)).astype('uint8')
                currentImage = np.flip(currentImage, axis=0)
                # if it is the first image in one row, stitch to the previous row
                if colid == 0:
                    # grab the first image of last row
                    previouscolid = colid
                    previousrowid = rowid - 1
                    #
                    previouszscanid = coor_table.iloc[previousrowid, previouscolid]
                    previousdapifilename = dapiDir / 'zscan_{}_hyb1_dapi_corrected.tif'.format(str(previouszscanid).zfill(num_digit))
                    #
                    previousImage = np.array(Image.open(previousdapifilename)).astype('uint8')
                    previousImage = np.flip(previousImage, axis=0)
                    #
                    previousx_coord, previousy_coord = new_coord.iloc[previousrowid, previouscolid]
                    # start matching
                    img = previousImage
                    template = currentImage[0:50, 200:2000]
                    # looping though different match template
                    shiftsx = []
                    shiftsy = []
                    for method in range(6):
                        res = cv2.matchTemplate(img,template, method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        if method == 1 or method == 0:
                            shiftsx.append(min_loc[0])
                            shiftsy.append(min_loc[1])
                        else:
                            shiftsx.append(max_loc[0])
                            shiftsy.append(max_loc[1])
                    # filter. shift y (vertical axis) should be around 2048 (bottom edge)
                    # shift x (horizontal axis) should be around 0 (leftmost)
                    # shiftsy ~ [1800 - 2300] means 2048 + [-248, 250]
                    # since template area starts at 200, 0 + [0, 400] means 0 + [-200, 200]
                    # step1, filter those values not in correct range
                    yesind_shiftsx = [ind for ind, x in enumerate(shiftsx) if x > 0 and x < 400]
                    yesind_shiftsy = [ind for ind, y in enumerate(shiftsy) if y > 1800 and y < 2300]
                    shiftsx = [shiftsx[ind] for ind in set(yesind_shiftsx).intersection(yesind_shiftsy)]
                    shiftsy = [shiftsy[ind] for ind in set(yesind_shiftsx).intersection(yesind_shiftsy)]
                    # now select the best one
                    try:
                        sx = max(set(shiftsx), key = shiftsx.count)
                        sy = max(set(shiftsy), key = shiftsy.count)
                        #
                        locx = previousx_coord + sx - 200
                        locy = previousy_coord + sy
                    except ValueError:
                        print('Error: Could not find a good pos for', view, rowid, colid, zscanid)
                        # directly add 2048
                        locx = previousx_coord
                        locy = previousy_coord + 2048
                    #
                    new_coord.iloc[rowid, colid] = (locx, locy)
                # stitch row
                else:
                    previouscolid = colid - 1
                    previousrowid = rowid
                    #
                    previouszscanid = coor_table.iloc[previousrowid, previouscolid]
                    previousdapifilename = dapiDir / 'zscan_{}_hyb1_dapi_corrected.tif'.format(str(previouszscanid).zfill(num_digit))
                    #
                    previousImage = np.array(Image.open(previousdapifilename)).astype('uint8')
                    previousImage = np.flip(previousImage, axis=0)
                    #
                    previousx_coord, previousy_coord = new_coord.iloc[previousrowid, previouscolid]
                    #
                    img = previousImage
                    template = currentImage[200:2000, 0:50]
                    # find shifts
                    shiftsx = []
                    shiftsy = []
                    for method in range(6):
                        res = cv2.matchTemplate(img,template, method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        if method == 1 or method == 0:
                            shiftsx.append(min_loc[0])
                            shiftsy.append(min_loc[1])
                        else:
                            shiftsx.append(max_loc[0])
                            shiftsy.append(max_loc[1])
                    # filter, for shifts x, should be around 2048 (rightmost)
                    # for shifts y, should be around 0, parallel to the original one
                    # shiftsx ~ [- 250, 250] == [2048 - 250, 2048 + 250]
                    # shiftsy ~ [0, 400] means 0 + [-200, 200]
                    # step1, filter those values not in correct range
                    yesind_shiftsx = [ind for ind, x in enumerate(shiftsx) if x > 1800 and x < 2300]
                    yesind_shiftsy = [ind for ind, y in enumerate(shiftsy) if y > 0 and y < 400]
                    shiftsx = [shiftsx[ind] for ind in set(yesind_shiftsx).intersection(yesind_shiftsy)]
                    shiftsy = [shiftsy[ind] for ind in set(yesind_shiftsx).intersection(yesind_shiftsy)]
                    #
                    try:
                        sx = max(set(shiftsx), key = shiftsx.count)
                        sy = max(set(shiftsy), key = shiftsy.count)
                        #
                        locx = previousx_coord + sx
                        locy = previousy_coord + sy - 200
                    except ValueError:
                        print('Error: Could not find a good pos for', view, rowid, colid, zscanid)
                        locx = previousx_coord + 2048
                        locy = previousy_coord
                    #
                    new_coord.iloc[rowid, colid] = (locx, locy)
    #
    new_coord.to_csv(coorSavePath / 'new_coord_view_{}_{}.csv'.format(view[0], view[1]))


def stitch_dapi_experiment(view_list, downstreamPath, imageSavePath, dapiDir, coorSavePath, num_digit):
    # function to parallize across views
    # makedir if not exists
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # start loop
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(stitch_dapi,
                            view_list,
                            repeat(downstreamPath),
                            repeat(imageSavePath),
                            repeat(dapiDir),
                            repeat(coorSavePath),
                            repeat(num_digit))


def stitch_dapi(view, downstreamPath, imageSavePath, dapiDir, coorSavePath, num_digit):
    # stitch zscans in one view together
    #
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # stitch pics
    new_coordinate_fn = 'new_coord_view_{}_{}.csv'.format(view[0], view[1])
    new_coord = pd.read_csv(coorSavePath / new_coordinate_fn, index_col = 0)
    for rowid in new_coord.index:
        for colid in new_coord.columns:
            new_coord.iloc[rowid, int(colid)] = eval(new_coord.iloc[rowid, int(colid)])
    # get size of view
    minwidth = np.min([x[0] for x in new_coord.to_numpy().flatten()])
    minheight = np.min([x[1] for x in new_coord.to_numpy().flatten()])
    maxwidth = np.max([x[0] for x in new_coord.to_numpy().flatten()])
    maxheight = np.max([x[1] for x in new_coord.to_numpy().flatten()])
    width = maxwidth - minwidth + 2048
    height = maxheight - minheight + 2048
    # push corrdinates that is of negative values to positive
    # pad zeros on the edges
    correct_new_coord = new_coord.copy()
    for rowid in new_coord.index:
        for colid in new_coord.columns:
            oldx, oldy = new_coord.iloc[rowid, int(colid)]
            correct_new_coord.iloc[rowid, int(colid)] = (oldx - minwidth, oldy - minheight)
    # save
    correct_new_coord.to_csv(coorSavePath / 'correct_new_coord_view_{}_{}.csv'.format(view[0], view[1]))
    # read in coor table
    coor_table_fn = 'coor_table_view_{}_{}.csv'.format(view[0], view[1])
    coor_table = pd.read_csv(coorSavePath / coor_table_fn, index_col = 0)
    # create initial canvas to put in images
    A = np.zeros(shape = (height, width))
    # loop though view
    for colid in new_coord.columns:
        for rowid in new_coord.index:
            # get input
            zscanid = coor_table.iloc[rowid, int(colid)]
            print('Adding', view, rowid, colid, zscanid)
            # read in images
            dapifilename = dapiDir / 'zscan_{}_hyb1_dapi_corrected.tif'.format(str(zscanid).zfill(num_digit))
            currentImage = np.array(Image.open(dapifilename)).astype('uint8')
            currentImage = np.flip(currentImage, axis=0)
            # get location of that zscan
            coordx, coordy = correct_new_coord.iloc[rowid, int(colid)]
            A[coordy: coordy + 2048, coordx: coordx + 2048] = currentImage
    #plt.imshow(A)
    #plt.show()
    immmm = Image.fromarray(A.astype('uint8'))
    immmm.save(imageSavePath / 'stitch_view_{}_{}_clear.tif'.format(view[0], view[1]))


def generate_new_coord(new_coord):
    # get size of view
    minwidth = np.min([x[0] for x in new_coord.to_numpy().flatten()])
    minheight = np.min([x[1] for x in new_coord.to_numpy().flatten()])
    maxwidth = np.max([x[0] for x in new_coord.to_numpy().flatten()])
    maxheight = np.max([x[1] for x in new_coord.to_numpy().flatten()])
    width = maxwidth - minwidth + 2048
    height = maxheight - minheight + 2048
    # push corrdinates that is of negative values to positive
    # pad zeros on the edges
    correct_new_coord = new_coord.copy()
    for rowid in new_coord.index:
        for colid in new_coord.columns:
            oldx, oldy = new_coord.iloc[rowid, int(colid)]
            correct_new_coord.iloc[rowid, int(colid)] = (oldx - minwidth, oldy - minheight)
    return correct_new_coord




def stitch_others_view(view, downstreamPath, coorSavePath, imageSavePath, filenames, experimentPath, hyb_bit_dict, num_digit, fftthr = 0, correct = True):
    # function to parallize across views
    # makedir if not exists
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # start loop
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(stitch_others,
                            repeat(view),
                            repeat(downstreamPath),
                            repeat(coorSavePath),
                            repeat(imageSavePath),
                            filenames,
                            repeat(experimentPath),
                            repeat(hyb_bit_dict),
                            repeat(num_digit),
                            repeat(fftthr),
                            repeat(correct))


def stitch_others_view_shrink(view, downstreamPath, coorSavePath, imageSavePath, filenames, experimentPath, hyb_bit_dict, num_digit, fftthr = 0, correct = True):
    # function to parallize across views
    # makedir if not exists
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # start loop
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(stitch_others_shrink,
                            repeat(view),
                            repeat(downstreamPath),
                            repeat(coorSavePath),
                            repeat(imageSavePath),
                            filenames,
                            repeat(experimentPath),
                            repeat(hyb_bit_dict),
                            repeat(num_digit),
                            repeat(fftthr),
                            repeat(correct))

def stitch_others(view, downstreamPath, coorSavePath, imageSavePath, filename, experimentPath, hyb_bit_dict, num_digit, fftthr = 0, correct = True):
    #
    #
    print(filename)
    #
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # stitch pics
    correct_new_coordinate_fn = 'correct_new_coord_view_{}_{}.csv'.format(view[0], view[1])
    correct_new_coord = pd.read_csv(coorSavePath + '/' + correct_new_coordinate_fn, index_col = 0)
    #
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            correct_new_coord.iloc[rowid, int(colid)] = eval(correct_new_coord.iloc[rowid, int(colid)])
    # get parameters
    width = np.max([x[0] for x in correct_new_coord.to_numpy().flatten()]) + 2048
    height = np.max([x[1] for x in correct_new_coord.to_numpy().flatten()]) + 2048
    #
    coor_table_fn = 'coor_table_view_{}_{}.csv'.format(view[0], view[1])
    coor_table = pd.read_csv(coorSavePath + '/' + coor_table_fn, index_col = 0)
    #
    if correct:
        correct_dict = generate_correct_dict()
    # create canvas
    A = np.zeros(shape = (height, width))
    # loop through
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            # get zscan
            zscanid = coor_table.iloc[rowid, int(colid)]
            print(filename, view, rowid, colid, zscanid)
            # get image
            dapifilename = '{}/zscan_{}/{}'.format(experimentPath, str(zscanid).zfill(num_digit), filename)
            currentImage = np.array(Image.open(dapifilename)).astype('uint16')
            # do fft if necessary
            if fftthr:
                filter_temp = filter2d(fftthr,10000000,save=False,order=2,xdim=2048,
                        ydim=2048,plot_filter=False,plot_impulse_response=False)
                currentImage = do_filter_im(currentImage, filter_temp)
            # do correct
            if correct:
                currentImage = correct_image_im(currentImage, filename, correct_dict, hyb_bit_dict)
            # get location
            coordx, coordy = correct_new_coord.iloc[rowid, int(colid)]
            #
            currentImage = np.flip(currentImage, axis=0)
            A[coordy: coordy + 2048, coordx: coordx + 2048] = currentImage
    #plt.imshow(A)
    #plt.show()
    # adjust range
    highthr = np.percentile(A.flatten(), 99.9)
    #highthr = np.max(A.flatten())
    lowthr = np.min(A.flatten())
    A[A > highthr] = highthr
    AA = 255 * (A.astype('float') / highthr)
    # Shift it if necessary
    # get info
    bitid = int(filename.split('.')[0].split('_')[1])
    hyb = [x for x in hyb_bit_dict.keys() if bitid in hyb_bit_dict[x]][0]
    # get hyb shift
    hybshift = pd.read_csv('{}/computed_hyb_shifts.csv'.format(downstreamPath))
    viewid = '{}_{}'.format(view[0], view[1])
    hybshift_view = hybshift[hybshift['view'] == viewid]
    shiftsx, shiftsy = hybshift_view.loc[hybshift_view['hyb'] == hyb, ['shiftsx', 'shiftsy']].values[0]
    # shift: go shiftsx rightward and shiftsy upward (when they are both pos)
    # need to figure out how the images are moved using matrix
    # axis 0 correspond to shiftsy, axis1 correspond to shiftsx
    # shiftsx pos, go rightward, the start position go from 0 to shiftsx
    # the rightmost will exceed original range
    # shift entire image by shiftsx, if positive, rightway, negative, leftway
    if shiftsx > 0:
        shifted_x_AA =  np.zeros(shape = AA.shape)
        shifted_x_AA[:, shiftsx:] = AA[:, :-shiftsx]
    # shifts x negative, go left
    # the leftmost region will exceed range
    elif shiftsx < 0:
        shifted_x_AA = np.zeros(shape = AA.shape)
        shifted_x_AA[:, :shiftsx] = AA[:, -shiftsx:]
    else:
        shifted_x_AA = AA
    # shift y direction
    # shiftsy pos, go upward, the topmost most part will exceed
    if shiftsy > 0:
        shifted_xy_AA =  np.zeros(shape = AA.shape)
        shifted_xy_AA[shiftsy:, :] = shifted_x_AA[:-shiftsy, :]
    # shiftsy negative, go downward, the bottom part exceed range
    elif shiftsy < 0:
        shifted_xy_AA =  np.zeros(shape = AA.shape)
        shifted_xy_AA[:shiftsy, :] = shifted_x_AA[-shiftsy:, :]
    else:
        shifted_xy_AA = shifted_x_AA
    # create outputfilename
    tag = ''
    if fftthr:
        tag = tag + '_fft_' + str(fftthr)
    #
    if correct:
        tag = tag + '_corrected'
    #
    outputfilename = '{}/stitch_{}_view_{}_{}{}.tif'.format(imageSavePath, filename.split('.')[0], view[0], view[1], tag)
    Image.fromarray(shifted_xy_AA.astype('uint8')).save(outputfilename)
    print(outputfilename, 'saved!')
    return


def stitch_others_shrink(view, downstreamPath, coorSavePath, imageSavePath, filename, experimentPath, hyb_bit_dict, num_digit, fftthr = 0, correct = True):
    #
    #
    print(filename)
    #
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    # stitch pics
    correct_new_coordinate_fn = 'correct_new_coord_view_{}_{}.csv'.format(view[0], view[1])
    correct_new_coord = pd.read_csv(coorSavePath + '/' + correct_new_coordinate_fn, index_col = 0)
    #
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            correct_new_coord.iloc[rowid, int(colid)] = eval(correct_new_coord.iloc[rowid, int(colid)])
    # get parameters
    width = np.max([x[0] for x in correct_new_coord.to_numpy().flatten()]) + 2048
    height = np.max([x[1] for x in correct_new_coord.to_numpy().flatten()]) + 2048
    #
    coor_table_fn = 'coor_table_view_{}_{}.csv'.format(view[0], view[1])
    coor_table = pd.read_csv(coorSavePath + '/' + coor_table_fn, index_col = 0)
    #
    if correct:
        correct_dict = generate_correct_dict()
    # create canvas
    A = np.zeros(shape = (height, width))
    # loop through
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            # get zscan
            zscanid = coor_table.iloc[rowid, int(colid)]
            print(filename, view, rowid, colid, zscanid)
            # get image
            dapifilename = '{}/zscan_{}/{}'.format(experimentPath, str(zscanid).zfill(num_digit), filename)
            currentImage = np.array(Image.open(dapifilename)).astype('uint16')
            # do fft if necessary
            if fftthr:
                filter_temp = filter2d(fftthr,10000000,save=False,order=2,xdim=2048,
                        ydim=2048,plot_filter=False,plot_impulse_response=False)
                currentImage = do_filter_im(currentImage, filter_temp)
            # do correct
            if correct:
                currentImage = correct_image_im(currentImage, filename, correct_dict, hyb_bit_dict)
            # get location
            coordx, coordy = correct_new_coord.iloc[rowid, int(colid)]
            #
            currentImage = np.flip(currentImage, axis=0)
            A[coordy: coordy + 2048, coordx: coordx + 2048] = currentImage
    #plt.imshow(A)
    #plt.show()
    # adjust range
    highthr = np.percentile(A.flatten(), 99.9)
    #highthr = np.max(A.flatten())
    lowthr = np.min(A.flatten())
    A[A > highthr] = highthr
    AA = 255 * (A.astype('float') / highthr)
    # Shift it if necessary
    # get info
    bitid = int(filename.split('.')[0].split('_')[1])
    hyb = [x for x in hyb_bit_dict.keys() if bitid in hyb_bit_dict[x]][0]
    # get hyb shift
    hybshift = pd.read_csv('{}/computed_hyb_shifts.csv'.format(downstreamPath))
    viewid = '{}_{}'.format(view[0], view[1])
    hybshift_view = hybshift[hybshift['view'] == viewid]
    shiftsx, shiftsy = hybshift_view.loc[hybshift_view['hyb'] == hyb, ['shiftsx', 'shiftsy']].values[0]
    # shift: go shiftsx rightward and shiftsy upward (when they are both pos)
    # need to figure out how the images are moved using matrix
    # axis 0 correspond to shiftsy, axis1 correspond to shiftsx
    # shiftsx pos, go rightward, the start position go from 0 to shiftsx
    # the rightmost will exceed original range
    # shift entire image by shiftsx, if positive, rightway, negative, leftway
    if shiftsx > 0:
        shifted_x_AA =  np.zeros(shape = AA.shape)
        shifted_x_AA[:, shiftsx:] = AA[:, :-shiftsx]
    # shifts x negative, go left
    # the leftmost region will exceed range
    elif shiftsx < 0:
        shifted_x_AA = np.zeros(shape = AA.shape)
        shifted_x_AA[:, :shiftsx] = AA[:, -shiftsx:]
    else:
        shifted_x_AA = AA
    # shift y direction
    # shiftsy pos, go upward, the topmost most part will exceed
    if shiftsy > 0:
        shifted_xy_AA =  np.zeros(shape = AA.shape)
        shifted_xy_AA[shiftsy:, :] = shifted_x_AA[:-shiftsy, :]
    # shiftsy negative, go downward, the bottom part exceed range
    elif shiftsy < 0:
        shifted_xy_AA =  np.zeros(shape = AA.shape)
        shifted_xy_AA[:shiftsy, :] = shifted_x_AA[-shiftsy:, :]
    else:
        shifted_xy_AA = shifted_x_AA
    # create outputfilename
    tag = ''
    if fftthr:
        tag = tag + '_fft_' + str(fftthr)
    #
    if correct:
        tag = tag + '_corrected'
    #
    outputfilename = '{}/stitch_{}_view_{}_{}{}_shrink.tif'.format(imageSavePath, filename.split('.')[0], view[0], view[1], tag)
    im = Image.fromarray(shifted_xy_AA.astype('uint8'))
    im = im.resize((int(np.round(im.size[0] / 10)), int(np.round(im.size[1] / 10))))
    im.save(outputfilename)
    print(outputfilename, 'saved!')
    return


def generate_correct_dict():
    try:
        dict_paths = ['/mnt/winstor/stella/downstream/me210921Seqfish_lib8/average_488.tif',
                        '/mnt/winstor/stella/downstream/me210921Seqfish_lib8/average_cy3.tif',
                        '/mnt/winstor/stella/downstream/me210921Seqfish_lib8/average_cy5.tif']
        correct_dict1 = [np.array(Image.open(f)) for f in dict_paths]
        correct_dict = [c / np.max(c.flatten()) for c in correct_dict1]
        print("Import mnt")
    except FileNotFoundError:
        dict_paths = ['/nfs/winstor/isogai/stella/downstream/me210921Seqfish_lib8/average_488.tif',
                        '/nfs/winstor/isogai/stella/downstream/me210921Seqfish_lib8/average_cy3.tif',
                        '/nfs/winstor/isogai/stella/downstream/me210921Seqfish_lib8/average_cy5.tif']
        correct_dict1 = [np.array(Image.open(f)) for f in dict_paths]
        correct_dict = [c / np.max(c.flatten()) for c in correct_dict1]
        print("Import nfs")
    return correct_dict



def correct_image_im(currentImage, filename, correct_dict, hyb_bit_dict):
    # correct image according to its channel number
    #print('I am being corrected.')
    # get the channcel number using hyb_bit_dict
    bitid = int(filename.split('.tif')[0].split('_')[1])
    hyb = [x for x in hyb_bit_dict.keys() if bitid in hyb_bit_dict[x]][0]
    channel = np.where(np.array(hyb_bit_dict[hyb]) == bitid)[0][0]
    # do correction, correct twice
    currentImage_corrected1 = currentImage / correct_dict[channel]
    currentImage_corrected2 = currentImage_corrected1 / correct_dict[channel]
    return currentImage_corrected2


def do_filter_im(im, freq_filter):
    # ----------
    #im = np.array(Image.open(file_path))
    #print(f"Image shape is {im.shape}")
    #print(f"Image shape is {im.shape}")
    # ____ FFT ____
    im = im.astype('float')
    imfft = np.fft.fftshift(np.fft.fft2(im))
    # ____ Image (FFT and IFFT back) ____
    # ____ Filtered FFT ____
    imfft_after_filter = imfft * freq_filter
    # ____ Image (FFT, Filter and IFFT back) ____
    im_after_filter = np.fft.ifft2(np.fft.fftshift(imfft_after_filter))
    # ======================================================================
    #  Figures of image, freqency space image and filtered (image and freq)
    # ======================================================================
    # set up figure
    im_pos = np.abs(im_after_filter)
    #im_pos = im_after_filter.real
    #im_pos[im_pos < 0] = 0
    return im_pos


def filter2d(low_cut: Union[float, None] = None,
             high_cut: Union[float, None] = None,
             order: int = 1,
             filter_path: str = None,
             use_existing: bool = True,
             save: bool = True,
             image: np.ndarray = None,
             ydim: int = None,
             xdim: int = None,
             plot_filter: bool = False,
             plot_impulse_response: bool = False,
             verbose: bool = True,
             ) -> np.ndarray:
    """
    returns 2D butterworth filter centered around 0
    low_cut:
        lower frequency cut - acts as high pass filter
    high_cut:
        higher frequency cut - acts as low pass filter
    filter_path: str
        directory where filters are saved / will be saved
    use_existing: bool
        whether to use existing filter (if found)
    save: bool
        whether to save filter in data_path for future use
    order: int
        how steeply the filter changes at the cutoff frequency
    image (2D ndarray), ydim, xdim:
        get dimension of image from the image provided, or
        directly specify y dimension (ydim) and x dimension (xdim)
    plot_filter:
        whether to show a plot of the filter
    plot_impulse_response:
        whether to plot the impulse response of the filter
    """
    # Create a descriptive name for the filter from provided parameters
    # -----------------------------------------------------------------
    filename = f"filter2d_order{order}"
    if low_cut is not None:
        filename += f"_lowcut_{low_cut}"
    if high_cut is not None:
        filename += f"_highcut_{high_cut}"
    filename += f"_{ydim}_{xdim}.npy"
    # Check if we already have the correct filter saved.
    # If found, use it instead of computing from scratch
    if filter_path is not None:
        fullpath = os.path.join(filter_path, filename)
        if verbose:
            print("Filter filepath:", fullpath)
        if use_existing and os.path.isfile(fullpath):
            frequency_mask = np.load(fullpath)
            if verbose:
                print(f"\nExisting filter found: {filename}. Using existing...\n")
            return frequency_mask
    # Get dimensions of image
    # -----------------------
    if image is not None and len(image.shape) == 2:
        ydim, xdim = image.shape
    else:
        assert ydim is not None, f"y dimension not provided!"
        assert xdim is not None, f"x dimension not provided!"
    # Initialize arrays
    # ----------------
    high_cut_mask = np.ones((ydim, xdim), dtype=np.float64)
    low_cut_mask = np.ones_like(high_cut_mask)
    # find mid-point (must -0.5 to match pixel center position)
    y_mid, x_mid = ydim / 2 - 0.5, xdim / 2 - 0.5
    if verbose:
        print(f"Mid point of filter: y = {y_mid:.3f}, x = {x_mid:.3f}")
    grid = np.mgrid[0:ydim, 0:xdim]  # grid[0] is y-coordinate, grid[1] is x-coordinate
    distance_to_mid = np.sqrt((grid[0] - y_mid) ** 2 + (grid[1] - x_mid) ** 2)
    if high_cut is not None:
        # no need to worry about dividing by 0 because we are not dividing by the distance-to-mid
        high_cut_mask = 1 / np.sqrt(
            1 + (np.sqrt((grid[0] - y_mid) ** 2 + (grid[1] - x_mid) ** 2) / high_cut) ** (2 * order)
        )
    if low_cut is not None:
        # the following is done to prevent divide by 0 errors
        # right at the center where distance to mid is 0
        where_to_operate = distance_to_mid != 0
        zeros_array = np.zeros_like(low_cut_mask)
        omega_fraction = np.divide(low_cut,
                                   distance_to_mid,
                                   out=zeros_array,
                                   where=where_to_operate)
        # print(np.argwhere(np.logical_or(np.isnan(omega_fraction), np.isinf(omega_fraction))))
        low_cut_mask = np.divide(low_cut_mask,  # an array of ones
                                 np.sqrt(1 + omega_fraction ** (2 * order)),
                                 out=zeros_array,
                                 where=where_to_operate)
    frequency_mask = high_cut_mask * low_cut_mask
    # check for funny values
    # print(np.argwhere(np.logical_or(np.isnan(frequency_mask), np.isinf(frequency_mask))))
    # frequency_mask = np.nan_to_num(frequency_mask, copy=False)
    # save filter
    # -----------
    if save:
        if filter_path is None:
            print("Filter path not provided. Not saving filter")
        else:
            if not os.path.isdir(filter_path):
                os.mkdir(filter_path)
            fullpath = os.path.join(filter_path, filename)
            print(f"Saving file as: {fullpath}")
            np.save(fullpath, frequency_mask)
    #
    # Plot the filter and/or impulse response, with a colorbar
    # --------------------------------------------------------
    #
    return frequency_mask
