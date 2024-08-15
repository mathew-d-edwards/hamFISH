# functions to process seqfish data
# Isogai Lab September 20th 2021
# Create by Julia

# added in different high cutoff threshold feature for each bit
# Jan 10, 2022 by julia

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
from itertools import repeat
from typing import Union


''' calculate positive cells ###
### calculate positive cells ###
### calculate positive cells '''


def run_calculate_cell_lumi_seqfish_experiment(fftthr, 
                                               view_list, 
                                               experimentPath, 
                                               imageSavePath, 
                                               coorSavePath, 
                                               downstreamPath, 
                                               hyb_bit_dict, 
                                               num_digit = 3, 
                                               correct = False, 
                                               bit_ratio_filter = {},
                                               correct_files = []):
    # calculate the seqfish for entire experiment
    # loop through view_list first
    # then over zscan, then over hyb, finally over bitsa
    # start loop
    if not os.path.exists(downstreamPath / 'complex_lumi'):
        os.mkdir(downstreamPath / 'complex_lumi')
    #a
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(run_calculate_cell_lumi_seqfish_view,
                            view_list,
                            repeat(experimentPath),
                            repeat(imageSavePath),
                            repeat(coorSavePath),
                            repeat(downstreamPath),
                            repeat(fftthr),
                            repeat(hyb_bit_dict),
                            repeat(num_digit),
                            repeat(correct),
                            repeat(bit_ratio_filter),
                            repeat(correct_files))





def run_calculate_cell_lumi_seqfish_view(view, 
                                         experimentPath, 
                                         imageSavePath, 
                                         coorSavePath, 
                                         downstreamPath, 
                                         fftthr,
                                         hyb_bit_dict, 
                                         num_digit, 
                                         correct, 
                                         bit_ratio_filter,
                                         correct_files):
    # import all the data for that view
    # the minimum unit is one view, you are adjust the number of genes using fftthr
    if not os.path.exists(downstreamPath / 'complex_lumi'):
        os.mkdir(downstreamPath / 'complex_lumi')
    #
    print('Processing', view, 'using fft', fftthr, 'using cutoff', bit_ratio_filter)
    # import all the things
    filter_temp = filter2d(fftthr,1000,save=False,order=2,xdim=2048,
                        ydim=2048,plot_filter=False,plot_impulse_response=False)
    #
    if correct:
        correct_dict = generate_correct_dict(correct_files)
    else:
        correct_dict = []
    # cellpose
    cellpose_fn = imageSavePath / 'stitch_view_{}_{}_clear_cp_masks.tif'.format(view[0], view[1])
    cellpose = np.array(Image.open(cellpose_fn))
    # get tables
    coor_table_fn = coorSavePath / 'coor_table_view_{}_{}.csv'.format(view[0], view[1])
    coor_table = pd.read_csv(coor_table_fn, index_col = 0)
    # get coordinates
    correct_new_coord_fn = coorSavePath / 'correct_new_coord_view_{}_{}.csv'.format(view[0], view[1])
    correct_new_coord = pd.read_csv(correct_new_coord_fn, index_col = 0)
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            correct_new_coord.iloc[rowid, int(colid)] = eval(correct_new_coord.iloc[rowid, int(colid)])
    # get hybshift
    hybshift = pd.read_csv(downstreamPath / 'computed_hyb_shifts.csv')
    viewid = '{}_{}'.format(view[0], view[1])
    hybshift_view = hybshift[hybshift['view'] == viewid]
    # this zscanlist contains infor for rowid, colid and zscanid
    zscanlist = [[rowid, int(colid), coor_table.iloc[rowid, int(colid)]] for rowid in coor_table.index for colid in coor_table.columns]
    # calculate complex lumi for each individual zscan
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(run_calculate_cell_lumi_seqfish_zscan,
                        zscanlist,
                        repeat(experimentPath),
                        repeat(correct_new_coord),
                        repeat(hybshift_view),
                        repeat(cellpose),
                        repeat(hyb_bit_dict),
                        repeat(filter_temp),
                        repeat(correct_dict),
                        repeat(num_digit))
    # end loop
    cell_lumiii_complex_view = pd.concat(list(results))
    # merged and add loc
    merged_cellloc = merge_cells_and_get_cellpose(cell_lumiii_complex_view, view, downstreamPath)
    merged_cellloc = merged_cellloc.sort_values(['cellid', 'bit'])
    ## add loc
    #merged_cellloc = add_cell_loc(merged, view, downstreamPath)
    # filter for high r
    rfiltered_merged_cellloc = filter_r(merged_cellloc, bit_ratio_filter)
    # save
    if correct:
        outputfilename = downstreamPath/ 'complex_lumi/cell_lumiii_complex_view_{}_{}_fft_{}_corrected.csv'.format(view[0], view[1], fftthr)
    else:
        outputfilename = downstreamPath / 'complex_lumi/cell_lumiii_complex_view_{}_{}_fft_{}.csv'.format(view[0], view[1], fftthr)
    # save
    rfiltered_merged_cellloc.to_csv(outputfilename, index = False)
    print(outputfilename, 'done!')
    return merged_cellloc


def filter_r(merged_cellloc, bit_ratio_filter):
    #
    filter_merged_cellloc = merged_cellloc.copy()
    filter_merged_cellloc.index = np.arange(merged_cellloc.shape[0])
    # filter for each bit
    for bit in bit_ratio_filter.keys():
        if bit in np.unique(filter_merged_cellloc['bit']):
            temp_merged_cellloc_bit = filter_merged_cellloc[filter_merged_cellloc['bit'] == int(bit)]
            r_thr = bit_ratio_filter[bit]
            discard_index = temp_merged_cellloc_bit[temp_merged_cellloc_bit['noise_ratio'] > r_thr].index
            filter_merged_cellloc = filter_merged_cellloc.drop(discard_index)
    # done
    return filter_merged_cellloc


def run_calculate_cell_lumi_seqfish_zscan(x, experimentPath, correct_new_coord, hybshift_view, cellpose, hyb_bit_dict, filter_temp, correct_dict, num_digit):
    # now loop though hybs in one zscan
    # will do fft thr and shading correction
    # grab basic info
    rowid = x[0]
    colid = x[1]
    zscanid = 'zscan_{}'.format(str(x[2]).zfill(num_digit))
    #
    print('Processing', zscanid)
    #
    coor_x, coor_y = correct_new_coord.iloc[rowid, colid]
    height, width = cellpose.shape
    # looping through hybs
    storage = []
    hyblist = list(hyb_bit_dict.keys())
    for hyb in hyblist:
        # find cellpose region that match with bit image
        # this is reverse transform, cellpose need to match with bit
        shiftx, shifty = hybshift_view.loc[hybshift_view['hyb'] == hyb, ['shiftsx', 'shiftsy']].values[0]
        # now start to get cropped region
        x_loc = coor_x + shiftx
        y_loc = coor_y + shifty
        # dealing with extreme cases
        if x_loc <= 0:
            xxstart = 0
            xxend = x_loc + 2048
        elif x_loc + 2048 > width:
            xxstart = x_loc
            xxend = width
        else:
            xxstart = x_loc
            xxend = x_loc + 2048
        # get y
        if y_loc <= 0:
            yystart = 0
            yyend = y_loc + 2048
        elif y_loc + 2048 > height:
            yystart = y_loc
            yyend = height
        else:
            yystart = y_loc
            yyend = y_loc + 2048
        # now get the region
        cellpose_frag = cellpose[yystart: yyend, xxstart: xxend]
        # now fill in rest
        if (cellpose_frag.shape[0] == 2048 and cellpose_frag.shape[1] == 2048):
            cellpose_frag_canvas = cellpose_frag
        else:
            print('This is fragmented', rowid, colid, zscanid, hyb, cellpose_frag.shape)
            if x_loc < 0:
                canvas_start_x = np.abs(x_loc)
                canvas_end_x = 2048
            else:
                canvas_start_x = 0
                canvas_end_x = cellpose_frag.shape[1]
            # now do y
            if y_loc < 0:
                canvas_start_y = np.abs(y_loc)
                canvas_end_y = 2048
            else:
                canvas_start_y = 0
                canvas_end_y = cellpose_frag.shape[0]
            # put it in
            cellpose_frag_canvas = np.zeros(shape = (2048, 2048))
            cellpose_frag_canvas[canvas_start_y : canvas_end_y, canvas_start_x : canvas_end_x] = cellpose_frag
        # do things
        hyb_bitlist = hyb_bit_dict[hyb]
        hyb_bitlist = [b for b in hyb_bitlist if b != 0]
        bitFilePaths = [experimentPath / str(zscanid) / 'bit_{}.tif'.format(b) for b in hyb_bitlist]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_calculate_cell_lumi_seqfish_single,
                            bitFilePaths,
                            repeat(cellpose_frag_canvas),
                            repeat(filter_temp),
                            repeat(correct_dict),
                            repeat(hyb_bit_dict))
        # end loop
        cell_lumiii_complex_hyb = pd.concat(list(results))
        cell_lumiii_complex_hyb['hyb'] = hyb
        storage.append(cell_lumiii_complex_hyb)
    #
    cell_lumiii_complex_zscan = pd.concat(storage)
    cell_lumiii_complex_zscan['zscanid'] = zscanid
    cell_lumiii_complex_zscan = cell_lumiii_complex_zscan.sort_values(['cellid', 'bit'])
    return cell_lumiii_complex_zscan

'''
plt.imshow(cellpose_frag_canvas, origin = 'lower')
plt.imshow(use_bit_file, origin = 'lower', vmax = 3000)
plt.show()'''


def run_calculate_cell_lumi_seqfish_single(bit_filepath, cellpose_frag, filter_temp, correct_dict, hyb_bit_dict):
    # calculate the complex for one bit file
    # bit file is already ffted and flipped
    # get bitid
    bitid = int(bit_filepath.stem.split('_')[1])
    '''
    # if correct, use correct function
    if len(correct_dict) > 0:
        if bitid in noise_cutoff.keys():
            use_bit_file = correct_image(bit_filepath, correct_dict, filter_temp, hyb_bit_dict, noise_cutoff[bitid])
        else:
            # 100 means no cutoff
            use_bit_file = correct_image(bit_filepath, correct_dict, filter_temp, hyb_bit_dict, 100)
    else:
        bit_file = np.array(Image.open(bit_filepath))
        if bitid in noise_cutoff.keys():
            im_highcutoff = noise_cutoff[bitid]
        else:
            im_highcutoff = 100
        #
       	cutoff_value = np.percentile(bit_file, im_highcutoff)
        bit_file[bit_file > cutoff_value] = cutoff_value
        bit_file_flipped = np.flip(bit_file, axis = 0)
        use_bit_file = do_filter_im(bit_file_flipped, filter_temp)
    '''
    # import original first
    original_bit_file = np.array(Image.open(bit_filepath))
    original_bit_file_flipped = np.flip(original_bit_file, axis = 0)
    # if correct, use correct function
    if len(correct_dict) > 0:
        use_bit_file = correct_image(bit_filepath, correct_dict, filter_temp, hyb_bit_dict, 100)
    else:
        #bit_file = np.array(Image.open(bit_filepath))
        #bit_file_flipped = np.flip(original_bit_file, axis = 0)
        use_bit_file = do_filter_im(original_bit_file_flipped, filter_temp)
    # get basic stats
    background_intensity = np.percentile(original_bit_file_flipped, 98)
    high_intensity = np.percentile(original_bit_file_flipped, 99.99)
    #
    cell_list = np.unique(cellpose_frag.flatten())
    cell_list.sort()
    cell_list = [x for x in cell_list if x != 0]
    #
    cell_complex = []
    cell_discarded = []
    for cellid in cell_list:
        #print(cellid)
        #print('Processing cell', cellid)
        # get the pixel locations of that cell
        cell_locs = np.where(cellpose_frag == cellid)
        cell_locsx = cell_locs[1]
        cell_locsy = cell_locs[0]
        ind = [i for i in range(len(cell_locsx)) if cell_locsx[i] >= 0 and cell_locsx[i] < 2048 and cell_locsy[i] >= 0 and cell_locsy[i] < 2048]
        cell_locsx = cell_locsx[ind]
        cell_locsy = cell_locsy[ind]
        bb = use_bit_file[(cell_locsy, cell_locsx)]
        original_bb = original_bit_file_flipped[(cell_locsy, cell_locsx)]
        #print('original bb', np.sum(original_bb))
        '''
        plt.scatter(cell_locsx, cell_locsy, c = bb)
        plt.show()
        ccc = cellpose_frag_canvas.copy()
        ccc[ccc > 0] = 100
        plt.imshow(ccc + use_bit_file, origin = 'lower', vmax = 1000)
        #plt.imshow(use_bit_file, origin = 'lower', vmax = np.percentile(use_bit_file, 99.9))
        plt.show()
        '''
        # get the ratio of bright over normal
        if len(original_bb[original_bb > background_intensity]) == 0:
            r = 0
        else:
            r = len(original_bb[original_bb > high_intensity]) / len(original_bb[original_bb > background_intensity])
            #print(len(original_bb[original_bb > high_intensity]) / len(original_bb[original_bb > background_intensity]),
            #    len(bb[bb > high_intensity]) / len(bb[bb > background_intensity]))
        # if the ratio is smaller than cutoff, save
        #if r <= noise_cutoff:
        cell_complex.append([bitid, cellid, len(bb), np.sum(bb), np.average(bb), r])
        #else:
            #cell_discarded.append([np.mean(cell_locsx), np.mean(cell_locsy)])
            #print(cellid, r)
    # save
    cell_lumiii_complex = pd.DataFrame(cell_complex, columns = ['bit', 'cellid', 'cellsize', 'total_lumi', 'average_lumi', 'noise_ratio'])
    #cell_discarded = pd.DataFrame(cell_discarded, columns = ['x', 'y'])
    #cell_discarded.to_csv('/mnt/winstor/stella/downstream/me211216Seqfish_lib8_corrected/{}_discarded.csv'.format(bit_filepath.split('/')[-1]))
    #save_diagnosis(cellpose_frag, use_bit_file, bit_filepath)
    return cell_lumiii_complex






### helper functions ###

def merge_cells_and_get_cellpose(cell_lumiii_complex_view, view, downstreamPath):
    # for cells on the edge, since we stitch it, it could be share between two zscans
    # need to merge replicated entries
    # also need to filter out small cells
    # grab cell position info (which already filtered small cells)
    cellloc_view = downstreamPath / 'cell_position/view_{}_{}_cell_positions.csv'.format(view[0], view[1])
    cellloc_view = pd.read_csv(cellloc_view)
    # subset the data
    cv_selected = cell_lumiii_complex_view[cell_lumiii_complex_view['cellid'].isin(cellloc_view['cellid'])]
    # start creating new data
    # since each bit might have unique cell list
    bitlist = list(set(cell_lumiii_complex_view['bit']))
    t = []
    for bit in bitlist:
        #
        cv_selected_bit = cv_selected[cv_selected['bit'] == bit]
        # from one bit, get the unique and duplicated cellids
        cellids = list(cv_selected_bit['cellid'])
        unique_cellids = list(set(cellids))
        duplicates = [x for x in unique_cellids if cellids.count(x) > 1]
        # create storage file
        column = list(cv_selected.columns) + ['X', 'Y', 'cellsize_hyb1']
        new_temp = pd.DataFrame(index = unique_cellids, columns = column)
        for cellid in unique_cellids:
            # grab cell location and size info
            [X, Y, cellsize] = cellloc_view.loc[cellloc_view['cellid'] == cellid, ['X', 'Y', 'cellsize']].values[0]
            if cellid in duplicates:
                # if that one is duplicated, take average
                rows = cv_selected_bit.loc[cv_selected_bit['cellid'] == cellid]
                # grab info
                bit = rows['bit'].values[0]
                hyb = rows['hyb'].values[0]
                noise_ratio = rows['noise_ratio'].values[0]
                # assign zscanid to the zscan with largest area
                zscanid = rows.loc[rows['cellsize'].astype(int).idxmax(), 'zscanid']
                # get average
                sumcellsize = np.sum(rows['cellsize'])
                total_lumi = np.sum(rows['total_lumi'])
                average_lumi = np.sum(rows['total_lumi']) / np.sum(rows['cellsize'])
                # now append
                new_temp.loc[cellid] = np.array([bit, cellid, sumcellsize, total_lumi, average_lumi, noise_ratio, hyb, zscanid, X, Y, cellsize], dtype=object)
            else:
                new_temp.loc[cellid] = np.array(list(cv_selected_bit.loc[cv_selected_bit['cellid'] == cellid].values[0]) + [X, Y, cellsize], dtype=object)
        #
        t.append(new_temp)
    #
    merged = pd.concat(t)
    return merged


# legacy now
def add_cell_loc(merged, view, downstreamPath):
    #
    cellloc_view = downstreamPath / 'cell_position' / 'view_{}_{}_cell_positions.csv'.format(view[0], view[1])
    cellloc_view = pd.read_csv(cellloc_view)
    #
    new_temp = merged.copy()
    new_temp[['cellsize_hyb1', 'X', 'Y']] = np.nan
    new_temp.index = range(new_temp.shape[0])
    cellids = list(set(merged['cellid']).intersection(set(cellloc_view['cellid'])))
    for cellid in cellids:
        print(cellid)
        rowids = new_temp[new_temp['cellid'] == cellid].index
        new_temp.loc[rowids, ['cellsize_hyb1', 'X', 'Y']] = cellloc_view.loc[cellloc_view['cellid'] == cellid, ['cellsize', 'X', 'Y']].values[0]
    #
    return new_temp




def save_diagnosis(cellpose_frag, use_bit_file, bit_filepath):
    # save diagnosis file
    cc = cellpose_frag.copy()
    cc[cc > 0] = 500
    dd = use_bit_file + cc
    Image.fromarray(dd).save(str(bit_filepath).split('.')[0] + '_dig.tif')



def compile_result(downstreamPath, gene_bit_dict, tag):
    # concat every lumi into one file
    # no thresholding, just saving data
    # get data
    fps = glob.glob('{}/complex_lumi/*.csv'.format(downstreamPath))
    fps = [fp for fp in fps if tag in fp]
    # read in all the files and add full cellid
    # expid + viewid + cellid
    dd = []
    for fp in fps:
        # get info
        view_id = '_'.join(fp.split('/')[-1].split('.csv')[0].split('_')[4:6])
        expid = fp.split('/')[-3][2:8]
        #
        readin = pd.read_csv(fp)
        new_cellid = ['_'.join([expid, view_id, str(int(cid))]) for cid in readin['cellid']]
        readin['new_cellid'] = new_cellid
        dd.append(readin)
    # concat that together
    data = pd.concat(dd)
    # get unique cellid and bitid
    cellid = list(set(data['new_cellid']))
    bitid = list(set(data['bit']))
    # compile result
    result = pd.DataFrame(index = cellid, columns = bitid)
    for bit in bitid:
        subset = data[data['bit'] == bit]
        result.loc[subset['new_cellid'], bit] = subset['average_lumi'].values
    # put bitid into gene
    new_col = [gene_bit_dict[bit] for bit in bitid]
    result.columns = new_col
    # save
    result.to_csv(downstreamPath / 'raw_gene_lumi_{}_{}.csv'.format(downstreamPath.stem, tag))


def generate_correct_dict(correct_files):
    dict_paths = correct_files
    correct_dict1 = [np.array(Image.open(f)) for f in dict_paths]
    correct_dict = [c / np.max(c.flatten()) for c in correct_dict1]
    return correct_dict



"""
# legacy
def generate_correct_dict(correct_files):
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
"""


def generate_new_coord(new_coord):
    # get absolute value
    minwidth = np.min([x[0] for x in new_coord.to_numpy().flatten()])
    minheight = np.min([x[1] for x in new_coord.to_numpy().flatten()])
    maxwidth = np.max([x[0] for x in new_coord.to_numpy().flatten()])
    maxheight = np.max([x[1] for x in new_coord.to_numpy().flatten()])
    width = maxwidth - minwidth + 2048
    height = maxheight - minheight + 2048
    #
    correct_new_coord = new_coord.copy()
    for rowid in correct_new_coord.index:
        for colid in correct_new_coord.columns:
            oldx, oldy = new_coord.iloc[rowid, int(colid)]
            correct_new_coord.iloc[rowid, int(colid)] = (oldx - minwidth, oldy - minheight)
    return correct_new_coord


def correct_image(bit_filepath, correct_dict, filter_temp, hyb_bit_dict, im_highcutoff):
    #print('I am being corrected.')
    # import that file
    bit_file = np.array(Image.open(bit_filepath))
    # set up a cutoff threshold
    cutoff_value = np.percentile(bit_file, im_highcutoff)
    bit_file[bit_file > cutoff_value] = cutoff_value
    # flip the image
    bit_file_flipped = np.flip(bit_file, axis = 0)
    # fft
    bit_file_flipped_fft = do_filter_im(bit_file_flipped, filter_temp)
    # do shading
    bitid = int(bit_filepath.stem.split('_')[1])
    hyb = [x for x in hyb_bit_dict.keys() if bitid in hyb_bit_dict[x]][0]
    channel = np.where(np.array(hyb_bit_dict[hyb]) == bitid)[0][0]
    # do correction
    bit_file_flipped_fft_corrected1 = bit_file_flipped_fft / correct_dict[channel]
    bit_file_flipped_fft_corrected = bit_file_flipped_fft_corrected1 / correct_dict[channel]
    return bit_file_flipped_fft_corrected


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






'''
diagnoistic codes

bit_filepath = '/mnt/winstor/stella/processed/me210921Seqfish_lib8/zscan_59/bit_28.tif'
cellpose_fp = '/mnt/winstor/stella/processed/me210921Seqfish_lib8/zscan_60/hyb10_dapi.tif'
shading = '/mnt/winstor/stella/processed/me210921Seqfish_lib8/zscan_60/hyb10_dapi.tif'
bit_file = np.array(Image.open(bit_filepath))
bit_file_flipped = np.flip(bit_file, axis = 0)
cellpose_frag = np.array(Image.open(cellpose_fp))
cellpose_frag_flipped = np.flip(cellpose_frag, axis = 0)
plt.imshow(cellpose_frag_flipped)
plt.show()
plt.imshow(bit_file_flipped, vmax = cutoff_value)
plt.show()

bit_filepath = '/mnt/winstor/stella/processed/me210921Seqfish_lib8/zscan_59/bit_32.tif'
bit_file = np.array(Image.open(bit_filepath))
bit_file_flipped = np.flip(bit_file, axis = 0)
plt.imshow(bit_file_flipped, vmax = 5000)
plt.show()

# set up a cutoff threshold
bit_file = np.array(Image.open(bit_filepath))
im_highcutoff = 99.7
cutoff_value = np.percentile(bit_file, im_highcutoff)
bit_file[bit_file > cutoff_value] = cutoff_value
# flip the image
bit_file_flipped = np.flip(bit_file, axis = 0)
plt.imshow(bit_file_flipped, vmax = cutoff_value)
plt.show()

# fft
filter_temp = filter2d(fftthr,1000,save=False,order=2,xdim=2048,
                        ydim=2048,plot_filter=False,plot_impulse_response=False)
bit_file_flipped_fft = do_filter_im(bit_file_flipped, filter_temp)
plt.imshow(bit_file_flipped_fft, vmax = 200)
plt.show()



filter_temp = filter2d(fftthr,1000,save=False,order=2,xdim=2048,
                        ydim=2048,plot_filter=False,plot_impulse_response=False)
bit_file_flipped_fft = do_filter_im(bit_file_flipped, filter_temp)
plt.imshow(bit_file_flipped_fft, vmax = 200)
plt.show()

bit_file_flipped_fft1 = bit_file_flipped_fft / correct_dict[2]
plt.imshow(bit_file_flipped_fft1, vmax =1000)
plt.show()

bit_file_flipped_fft2 = bit_file_flipped_fft1 / correct_dict[2]
plt.imshow(bit_file_flipped_fft2, vmax = 6000)
plt.show()

bit_file_flipped_fft3 = bit_file_flipped_fft2 / correct_dict[0]
plt.imshow(bit_file_flipped_fft3, vmax = 6000)
plt.show()

bit_file_flipped_fft2 = bit_file_flipped_fft1 / correct_dict[0]
plt.imshow(bit_file_flipped_fft2, vmax = 6000)
plt.show()


shading_img = '/mnt/winstor/stella/tiledviews/me210921Seqfish_lib8/me210921Seqfish_lib8_average_dapi.tif'
shading = np.array(Image.open(shading_img))
plt.imshow(shading)
plt.show()

shading = shading / np.max(shading.flatten())
correct_img = bit_file_flipped_fft / shading
plt.imshow(correct_img, vmax = 7000)
plt.show()

plt.imshow(bit_file_flipped_fft, vmax = 5000)
plt.show()
'''

