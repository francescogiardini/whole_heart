import os
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
# from skimage.external.tifffile import imsave
from tifffile import imwrite as imsave
from tifffile import imread as imread

import warnings

from custom_tool_kit import magnitude


class ImgFrmt:
    # image_format STRINGS
    EPS = 'EPS'
    TIFF = 'TIFF'
    SVG = 'SVG'

# deprecated - it saved with a wrong pixel size and normalized the values
# def plot_map_and_save(matrix, np_filename, base_path, shape_G, shape_P, img_format=ImgFrmt.TIFF, _do_norm=False):
#     """
#     # plot LOCAL disarray (or AVERAGED FA) matrix as frames
#
#     # map_name = 'FA', or 'DISARRAY_ARIT' or 'DISARRAY_WEIGH'
#     # es: plot_map_and_save(matrix_of_disarray, disarray_numpy_filename, True, IMG_TIFF)
#     # es: plot_map_and_save(matrix_of_local_FA, FA_numpy_filename, True, IMG_TIFF)
#
#     :param matrix:
#     :param np_filename:
#     :param save_plot:
#     :param img_format:
#     :param _do_norm:
#     :return:
#     """
#
#     # create folder_path and filename from numpy_filename
#     plot_folder_name = np_filename.split('.')[0]
#     plot_filebasename = '_'.join(np_filename.split('.')[0].split('_')[0:2])
#
#     # create path where save images
#     plot_path = os.path.join(base_path, plot_folder_name)
#     # check if it exist
#     if not os.path.isdir(plot_path):
#         os.mkdir(plot_path)
#
#     # iteration on the z axis
#     for i in range(0, matrix.shape[2]):
#
#         # extract data from the frame to plot
#         if _do_norm:
#             img = normalize(matrix[..., i])
#         else:
#             img = matrix[..., i]
#
#         # evaluate the depth in the volume space
#         z_frame = int((i + 0.5) * shape_G[2] * shape_P[2])
#         # create title of figure
#         title = plot_filebasename + '. Grane: ({} x {} x {}) vectors; Depth_in_frame = {}'.format(
#             int(shape_G[0]), int(shape_G[1]), int(shape_G[2]), z_frame)
#
#         # create plot
#         fig = plt.figure(figsize=(15, 15))
#         plt.imshow(img, cmap='gray')
#         plt.title(title)
#         # plt.show()
#
#         # create fname for this frame
#         fname = plot_filebasename + '_z={}'.format(z_frame)
#
#         if img_format == ImgFrmt.SVG:
#             # formato SVG -> puoi decidere dopo la risoluzione aprendolo con fiji
#             fig.savefig(str(os.path.join(plot_path, fname) + '.svg'), format='svg',
#                         dpi=1200, bbox_inches='tight', pad_inches=0)
#
#         elif img_format == ImgFrmt.EPS:
#             # formato EPS buono per latex (latex lo converte automat. in pdf)
#             fig.savefig(str(os.path.join(plot_path, fname) + '_black.eps'), format='eps', dpi=400,
#                         bbox_inches='tight', pad_inches=0)
#
#         elif img_format == ImgFrmt.TIFF:
#             png1 = BytesIO()
#             fig.savefig(png1, format='png')
#             png2 = Image.open(png1)
#             png2.save((str(os.path.join(plot_path, fname) + '.tiff')))
#             png1.close()
#
#         plt.close(fig)
#
#     return plot_path


def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)


def load_tiff_stack_zyx(filepath):
    '''
    input: filepath
    output: data, shape
    load tifffile from filepath and move axis to (z, y, x)
    return data and shape '''
    data = imread(filepath) # (z, y, x)
    shape = data.shape
    return data, shape


def plot_map_and_save(matrix, np_filename, dest_path, res_xy, res_z, shape_G, shape_P,
                      _save_MIP=False, _save_AVG=False, img_format=ImgFrmt.TIFF):

    # evaluate pixel size of matrix disarray
    ps_disarray = (res_xy * shape_P[0] * shape_G[0],
                   res_xy * shape_P[1] * shape_G[1],
                   res_z * shape_P[2] * shape_G[2])

    # create folder_path and filename from numpy_filename
    plot_folder_name = np_filename.split('.')[0] + '_Dps{0:0.0f}'.format(ps_disarray[0])
    plot_filebasename = '_'.join(np_filename.split('.')[0].split('_')[0:2]) + '_Dps{0:0.0f}'.format(ps_disarray[0])

    # create path where save images
    map_plot_path = os.path.join(dest_path, plot_folder_name)
    # check if it exist
    if not os.path.isdir(map_plot_path):
        os.mkdir(map_plot_path), print('Created ', map_plot_path)

    # figures shape
    w = matrix.shape[1]
    h = matrix.shape[0]

    # iteration on the z axis
    for i in range(0, matrix.shape[2]):

        # extract the current plane to plot
        plane = matrix[..., i]
        # print(plane.shape)
        # print('plane.max: ', plane.max(), '\nplane.min: ', plane.min())

        # estimate profondity of current plane in micron
        z_um = int((i + 0.5) * shape_G[2] * shape_P[2] * res_z)

        # create title of figure
        title = plot_filebasename + '- ps: {0:4.0f} ({1} x {2} x {3}) vectors - z: {4} um.'.format(
            ps_disarray[0], int(shape_G[0]), int(shape_G[1]), int(shape_G[2]), z_um)

        # create fname for this frame
        fname = plot_filebasename + '_z={}um'.format(z_um)

        if img_format == ImgFrmt.TIFF:
            # prepare fig
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # plot data
            ax.imshow(plane, cmap='gray', vmin=0, vmax=255)  # without normalization of the original values

            # save maps and close
            fig.savefig(str(os.path.join(map_plot_path, fname) + '.tiff'), dpi=1)

        elif img_format in [ImgFrmt.SVG, ImgFrmt.EPS]:

            # create plot
            fig = plt.figure(figsize=(15, 15))
            plt.imshow(plane, cmap='gray')
            plt.title(title)

            # save the fig
            if img_format == ImgFrmt.SVG:
                # formato SVG -> puoi decidere dopo la risoluzione aprendolo con fiji
                fig.savefig(str(os.path.join(map_plot_path, fname) + '.svg'), format='svg',
                            dpi=1200, bbox_inches='tight', pad_inches=0)

            elif img_format == ImgFrmt.EPS:
                # formato EPS buono per latex (latex lo converte automat. in pdf)
                fig.savefig(str(os.path.join(map_plot_path, fname) + '_black.eps'), format='eps', dpi=400,
                            bbox_inches='tight', pad_inches=0)

        else:
            print('[plot_map_and_save] - Wrong format passed for save plots.')

        # close the fig
        try:
            plt.close(fig)
        except(NameError, ValueError):
            print('[plot_map_and_save] - There is no fig to close.')

    # save the MIP in tiff
    if _save_MIP:

        # path of the mip
        mip_fname = str('MIP_' + plot_folder_name + '.tiff')
        mip_path = str(os.path.join(os.path.dirname(map_plot_path), mip_fname))

        # convert invalid values to zero (background = not analyzed)
        matrix_with_zero = np.copy(matrix)
        matrix_with_zero[matrix_with_zero < 0] = 0
        MIP = np.max(matrix_with_zero, axis=2)  # sum on Z (rcz)

        # prepare a new fig
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plot MIP
        ax.imshow(MIP, cmap='gray', vmin=0, vmax=255)  # without normalization of the original values

        # save MIP and close
        fig.savefig(mip_path, dpi=1)

    # save the AVERAGE along z axis in tiff (excluding invalid elements == -1)
    if _save_AVG:
        # path of the AVG
        avg_fname = str('AVG_' + plot_folder_name + '.tiff')
        avg_path = str(os.path.join(os.path.dirname(map_plot_path), avg_fname))

        # convert invalid values to NaN
        matrix_with_nan = nan_if(matrix, -1)

        # average along z axis (ignoring the warning about the column with all NaNs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AVG = np.nanmean(matrix_with_nan, axis=2)  # AVG on Z (rcz)

        # if there is nan elements in the final average, convert to zero
        AVG = np.nan_to_num(AVG)

        # prepare a new fig
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plot MIP
        ax.imshow(AVG, cmap='gray', vmin=0, vmax=255)  # without normalization of the original values

        # save AVG and close
        fig.savefig(avg_path, dpi=1)


    return map_plot_path


def plot_histogram(x, xlabel='', ylabel='', xmin=0, xmax=100, bins=100, _save=True, filepath=None,
                   xlabelfontsize=20, ylabelfontsize=20, xticksfontsize=16, yticksfontsize=16):
    """

    :param x: numpy array of values to plot
    :param xlabel: string
    :param ylabel: string
    :param bins: int
    :param _save: boolean
    :param filepath: string with a complete path = '/a/b/filename.tiff' or '/a/b/filename'
    :param xlabelfontsize: int
    :param ylabelfontsize: int
    :param xticksfontsize: int
    :param yticksfontsize: int
    :return: filepath of the saved image (if saved, else None)
    """

    fig = plt.figure(tight_layout=True, figsize=(15, 15))
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.xticks(fontsize=xticksfontsize)
    plt.yticks(fontsize=yticksfontsize)
    plt.hist(x, bins=bins)
    plt.xlim((xmin, xmax))

    if _save:
        png1 = BytesIO()
        fig.savefig(png1, format='png')
        png2 = Image.open(png1)

        # add tiff extension if missing
        if filepath.split('.')[-1] != 'tiff' or filepath.split('.')[-1] != 'tif':
            filepath = filepath + '.tiff'

        png2.save(filepath)
        png1.close()
        return filepath
    return None


# TODO: espamdere ai tre canali con ...
def bleach_correction(vol, z_ref=None, verbose=False):
    '''
    Simple Ratio Bleah correction in the 3d data 'vol'
    '''
    print(vol.shape) # remove

    if vol is not None:
        if len(vol.shape) == 3:
            if vol.shape[0] > 1:

                if z_ref is None:
                    # extract middle frame intensity
                    z_ref = int(vol.shape[0] / 2)
                    while True:
                        ref_value = np.mean(vol[z_ref])
                        if ref_value != 0:
                            break
                        else:
                            z_ref = z_ref + 1
                else:
                    ref_value = np.mean(vol[z_ref])

                # bleach correction
                for z in range(vol.shape[0]):
                    if verbose: print('z: {}'.format(z), end='')
                    m = np.mean(vol[z])
                    if m != 0:
                        ratio = ref_value / m
                        vol[z] = vol[z] * ratio
                        if verbose: print(' - ratio: {0:0.3f}'.format(ratio))
                    else:
                        if verbose: print(' - is empty')
    return vol


def save_tiff(img, img_name, comment='', folder_path='', prefix=''):
    # folder_path must end with '/'

    # check if name end with .tif
    if img_name.endswith('.tiff'):
        base, ext = os.path.splitext(img_name)
        img_name = prefix + base + '_' + comment + '.tiff'
    else:
        img_name = prefix + img_name + '_' + comment + '.tiff'

    # check if path end with '/'
    if not folder_path.endswith('/'):
        folder_path += '/'

    # imsave(folder_path + img_name, (img * (255 / np.max(img))).astype(np.uint8))
    imsave(folder_path + img_name, img)


def image_have_info(img, t=0.09):
    rating = np.sum(img) / np.prod(img.shape)
    if rating > t:
        return True
    else:
        return False


def normalize(img, max_value=255.0, dtype=np.uint8):
    max_v = img.max()
    min_v = img.min()
    if max_v != 0:
        if max_v != min_v:
            return (((img - min_v) / (max_v - min_v)) * max_value).astype(dtype)
        else:
            return ((img / max_v) * max_value).astype(dtype)
    else:
        return img.astype(dtype)


def normalize_ext(img, max_value=None, dtype=None):
    """ Stretch or shrink the values of img.
        img is a numpy array """
    if img is not None:
        if dtype is None:
            dtype = img.dtype
        if max_value is None:
            try:
                max_value = np.iinfo(dtype).max
            except (NameError, ValueError):
                pass

            try:
                max_value = np.finfo(dtype).max
            except (NameError, ValueError):
                pass

            finally:
                max_value = 255 if max_value is None else max_value

        max_v = img.max()
        min_v = img.min()
        if max_v != 0:
            if max_v != min_v:
                return (((img - min_v)/(max_v - min_v)) * max_value).astype(dtype)
            else:
                return ((img / max_v) * max_value).astype(dtype)
        else:
            return img.astype(dtype)
    return None


def split_channel(img):
    img_red = np.array(img)[:, :, 0]
    img_green = np.array(img)[:, :, 1]
    img_blue = np.array(img)[:, :, 2]
    return img_red, img_green, img_blue


def print_info(X, text='', _std=False, _return=False):
    if X is None:
        return None

    info = list()

    info.append(text)
    info.append(' * Image dtype: {}'.format(X.dtype))
    info.append(' * Image shape: {}'.format(X.shape))
    info.append(' * Image max value: {}'.format(X.max()))
    info.append(' * Image min value: {}'.format(X.min()))
    info.append(' * Image mean value: {}'.format(X.mean()))
    if _std:
        info.append(' * Image std value: {}'.format(X.std()))

    if _return:
        return info
    else:
        for l in info:
            print(l)


def create_img_name_from_index(i, pre='', post='', tot_digits=5):
    if 0 <= i < 10 ** tot_digits - 1:
        if i == 0:
            non_zero_digits = 1
        else:
            non_zero_digits = magnitude(i) + 1
        zeros = '0' * (tot_digits - non_zero_digits)
        return str(pre + zeros + str(i) + post + '.tif')
    else:
        return 'image_index_out_of_range'
