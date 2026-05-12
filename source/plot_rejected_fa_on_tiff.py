import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
import argparse

from skimage.exposure import equalize_adapthist as skimage_clahe
from tifffile import imread as imread

from custom_tool_kit import search_value_in_txt
from custom_image_base_tool import print_info, normalize

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 14)

Image.MAX_IMAGE_PIXELS = 300000000  # Alza il limite di pixel che può gestire per ogni immagine


# image_format STRINGS
IMG_TIFF = 'TIFF'


class Bcolors:
    V = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_dots_2d_for_save(xc, yc, img=None, shape=None, origin='upper',
                          cmap_image='gray', marker_size=30, marker_color='r',
                          edge_color='black', edge_width=2,
                          _show_plot=False):
    """
    Plot dots at positions (xc, yc) over an image.
    xc, yc: arrays of x (horizontal axis) and y (vertical axis) coordinates of dot centers.
    """
    fig = plt.figure(tight_layout=True, figsize=(15, 15))
    dpi = fig.get_dpi()
    print("Generated a Fig with dpi: ", dpi)

    # plot tiff frame under the dots
    if img is not None:
        plt.imshow(img, origin=origin, cmap=cmap_image, alpha=0.9)
        if shape is None:
            shape = img.shape

    # remove axes to reduce white border when saving
    ax = plt.gca()
    ax.set_axis_off()
    fig.add_axes(ax)

    # plot all dots
    plt.scatter(xc, yc, s=marker_size, c=marker_color, marker='o',
                edgecolors=edge_color, linewidths=edge_width, alpha=0.8)

    if shape is not None:
        fig.set_size_inches(shape[1] / dpi, shape[0] / dpi)
        print('shape_inside: ', shape)
        print('set_size_inches : ', shape[1] / dpi, shape[0] / dpi)

    if _show_plot:
        plt.show()
    return fig, plt


def save_fig_as_tiff(fig, filepath):
    """Save a matplotlib figure as TIFF via PNG in memory."""
    png1 = BytesIO()
    fig.savefig(png1, format='png')
    png2 = Image.open(png1)
    png2.save(filepath)
    png1.close()


def make_output_dir(base_path, fa_threshold, stack_prefix, plane_name, _equalize, clip):
    """Create and return the output directory path for a given plane.
    Structure: rejected_fa{TH}_R_{prefix}/{prefix}_{plane}/{prefix}_{plane}_raw (or _clahed)
    """
    rejected_path = os.path.join(base_path, 'rejected_fa{0}_R_{1}'.format(
        fa_threshold, stack_prefix))

    plane_with_prefix = '{0}_{1}'.format(stack_prefix, plane_name)
    rejected_path = os.path.join(rejected_path, plane_with_prefix)

    if _equalize:
        subfolder_name = '{0}_clahed{1:0.2f}'.format(plane_with_prefix, clip)
    else:
        subfolder_name = '{0}_raw'.format(plane_with_prefix)

    rejected_path = os.path.join(rejected_path, subfolder_name)
    rejected_path = rejected_path + '/'

    if not os.path.isdir(rejected_path):
        os.makedirs(rejected_path)

    return rejected_path


def plot_rejected_on_frames(source_path, parameter_filename, R_filepath, fa_threshold,
                            _equalize=False, clip=0.03, maxPixelValue=150,
                            _show_plots=False, marker_size=30, _orthogonals=True,
                            marker_color='r', edge_color='black', edge_width=2):

    base_path = os.path.dirname(source_path)
    stack_name = os.path.basename(source_path)
    stack_prefix = stack_name.split('.')[0]
    parameter_filepath = os.path.join(base_path, parameter_filename)

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'mode_ratio', 'threshold_on_cell_ratio',
                   'fwhm_xy', 'fwhm_z']

    param_values = search_value_in_txt(parameter_filepath, param_names)

    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])

    # ratio between pixel size in z and xy
    ps_ratio = parameters['px_size_z'] / parameters['px_size_xy']

    # analysis block dimension in z-axis
    num_of_slices_P = int(parameters['roi_xy_pix'] / ps_ratio)

    # Parameters of Acquisition System:
    res_z = parameters['px_size_z']
    res_xy = parameters['px_size_xy']
    print('Pixel size (Real) in XY : {} um'.format(res_xy))
    print('Pixel size (Real) in Z : {} um'.format(res_z))

    # dimension of analysis block (parallelogram)
    row_P = col_P = int(parameters['roi_xy_pix'])
    shape_P = np.array((row_P, col_P, num_of_slices_P)).astype(np.int32)
    print('Dimension of Parallelepiped : {} pixel'.format(shape_P))
    print('Dimension of Parallelepiped : {} um'.format(np.array(shape_P) * np.array([res_xy, res_xy, res_z])))

    # load R
    R_filename = os.path.basename(R_filepath)
    R = np.load(R_filepath)
    shape_R = R.shape
    print('R loaded from: ', R_filepath)
    print('shape_R: ', shape_R)

    # OPEN STACK
    t0 = time.time()
    volume = imread(source_path)

    # NB - Attenzione gestione assi: tifffile: (z,y,x) ---> ME: (r,c,z)=(y,x,z)
    volume = np.moveaxis(volume, 0, -1)  # (z, y, x) -> (r, c, z)

    t1 = time.time()

    # calculate dimension
    shape_V = np.array(volume.shape)
    pixels_for_slice = shape_V[0] * shape_V[1]
    total_voxel_V = pixels_for_slice * shape_V[2]
    print('Entire Volume dimension:')
    print('Volume shape     (r, c, z) : ({}, {}, {})'.format(shape_V[0], shape_V[1], shape_V[2]))
    print('Pixel for slice            : {}'.format(pixels_for_slice))
    print('Total voxel in Volume      : {}'.format(total_voxel_V))
    print('\n')
    print(' 3D stack readed in {0:.3f} seconds'.format(t1 - t0))

    print_info(volume, text='Volume')

    cmap_image = 'gray'

    # ==== EXTRACT REJECTED BLOCKS FROM R =====
    # Rejected = cell_info is True BUT fa <= fa_threshold
    rejected_bool = (R['cell_info'] == True) & (R['fa'] <= fa_threshold)

    # =====================================================================================
    # ============================= PLANE XY ==============================================
    # =====================================================================================
    print('\n' + '=' * 60)
    print('PLANE XY')
    print('=' * 60)

    # per ogni z di R, estraggo i blocchi rejected
    Rf_rejected_z = list()

    print('\nCollecting rejected blocks (cell_info=True, fa<=fa_threshold) for each XY plane:')
    for z in range(shape_R[2]):
        rejected_R_cells = R[:, :, z][rejected_bool[:, :, z]]
        print('z: {} -> rejected_cells: {}'.format(z, rejected_R_cells.shape[0]))
        Rf_rejected_z.append(rejected_R_cells)

    # elaborate every frame of R
    print('\nStart to generate XY plots.')
    for z_R in range(shape_R[2]):

        print('\n----- selected slice in R :      {} on {} - '.format(z_R, shape_R[2]), end='')

        n_rejected = Rf_rejected_z[z_R].shape[0]
        print('->  rejected dots plotted: {}'.format(n_rejected))

        # select central slice of volume for this R plane
        z_vol = int((z_R + 0.5) * shape_P[2])

        # check depth
        if z_vol >= shape_V[2]:
            z_vol = shape_V[2] - 1

        # extract frame: volume[row, col, z] -> XY plane at fixed z
        img_z = normalize(volume[:, :, z_vol], dtype=np.uint8, max_value=maxPixelValue)
        print('    selected slice in Volume : {} on {}'.format(z_vol, shape_V[2]))

        if _equalize:
            img_z = skimage_clahe(img_z, kernel_size=501, clip_limit=clip, nbins=256)

        # extract dot positions only if there are rejected blocks
        if n_rejected > 0:
            init_coord = Rf_rejected_z[z_R]['init_coord']  # rcz=yxz
            centers_z = init_coord + (shape_P / 2)  # rcz=yxz
            yc_z = centers_z[:, :, 0]
            xc_z = centers_z[:, :, 1]
        else:
            xc_z = np.array([])
            yc_z = np.array([])

        # prepare plot for save
        fig, frame_plot = plot_dots_2d_for_save(xc_z, yc_z, img=img_z,
                                                 cmap_image=cmap_image,
                                                 marker_size=marker_size,
                                                 marker_color=marker_color,
                                                 edge_color=edge_color,
                                                 edge_width=edge_width,
                                                 _show_plot=_show_plots)

        # saving image as TIFF
        xy_output_path = make_output_dir(base_path, fa_threshold, stack_prefix, 'XY', _equalize, clip)

        img_name = '{}_XY_z{}'.format(stack_prefix, z_vol)
        save_fig_as_tiff(fig, str(xy_output_path + img_name + '.tiff'))

        if not _show_plots:
            frame_plot.close(fig)

    print('\n ** XY planes finished - OK')

    # =====================================================================================
    # ============================= ORTHOGONAL PLANES =====================================
    # =====================================================================================
    if _orthogonals:

        # =================================================================================
        # ============================= PLANE XZ (fixed row) ==============================
        # =================================================================================
        # Iterare su r di R. Frame del volume: volume[r_vol, :, :] -> shape (col, z)
        # Vogliamo: asse orizzontale = col (X), asse verticale = z (Z)
        # Transpose -> (z, col) cosi imshow mostra righe=z, colonne=col
        # Centro dot: horizontal = col_center, vertical = z_center
        print('\n' + '=' * 60)
        print('PLANE XZ (iterate over rows of R)')
        print('=' * 60)

        Rf_rejected_r = list()

        print('\nCollecting rejected blocks for each XZ plane:')
        for r in range(shape_R[0]):
            rejected_R_cells = R[r, :, :][rejected_bool[r, :, :]]
            print('r: {} -> rejected_cells: {}'.format(r, rejected_R_cells.shape[0]))
            Rf_rejected_r.append(rejected_R_cells)

        print('\nStart to generate XZ plots.')
        for r_R in range(shape_R[0]):

            print('\n----- selected row in R :      {} on {} - '.format(r_R, shape_R[0]), end='')

            n_rejected = Rf_rejected_r[r_R].shape[0]
            print('->  rejected dots plotted: {}'.format(n_rejected))

            # select central row of volume for this R row
            r_vol = int((r_R + 0.5) * shape_P[0])

            # check bound
            if r_vol >= shape_V[0]:
                r_vol = shape_V[0] - 1

            # extract frame: volume[r, col, z] -> shape (col, z)
            img_xz = normalize(volume[r_vol, :, :], dtype=np.uint8, max_value=maxPixelValue)
            print('    selected row in Volume : {} on {}'.format(r_vol, shape_V[0]))

            if _equalize:
                img_xz = skimage_clahe(img_xz, kernel_size=501, clip_limit=clip, nbins=256)

            # Transpose: (col, z) -> (z, col) so imshow rows=z, cols=col
            img_xz_display = img_xz.T

            # extract dot positions only if there are rejected blocks
            if n_rejected > 0:
                init_coord = Rf_rejected_r[r_R]['init_coord']  # rcz=yxz
                centers = init_coord + (shape_P / 2)  # rcz=yxz
                col_centers = centers[:, :, 1]   # col -> horizontal axis (X)
                z_centers = centers[:, :, 2]     # z   -> vertical axis (Z)
            else:
                col_centers = np.array([])
                z_centers = np.array([])

            fig, frame_plot = plot_dots_2d_for_save(col_centers, z_centers, img=img_xz_display,
                                                     cmap_image=cmap_image,
                                                     marker_size=marker_size,
                                                     marker_color=marker_color,
                                                     edge_color=edge_color,
                                                     edge_width=edge_width,
                                                     _show_plot=_show_plots)

            xz_output_path = make_output_dir(base_path, fa_threshold, stack_prefix, 'XZ', _equalize, clip)

            img_name = '{}_XZ_r{}'.format(stack_prefix, r_vol)
            save_fig_as_tiff(fig, str(xz_output_path + img_name + '.tiff'))

            if not _show_plots:
                frame_plot.close(fig)

        print('\n ** XZ planes finished - OK')

        # =================================================================================
        # ============================= PLANE YZ (fixed col) ==============================
        # =================================================================================
        # Iterare su c di R. Frame del volume: volume[:, c_vol, :] -> shape (row, z)
        # Immagine: asse orizzontale = z (Z), asse verticale = row (Y)
        # Centro dot: horizontal = z_center, vertical = row_center
        print('\n' + '=' * 60)
        print('PLANE YZ (iterate over cols of R)')
        print('=' * 60)

        Rf_rejected_c = list()

        print('\nCollecting rejected blocks for each YZ plane:')
        for c in range(shape_R[1]):
            rejected_R_cells = R[:, c, :][rejected_bool[:, c, :]]
            print('c: {} -> rejected_cells: {}'.format(c, rejected_R_cells.shape[0]))
            Rf_rejected_c.append(rejected_R_cells)

        print('\nStart to generate YZ plots.')
        for c_R in range(shape_R[1]):

            print('\n----- selected col in R :      {} on {} - '.format(c_R, shape_R[1]), end='')

            n_rejected = Rf_rejected_c[c_R].shape[0]
            print('->  rejected dots plotted: {}'.format(n_rejected))

            # select central col of volume for this R col
            c_vol = int((c_R + 0.5) * shape_P[1])

            # check bound
            if c_vol >= shape_V[1]:
                c_vol = shape_V[1] - 1

            # extract frame: volume[row, c, z] -> shape (row, z)
            # imshow rows=row, cols=z -> vertical=Y, horizontal=Z
            img_yz = normalize(volume[:, c_vol, :], dtype=np.uint8, max_value=maxPixelValue)
            print('    selected col in Volume : {} on {}'.format(c_vol, shape_V[1]))

            if _equalize:
                img_yz = skimage_clahe(img_yz, kernel_size=501, clip_limit=clip, nbins=256)

            # extract dot positions only if there are rejected blocks
            if n_rejected > 0:
                init_coord = Rf_rejected_c[c_R]['init_coord']  # rcz=yxz
                centers = init_coord + (shape_P / 2)  # rcz=yxz
                row_centers = centers[:, :, 0]   # row -> vertical axis (Y)
                z_centers = centers[:, :, 2]     # z   -> horizontal axis (Z)
            else:
                row_centers = np.array([])
                z_centers = np.array([])

            fig, frame_plot = plot_dots_2d_for_save(z_centers, row_centers, img=img_yz,
                                                     cmap_image=cmap_image,
                                                     marker_size=marker_size,
                                                     marker_color=marker_color,
                                                     edge_color=edge_color,
                                                     edge_width=edge_width,
                                                     _show_plot=_show_plots)

            yz_output_path = make_output_dir(base_path, fa_threshold, stack_prefix, 'YZ', _equalize, clip)

            img_name = '{}_YZ_c{}'.format(stack_prefix, c_vol)
            save_fig_as_tiff(fig, str(yz_output_path + img_name + '.tiff'))

            if not _show_plots:
                frame_plot.close(fig)

        print('\n ** YZ planes finished - OK')

    print('\n ** Process finished - ALL PLANES OK')


def main(args):

    # map color name to RGB tuple
    COLOR_MAP = {
        'red':   (1.0, 0.0, 0.0),
        'green': (0.0, 1.0, 0.0),
        'blue':  (0.0, 0.0, 1.0),
        'cyan':  (0.0, 1.0, 1.0),
        'black': (0.0, 0.0, 0.0),
        'white': (1.0, 1.0, 1.0),
    }
    marker_color = COLOR_MAP[args.marker_color]
    edge_color = COLOR_MAP[args.edge_color]

    plot_rejected_on_frames(
        source_path=args.source_path,
        parameter_filename=args.parameter_filename,
        R_filepath=args.R_path,
        fa_threshold=args.fa,
        _equalize=args.equalize,
        clip=args.clip,
        maxPixelValue=args.maxPixelValue,
        _show_plots=False,
        marker_size=args.marker_size,
        _orthogonals=args.orthogonals,
        marker_color=marker_color,
        edge_color=edge_color,
        edge_width=args.edge_width
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot rejected blocks (low FA) on frames.')
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help='Complete Path of input tiff.')
    parser.add_argument('-p', '--parameter_filename', type=str, required=True,
                        help='Parameter filename (in the same folder of the tiff).')
    parser.add_argument('-r', '--R_path', type=str, required=True,
                        help='Complete path of the R .npy matrix.')
    parser.add_argument('-f', '--fa', type=float, required=True,
                        help='Threshold on Fractional Anisotropy. Blocks with cell_info=True and fa<=threshold are plotted as rejected.')
    parser.add_argument('--no-orthogonals', action='store_false', dest='orthogonals',
                        help='Disable orthogonal planes (XZ, YZ). By default orthogonals are enabled.')
    parser.set_defaults(orthogonals=True)
    parser.add_argument('--equalize', action='store_true', default=False,
                        help='Equalize the image (CLAHE).')
    parser.add_argument('--clip', type=float, default=0.03,
                        help='Clip limit for equalization.')
    parser.add_argument('--maxPixelValue', type=int, default=150,
                        help='Maximum pixel value for normalization.')
    parser.add_argument('--marker_size', type=int, default=40,
                        help='Size of the red dot markers.')
    parser.add_argument('--marker_color', type=str, default='red',
                        choices=['red', 'green', 'blue', 'cyan', 'black', 'white'],
                        help='Fill color of the dot markers (default: red).')
    parser.add_argument('--edge_color', type=str, default='black',
                        choices=['red', 'green', 'blue', 'cyan', 'black', 'white'],
                        help='Edge color of the dot markers (default: black).')
    parser.add_argument('--edge_width', type=float, default=2.0,
                        help='Edge width of the dot markers (default: 2.0).')
    args = parser.parse_args()
    main(args)