import numpy as np
import os
import time
from PIL import Image
from io import BytesIO

from scipy import ndimage
# from zetastitcher import InputFile
from tifffile import imread as imread

from custom_tool_kit import create_coord_by_iter, all_words_in_txt, search_value_in_txt
from custom_image_base_tool import print_info, normalize

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
plt.rcParams['figure.figsize']=(14,14)


def float_to_color(values, color_map=cm.viridis, _print_info=False):
    nz = mcolors.Normalize()
    nz.autoscale(values)
    if _print_info: print_info(values)
    colormap = color_map
    return colormap(nz(values))[:]


def chaos_normalizer(values, isolated_value=-1, assign='max'):
    new_vaues = np.copy(values)

    maxv = np.max(values[values >= 0])
    minv = np.min(values[values >= 0])

    new_vaues[new_vaues == isolated_value] = (maxv if (assign == 'max') else minv)

    return (new_vaues - minv) / (maxv - minv)


# color_map STRINGs
COL_XYANGLE = 0
COL_PARAM = 1
COL_ZETA = 2

# image_format STRINGS
IMG_EPS = 'EPS'
IMG_TIFF = 'TIFF'
IMG_SVG = 'SVG'


class Param:
    ID_BLOCK = 'id_block'  # unique identifier of block
    CELL_INFO = 'cell_info'  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
    ORIENT_INFO = 'orient_info'  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
    CELL_RATIO = 'cell_ratio'  # ratio between cell voxel and all voxel of block
    INIT_COORD = 'init_coord'   # absolute coord of voxel block[0,0,0] in Volume
    EW = 'ew'   # descending ordered eigenvalues.
    EV = 'ev'   # column ev[:,i] is the eigenvector of the eigenvalue w[i].
    STRENGHT = 'strenght'   # parametro forza del gradiente (w1 .=. w2 .=. w3)
    CILINDRICAL_DIM = 'cilindrical_dim'  # dimensionalità forma cilindrica (w1 .=. w2 >> w3)
    PLANAR_DIM = 'planar_dim'  # dimensionalità forma planare (w1 >> w2 .=. w3)
    FA = 'fa'  # fractional anisotropy (0-> isotropic, 1-> max anisotropy
    SUM_SHAPES = 'sum_shapes'  # def as: (fa + planar_dim + cilindrical_dim)
    LOCAL_DISARRAY = 'local_disarray'   # local_disarray
    LOCAL_DISARRAY_W = 'local_disarray_w'  # local_disarray using FA as weight for the versors


class Bcolors:
    V = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# CALL
# plot_quiver_2d_for_save(xc_z, yc_z, -yq_z, xq_z, color_values=color_values, img=img_z,
#                                            title='R:{} - Z:{}'.format(z_R, z_vol),
#                                            scale_units='xy', scale=scale, pivot='middle',
#                                            real=False, width=3,


def plot_quiver_2d_for_save(x0_c, x1_c, x0_q, x1_q, img=None, shape=None, origin='upper', title='',
                            color_values='r', scale_units='xy', scale=1., width=2, pivot='middle', real=False,
                            color_map=None, cmap_image='gray', _show_plot=False):
    # xc, yc -> array of coordinates of TAIL of the Arrow
    # xq, yq -> array of components of quiver (Head of artow relative to tail)
    # real -> plot quiver with real xy dimension

    fig = plt.figure(tight_layout=True)

    dpi = fig.get_dpi()
    # plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    print("Genarated a Fig with dip: ", dpi)

    # plotto tiff sotto i quiver
    if img is not None:
        plt.imshow(img, origin=origin, cmap=cmap_image, alpha=0.9)
        if shape is None:
            shape = img.shape
    else:
        color = 'k'  # background uniforme

    # rimuovo gli assi per ridurre la cornice bianca quando salvo l'immagine
    ax = plt.gca()
    ax.set_axis_off()
    fig.add_axes(ax)

    # plot all quivers
    if real:
        quiv = plt.quiver(x0_c, x1_c, x0_q, x1_q, headlength=0, headwidth=1, color=color)
    else:
        quiv = plt.quiver(x0_c, x1_c, x0_q, x1_q, color_values,
                          cmap=color_map,
                          units='xy', headwidth=1, headlength=0, width=width,
                          scale_units=scale_units, scale=scale, pivot=pivot)

    #     if shape is not None:
    #         plt.xlim((0, shape[1]))
    #         plt.ylim((0, shape[0]))  # remember Image standard system : x -> col, y -> row
    #         fig.set_size_inches(shape[1]/dpi, shape[0]/dpi)
    #         print('shape_inside: ', shape)
    #         print('set_size_inches : ', shape[1]/dpi, shape[0]/dpi)

    # if color_map is not None:
    # plt.colorbar(quiv,cmap=color_map)

    # image standard
    # plt.gca().invert_yaxis()

    if _show_plot:
        plt.show()
    return fig, plt


def plot_vectors_on_frames(home_path='/home', acquisition_folder='', stack_name='',
                           parameter_filename='parameters.txt', _save_all_frames=False, _save_manual_fig=False,
                           _save_all_R_planes=True, _plot_img=True, _plot_on_MIP=False, _save_on_MIP=False,
                           manual_z_R_selection=None, color_to_use=COL_ZETA, param_for_color=None,
                           color_map=cm.plasma, cmap_used='plasma', _blur_color_par=False, _image_white=False,
                           img_format=IMG_TIFF, ev_index=2, info_to_plot='none', scale=0.07, _show_plots=True):

    base_path = os.path.join(home_path, acquisition_folder)
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

    # Evaluate image analysis characteristics

    # ratio between pixel size in z and xy
    ps_ratio = parameters['px_size_z'] / parameters['px_size_xy']

    # analysis block dimension in z-axis
    num_of_slices_P = int(parameters['roi_xy_pix'] / ps_ratio)

    # Parameters of Acquisition System:
    res_z = parameters['px_size_z']
    res_xy = parameters['px_size_xy']
    resolution_factor = res_z / res_xy
    print('Pixel size (Real) in XY : {} um'.format(res_xy))
    print('Pixel size (Real) in Z : {} um'.format(res_z))

    # dimension of analysis block (parallelogram)
    block_side = row_P = col_P = int(parameters['roi_xy_pix'])
    shape_P = np.array((row_P, col_P, num_of_slices_P)).astype(np.int32)
    print('Dimension of Parallelepiped : {} pixel'.format(shape_P))
    print('Dimension of Parallelepiped : {} um'.format(np.array(shape_P) * np.array([res_xy, res_xy, res_z])))

    # load R
    stack_prefix = stack_name.split('.')[0]
    R_filename = 'R_' + stack_prefix + '_' + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.npy'
    print(R_filename)
    R_filepath = os.path.join(base_path, R_filename)
    R = np.load(R_filepath)
    shape_R = R.shape
    print('shape_R: ', shape_R)

    if color_to_use == COL_PARAM:
        # Normalize scalar parameter
        R[param_for_color] = normalize(R[param_for_color].copy(),
                                       max_value=1.0,
                                       dtype=R[param_for_color].dtype)

    # OPEN STACK solo se servono le immagini
    # if plot_img or plot_on_MIP:
    _open_stack = True if (_plot_img or _plot_on_MIP) else False
    # --------------------

    # OPEN STACK
    if _open_stack:

        t0 = time.time()
        source_path = os.path.join(base_path, stack_name)

        # infile = InputFile(source_path)
        # volume = infile.whole()
        volume = imread(source_path)

        # NB - Attenzione gestione assi: InputFile/tifffile: (z,y,x) ---> ME: (r,c,z)=(y,x,z) ---> move axis like above:
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

    else:
        shape_V = shape_R * shape_P

    print_info(volume, text='Volume')

    # auto set parameters and advertising
    if _save_all_frames:
        print(' ** ATTENTION : WAIT NOTIFICATION of END OF PROCESS \n')

    if _image_white:
        cmap_image = 'gray_r'
    else:
        cmap_image = 'gray'

    # ==== BLOCKS EXTRACTION FROM R AT EVERY Z PLANES =====

    # for each z in R, extract only cubes with freq info and their 'param_to_plot' values, and put in two list

    # extract bool maps of valid blocks
    orient_info_bool = R[
        'orient_info']  # TODO -> pensare a 'mappatura' di grane 'valide e non valide' tipo segmentazione

    # per ogni z di R, estraggo i cubi validi (utilizzando la mappa booleana orient_info_bool)
    # e li inserisco in una lista (Rf_z) alla z corrispondente
    Rf_z = list()
    param_to_plot_2d_list = list()

    print('\nCollecting valid orientation vectors for each plane:')
    for z in range(shape_R[2]):

        # extract map of valid cells
        print('z: {} -> valid_cells: {}'.format(z, R[:, :, z][orient_info_bool[:, :, z]].shape[0]))
        Rf_z.append(R[:, :, z][orient_info_bool[:, :, z]])  # R[allR, allrC, Z][bool_map_of_valid_blocks[allR, allC, Z]]

        # extract param_to_plt
        if _blur_color_par:
            # extract values
            par_matrix = R[:, :, z][param_for_color]
            # blurring
            par_blurred = ndimage.gaussian_filter(par_matrix.astype(np.float32), sigma=1.5).astype(np.float16)
            param_to_plot_2d_list.append(par_blurred[orient_info_bool[:, :, z]])

        else:
            param_to_plot_2d_list.append(R[:, :, z][orient_info_bool[:, :, z]][param_for_color])

    # =============================== CREATING DATA FOR PLOT () ===============================

    # define real z to plot in volume system (effective slices = tiff frames)
    # z_R --> in 'R' riferiment system
    # z_vol --> in 'volume' riferiment system

    if _save_all_frames or _save_all_R_planes:
        z_R_to_check = list(range(shape_R[2]))  # all z in R matrix
    else:
        z_R_to_check = list(manual_z_R_selection)  # manual selection

    # ensure there is at least one valid element to plot for each plane.
    # if not, that z is not inserted in the list 'z_R_to_plot'
    z_R_to_plot = list()
    for z_R in z_R_to_check:
        valid_elem = R[:, :, z_R][orient_info_bool[:, :, z_R]].shape[0]
        if valid_elem > 0:
            z_R_to_plot.append(z_R)
        else:
            print('z_R = {} has {} valid elements -> discarted.'.format(z_R, valid_elem))

    # elaborate every frames of R selected for the plot
    print('\nStart to generate plot of each plane.')
    for z_R in z_R_to_plot:

        if not _plot_on_MIP:
            print(' - selected slice in R :      {} on {} - '.format(z_R, shape_R[2]), end='')

        # extract position coordinate of every block
        init_coord = Rf_z[z_R]['init_coord']  # rcz=yxz

        # extract quiver position
        centers_z = init_coord + (shape_P / 2)  # rcz=yxz
        yc_z = centers_z[:, :, 0]
        xc_z = centers_z[:, :, 1]
        zc_z = centers_z[:, :, 2]

        # extract quiver component
        # E QUI NON SO SE SONO DAVVERO XYZ O YXZ
        quiver_z = Rf_z[z_R]['ev'][:, :, ev_index]  # [all cells, all components, 'ev_index = (0, 1 or 2)']
        yq_z = quiver_z[:, 0]  # all y component
        xq_z = quiver_z[:, 1]  # all x component
        zq_z = quiver_z[:, 2]  # all z component

        # extract quiver ids
        ids = Rf_z[z_R]['id_block']

        # valuer from R:
        # param_to_plot_z = Rf_z[z_R][param_for_plot]
        # valuer from param_to_plot_2d_list
        param_to_plot_z = param_to_plot_2d_list[z_R]
        # print('Plotted values from param_to_plot_2d_list[{}]:'.format(z_R))
        # print_info(param_to_plot_z)

        if not _plot_on_MIP:
            print('->  vectors plotted: {}'.format(len(Rf_z[z_R])))

        # param_to_plot_z = chaos_normalizer(param_to_plot_z, isolated_value=-1, assign='max')

        # create color map
        if color_to_use is COL_PARAM:
            color_values = param_to_plot_z
        elif color_to_use is COL_XYANGLE:
            color_values = (np.arctan2(yq_z, xq_z) % np.pi)[:]  # [:] because has shape (n_blocks, 1)
            color_values = 2 * np.abs((color_values / color_values.max()) - 0.5)  # simmetric map of angles
        elif color_to_use is COL_ZETA:
            color_values = normalize(np.abs(zq_z), max_value=1.0, dtype=np.float64)  # norm between [0,1]
            # zeta no simmetrica?
            # color_values = 2 * np.abs((color_values / color_values.max()) - 0.5)  # simmetric

        # maps color_values into scalar of 'color_map' matplotlib color map
        colors_2d = float_to_color(values=color_values, color_map=color_map)

        # create range of real frames of volume to plot
        if _save_all_frames:
            slices_to_plot = range(z_R * shape_P[2], (z_R + 1) * shape_P[2])  # plotta tutte le 8 slide per ogni cubetto
        elif _save_all_R_planes or _save_manual_fig:
            slices_to_plot = [((z_R + 1 / 2) * shape_P[2]).astype(int)]  # plotta solo la slide centrale

        if _plot_on_MIP:

            print('     plot on MIP')
            MIP = normalize(np.max(volume, axis=2), dtype=np.uint8, max_value=100)
            # shape_fig = MIP.shape  # image shape

            width = 5
            # prepare plot for save and/or plot image
            fig, mip_plot = plot_quiver_2d_for_save(xc_z, yc_z, -xq_z, yq_z, color_values=color_values, img=MIP,
                                                    title='MIP - ev: {0}th'.format(ev_index),
                                                    scale_units='xy', scale=scale, pivot='middle',
                                                    real=False, width=width,
                                                    color_map=color_map, cmap_image=cmap_image,
                                                    _show_plot=_show_plots)
            if _show_plots:
                mip_plot.show()

            if _save_on_MIP:
                # TIFF
                # (1) save the image in memory in PNG format
                png1 = BytesIO()
                fig.savefig(png1, format='png')
                # (2) load this image into PIL
                png2 = Image.open(png1)
                # (3) save as TIFF
                png2.save(os.path.join(base_path, 'quiver_on_MIP.tiff'))
                png1.close()
                print("saved fig MIP in ", base_path)

        else:
            # plot on the single frames
            for z_vol in slices_to_plot:
                # select z frame
                if _open_stack and _plot_img:

                    # check depth
                    if z_vol >= shape_V[2]:
                        # take the last one
                        z_vol = shape_V[2] - 1

                    # extract frame
                    img_z = normalize(volume[:, :, z_vol], dtype=np.uint8, max_value=100)
                    print('     selected slice in Volume : {} on {} \n'.format(z_vol, shape_V[2]))

                    # if _equalize :
                    # TODO
                else:
                    img_z = None  # raise error

                # shape_fig = volume[:, :, z_vol].shape  # image shape

                # ATTENZIONE   HO AGGIUNTO IL  MENO  ALLA  X   <-----------------------! ! ! !

                # prepare plot for save and/or plot image
                fig, frame_plot = plot_quiver_2d_for_save(xc_z, yc_z, -xq_z, yq_z, color_values=color_values, img=img_z,
                                                          title='R:{0} - Z:{1} - ev: {2}th'.format(z_R, z_vol, ev_index),
                                                          scale_units='xy', scale=scale, pivot='middle',
                                                          real=False, width=3,
                                                          color_map=color_map, cmap_image=cmap_image,
                                                          _show_plot=_show_plots)

                # [if selected] write selected info over the image for every vector
                if info_to_plot != 'none':
                    if info_to_plot == 'evZ':
                        to_write = Rf_z[z_R]['ev'][:, 2, ev_index]  # [all cells, z-comp, ev_index(default: 2th)]
                    if info_to_plot == 'evZerr':
                        to_write = 100 * (1 - np.abs(
                            Rf_z[z_R]['ev'][:, 2, ev_index]))  # [all cells, z-comp, ev_index(default: 2th)]
                    if info_to_plot == 'ids':
                        to_write = ids
                    if info_to_plot == 'ew':
                        to_write = Rf_z[z_R][info_to_plot][:, 0, ev_index]  # [all_blocks, {è in riga}, indice_ew]
                    if info_to_plot == 'ErrSumShapes':
                        to_write = 10 * (Rf_z[z_R]['sum_shapes'])
                    if info_to_plot in ['cell_ratio', 'cilindrical_dim', 'planar_dim', 'strenght', 'fa', 'sum_shapes']:
                        to_write = Rf_z[z_R][info_to_plot]

                    # debug
                    print('-------------------------------')
                    print(to_write.shape)
                    print(to_write)
                    print('-------------------------------')

                    for val, pos in zip(to_write, init_coord):
                        r = pos[0][0]  # shape is (1, rcz)
                        c = pos[0][1]  # shape is (1, rcz)
                        if info_to_plot == 'ids':
                            string = str(val)
                        elif info_to_plot == 'strenght':
                            string = '{0:.0f}'.format(val)
                        else:
                            string = '{0:.1f}'.format(val)
                        frame_plot.text(c, r, string, color='r', fontsize=10)
                        frame_plot.title(R_filename + ' ' + info_to_plot + ' ' + R_filename)

                    if _show_plots:
                        frame_plot.show()
                    else:
                        frame_plot.close(fig)

                # saving images?
                if _save_all_frames or _save_manual_fig or _save_all_R_planes:
                    quiver_path = os.path.join(base_path, 'quiver_{}_{}_{}_{}/'.
                                               format(info_to_plot, img_format,
                                                      R_filename.split('.')[0],
                                                      cmap_used))  # create path where save images
                    # check if it exist
                    if not os.path.isdir(quiver_path):
                        os.mkdir(quiver_path)

                    # create img name
                    img_name = str(z_vol) + 'ew{}'.format(ev_index)

                    if img_format == IMG_SVG:
                        # formato SVG -> puoi decidere dopo la risoluzione aprendolo con fiji
                        fig.savefig(str(quiver_path + img_name + '.svg'), format='svg',
                                    dpi=1200, bbox_inches='tight', pad_inches=0)

                    elif img_format == IMG_EPS:
                        # formato EPS buono per latex (latex lo converte automat. in pdf)
                        fig.savefig(str(quiver_path + img_name + '_black.eps'), format='eps', dpi=400,
                                    bbox_inches='tight', pad_inches=0)

                    elif img_format == IMG_TIFF:
                        # (1) save the image in memory in PNG format
                        png1 = BytesIO()
                        fig.savefig(png1, format='png')
                        # (2) load this image into PIL
                        png2 = Image.open(png1)
                        # (3) save as TIFF
                        png2.save((str(quiver_path + img_name + '.tiff')))
                        png1.close()


                    if not _show_plots:
                        frame_plot.close(fig)

    print('\n ** Process finished - OK')
    return quiver_path


# ================================================================================================================
# =============================== PLOT PARAMETERS ================================================================
# ================================================================================================================

# savings - choice only ONE mode!
_save_all_frames   = False  # save every frame of tiff file with the corrispondent R depth vectors (VERY time expensive)
_save_manual_fig   = False  # save only manual selected depth in 'img_format' format selected (time expensive)
_save_all_R_planes = True  # save one images for every R planes

_plot_img    = True  # plot the tiff frames under the vectors
_plot_on_MIP = False
_save_on_MIP = False
_show_plots  = False  # display the plots on the screen

# used only if save_manual_fig is True
# manual_z_R_selection = range(1, 65, 3)
manual_z_R_selection = [0, 1, 2, 3, 4]

# choice what plot and what color_map
color_to_use = COL_ZETA  # COL_XYANGLE, COL_PARAM, COL_ZETA
param_for_color = Param.FA  # choice from class Param -> [used only if color_to_use = COL_PARAM]
color_map, cmap_used = cm.plasma, 'plasma'
_blur_color_par = False  # gaussian blur on the color matrix

# black or whit# equalize image - TODO
# _equalize = True

# image background
_image_white = False  # True: LDG gray_R; False: LDS gray (normal)

# equalize image - TODO
# _equalize = True

# image format to save (is 'save_fig' is selected):
img_format = IMG_TIFF

# Select index of eigenvector to plot (0: max, 2:min)
ev_index = 2

# quivers info to plot over images
info_to_plot = 'none'  # option: 'ids', 'cell_ratio','cilindrical_dim','planar_dim',
#                                'strenght', 'none', 'local_disarray', 'ew', 'fa', 'ev', 'evZerr',
#                                'sum_shapes', 'ErrSumShapes'
# NB: 'evZ' è la componente Z dell'autovettore orientazione
# NB: evZerr p una misura di "scostamento" dal vettore unitario:
# evZerr = 100 * (1 - evZ) -> così se evZ è 0.980, mi plotta 2.0, se 0.873, mi plotta 12.7
# serve per vedere se vettore orientazione è troppo parallelo all'asse ottico <=> misura troppo sfuocata sul piano XY causa imaging
# NB: sum_shapes = fa + cilindrical_dim + planar_dim
# NB: ErrSumShapes = 10 * (1 - np.abs(sum_shapes))

# scale of quiver lenght
# scale = 0.05 # val più piccolo, quiver + lunghi
scale = 0.07  # val più piccolo, quiver + lunghi

# ================================================================================================================
# =============================== END PARAMETERS =================================================================
# ================================================================================================================

# ------------------------------------------------------- [  A  ]----------------------------------------------------------------
# ################################################  COMMENT OR DECOMMENT ##########################################################
# ########################################  USE THAT AS INPUT FOR ONLY ONE SAMPLE ################################################

# home_path = r'/home/francesco/LENS/Whole_Heart/mesoSPIM_AC/taskForce/Disarray/Samples/analyzed/data/CTRL'
home_path = r'/home/francesco/LENS/Whole_Heart/mesoSPIM_AC/taskForce/Disarray/Samples/analyzed_th80%/data/PATHOL'
acquisition_folder = r'7_months/N08'
stack_name = r'N08.tif'
parameter_filename = 'parameters_TaskForce.txt'

plot_vectors_on_frames(home_path=home_path, acquisition_folder=acquisition_folder, stack_name=stack_name,
                       parameter_filename=parameter_filename, _save_all_frames=_save_all_frames, _save_manual_fig=_save_manual_fig,
                       _save_all_R_planes=_save_all_R_planes, _plot_img=_plot_img, _plot_on_MIP=_plot_on_MIP, _save_on_MIP=_save_on_MIP,
                       manual_z_R_selection=manual_z_R_selection, color_to_use=color_to_use, param_for_color=param_for_color,
                       color_map=color_map, cmap_used=cmap_used, _blur_color_par=_blur_color_par, _image_white=_image_white,
                       img_format=img_format, ev_index=ev_index, info_to_plot=info_to_plot, scale=scale, _show_plots=_show_plots)
# #################################################################################################################################



# ------------------------------------------------------- [  B  ]----------------------------------------------------------------
################################################  COMMENT OR DECOMMENT ##########################################################
########################################  USE THAT AS INPUT FOR LIST OF SAMPLES  ################################################
#
# home_path = r'/home/francesco/LENS/Whole_Heart/mesoSPIM_AC/taskForce/Disarray/Samples/analyzed/data/{}'  # format(group)
# groups = ['CTRL', 'PATHOL']
# parameter_filename = 'parameters_TaskForce.txt'
#
# # start the plot_on_frames function for all samples of all groups
# for group in groups:
#     grouppath = home_path.format(group)
#
#     # list samples there
#     samplesnames = os.listdir(grouppath)
#     print('Start to prepare plots for the following samples:', samplesnames)
#
#     for sname in samplesnames:
#         acquisition_folder = r'{}'.format(sname)
#         stack_name = r'{}.tif'.format(sname)
#
#         print(acquisition_folder)
#         print(stack_name)
#
#         print(Bcolors.OKBLUE + 'Starting to plot vectors on frames for sample: {}'.format(sname) + Bcolors.ENDC)
#
#         try:
#             plot_vectors_on_frames(home_path=grouppath, acquisition_folder=acquisition_folder, stack_name=stack_name,
#                                    parameter_filename=parameter_filename, save_all_frames=save_all_frames, save_manual_fig=save_manual_fig,
#                                    save_all_R_planes=save_all_R_planes, plot_img=plot_img, plot_on_MIP=plot_on_MIP, save_on_MIP=save_on_MIP,
#                                    manual_z_R_selection=manual_z_R_selection, color_to_use=color_to_use, param_for_color=param_for_color,
#                                    color_map=color_map, cmap_used=cmap_used, _blur_color_par=_blur_color_par, image_white=image_white,
#                                    img_format=img_format, ev_index=ev_index, info_to_plot=info_to_plot, scale=scale, _show_plots=_show_plots)
#         except:
#             print(Bcolors.WARNING + 'something is wrong - plots not created. Skip to the next samples.' + Bcolors.ENDC)
# #################################################################################################################################
