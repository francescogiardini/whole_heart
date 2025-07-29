# system
import os
import argparse

# general
import numpy as np
from numpy import linalg as LA
from scipy import stats as scipy_stats


# image
from skimage.filters import gaussian
from skimage import transform
from scipy import ndimage as ndi

# custom codes
from custom_image_base_tool import normalize
from custom_tool_kit import Bcolors, create_coord_by_iter, create_slice_coordinate

# ###################################    CLASSES    #########################################à


class CONST:
    INV = -1


class Cell_Ratio_mode:
    NON_ZERO_RATIO = 0.0
    MEAN = 1


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
    MEAN_INTENSITY = 'mean_intensity'  # mean intensity of the block


class Stat:
    # statistics
    N_VALID_VALUES = 'n_valid_values'
    MIN = 'min'
    MAX = 'max'
    AVG = 'avg'  # statistics mean
    STD = 'std'
    SEM = 'sem'  # standard error of the mean
    MEDIAN = 'median'  # statistics median
    MODE = 'mode'  # statistics mode
    MODALITY = 'modality'  # for arithmetic or weighted


class Mode:
    # to select which type of local disarray is:
    # estimated with arithmetic average
    # or
    # estimated with weighted average
    ARITH = 'arithmetic'
    WEIGHT = 'weighted'


# ##############################################    METHODS     ################################################


def downsample_2_zeta_resolution(vol, px_size_xy, px_size_z, sigma):
    # Opera uno smoothing in xy per portare la larghezza della PSF in xy UGUALE alla larghezza della PSF in Zeta.
    # Poi fa downsampling per uniformare anche la pixel size.
    # L'immagine prodotta avrà le stesse caratteristiche delle sezioni XZ e YZ,
    # in modo da rendere il volume a risoluzione isotropa e
    # l'operazione di derivazione nei tre assi priva di sbilanciamenti.
    #
    # vol is np.uint8
    # downsampled is np.float32

    # estimate new isotropic sizes
    resize_ratio = px_size_xy / px_size_z
    resized_dim = int(resize_ratio * vol.shape[0])

    downsampled = np.zeros((resized_dim, resized_dim, vol.shape[2]), dtype=np.float32)

    for z in range(vol.shape[2]):
        # convolve with gaussian kernel for smooth
        blurred = gaussian(image=vol[:, :, z], sigma=sigma, mode='reflect')

        # resize and save into new downsampled volume
        downsampled[:, :, z] = normalize(
            img=transform.resize(image=blurred, output_shape=(resized_dim, resized_dim), mode='reflect'),
            dtype=np.float32)

    return downsampled


def sigma_for_uniform_resolution(FWHM_xy, FWHM_z, px_size_xy):
    # Legend: SD: Standard Deviation = sigma; Var: Variance = sigma**2
    #
    # For uniform resolution over 3 axes, xy plane are blurred by a gaussian kernel with sigma = sigma_s.
    # Sigma_s depends by resolution (FWHM) in xy and Z
    # f_xy convoled f_s = f_z, where f is Gaussian profile of 1d PSF.
    # so, sigma_xy**2 + sigma_s**2 = sigma_z**2
    #
    # That function calculate sigma_s (in micron) by converting FWHM_xy and FWHM_z in sigma_xy and sigma_z,
    # and calculate sigma_s in pixel by pixel sixe in xy.

    # estimate variance (sigma**2) by FWHM (in micron)
    sigma2_xy = FWHM_xy ** 2 / (8 * np.log(2))
    sigma2_z = FWHM_z ** 2 / (8 * np.log(2))

    # estimate variance (sigma_s**2) of gaussian kernel (in micron)
    sigma2_s = np.abs(sigma2_z - sigma2_xy)

    # estimate SD of Gaussian Kernel (in micron)
    sigma_s = np.sqrt(sigma2_s)

    # return SD in pixel
    return sigma_s / px_size_xy


def turn_in_upper_semisphere(v, axis=1):
    # if versor v is in the "axis < 0" semisphere,
    # turn the versor in the opposite direction
    if v[axis] >= 0:
        v_rot = np.copy(v)
    else:
        v_rot = -np.copy(v)
    return v_rot


def structure_tensor_analysis_3d(vol, _rotation=False):
    # HERE I USE (X, Y, Z) convention
    #
    # Structure Tensor definita così:
    # ogni elemento (i,j) dello ST sarà la media della matrice IiIj
    # create ST matrix with mean of Ixx, Ixy and Iyy
    #
    # |IxIx  IxIy  IxIz|          |mean(IxIx)  mean(IxIy)  mean(IxIz)|
    # |IyIx  IyIy  IyIz|     ->   |mean(IyIx)  mean(IyIy)  mean(IyIz)|
    # |IzIx  IzIy  IzIz|     ->   |mean(IzIx)  mean(IzIy)  mean(IzIz)|
    #   (3 x 3 x m x m)      -->                 (3 x 3)
    #    m : ROI side
    #
    # INPUT: vol (rcz = yxz)
    # OUTPUT: (eigenvalues, eigenvectors, shape_parameters)
    # eigenvalues and eigenvectors are oredered by descending eigenvalues values.
    # w[0] > w[1] >...
    # NB : the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    # if _rotation is True, each eigenvector are turned in the 'axis > 0' semisphere

    # Compute Gradient over x y z directions
    gx, gy, gz = np.gradient(vol)

    # compute second order moment of gradient
    Ixx = ndi.gaussian_filter(gx * gx, sigma=1, mode='constant', cval=0)
    Ixy = ndi.gaussian_filter(gx * gy, sigma=1, mode='constant', cval=0)
    Ixz = ndi.gaussian_filter(gx * gz, sigma=1, mode='constant', cval=0)
    # Iyx = Ixy, simmetric
    Iyy = ndi.gaussian_filter(gy * gy, sigma=1, mode='constant', cval=0)
    Iyz = ndi.gaussian_filter(gy * gz, sigma=1, mode='constant', cval=0)
    # Izx = Ixz, simmetric
    # Izy = Iyz, simmetric
    Izz = ndi.gaussian_filter(gz * gz, sigma=1, mode='constant', cval=0)

    # create ST matrix with mean of Ixx, Ixy and Iyy
    # |IxIx  IxIy  IxIz|          |mean(IxIx)  mean(IxIy)  mean(IxIz)|
    # |IyIx  IyIy  IyIz|     ->   |mean(IyIx)  mean(IyIy)  mean(IyIz)|
    # |IzIx  IzIy  IzIz|     ->   |mean(IzIx)  mean(IzIy)  mean(IzIz)|
    #   (3 x 3 x m x m)      -->                 (3 x 3)
    ST = np.array([[np.mean(Ixx), np.mean(Ixy), np.mean(Ixz)],
                   [np.mean(Ixy), np.mean(Iyy), np.mean(Iyz)],
                   [np.mean(Ixz), np.mean(Iyz), np.mean(Izz)]])

    # eigenvalues and eigenvectors decomposition
    w, v = LA.eig(ST)
    # NB : the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

    # ordino autovalori e autovettori t.c. w0 > w1 > w2
    order = np.argsort(w)[::-1]  # decrescent order
    w = np.copy(w[order])
    v = np.copy(v[:, order])

    if _rotation:
        # sposta autovettori sulla semisfera con 'axis=1' > 0
        ev = np.zeros_like(v)
        for ev_idx in range(v.shape[1]):
            ev[:, ev_idx] = turn_in_upper_semisphere(v[:, ev_idx], axis=1)
    else:
        ev = np.copy(v)

    # parameri di forma
    shape_parameters = dict()

    # calcolo fractional anisotropy (0 isotropy -> 1 anisotropy)
    denominator = np.sqrt(np.sum(w ** 2))
    if denominator == 0:
        # if denominator is 0, all eigenvalues are 0, so the FA is 0
        shape_parameters['fa'] = 0.0
    else:
        shape_parameters['fa'] = np.sqrt(1/2) * (
            np.sqrt((w[0] - w[1]) ** 2 + (w[1] - w[2]) ** 2 + (w[2] - w[0]) ** 2) / denominator
        )

    # calcolo parametro forza del gradiente (per distinguere contenuto da sfondo)
    # (w1 .=. w2 .=. w3)
    shape_parameters['strenght'] = np.sqrt(np.sum(w))

    # calcolo dimensionalità forma cilindrica (w1 .=. w2 >> w3)
    denominator = (w[1] + w[2])
    if denominator == 0:
        shape_parameters['cilindrical_dim'] = 0.0
    else:
        shape_parameters['cilindrical_dim'] = (w[1] - w[2]) / denominator

    # calcolo dimensionalità forma planare (w1 >> w2 .=. w3)
    denominator = (w[0] + w[1])
    if denominator == 0:
        shape_parameters['planar_dim'] = 0.0
    else:
        shape_parameters['planar_dim'] = (w[0] - w[1]) / denominator

    # parametro def da me che soma i fattori di forma
    shape_parameters['sum_shapes'] = shape_parameters['planar_dim'] + shape_parameters['cilindrical_dim'] + shape_parameters['fa']

    return (w, ev, shape_parameters)


def create_R(shape_V, shape_P):
    '''
    Define Results Matrix - 'R'

    :param shape_V: shape of entire Volume of data
    :param shape_P: shape of parallelepipeds extracted for orientation analysis
    :return: empty Result matrix 'R'
    '''

    if any(shape_V > shape_P):
        # shape
        shape_R = np.ceil(shape_V / shape_P).astype(np.int)

        # define empty Results matrix
        total_num_of_cells = np.prod(shape_R)
        R = np.zeros(
            total_num_of_cells,
            dtype=[(Param.ID_BLOCK, np.int64),  # unique identifier of block
                   (Param.CELL_INFO, bool),  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
                   (Param.ORIENT_INFO, bool),  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
                   (Param.CELL_RATIO, np.float16),  # ratio between cell voxel and all voxel of block
                   (Param.INIT_COORD, np.int32, (1, 3)),  # absolute coord of voxel block[0,0,0] in Volume
                   (Param.EW, np.float32, (1, 3)),  # descending ordered eigenvalues.
                   (Param.EV, np.float32, (3, 3)),  # column ev[:,i] is the eigenvector of the eigenvalue w[i].
                   (Param.STRENGHT, np.float16),  # parametro forza del gradiente (w1 .=. w2 .=. w3)
                   (Param.CILINDRICAL_DIM, np.float16),  # dimensionalità forma cilindrica (w1 .=. w2 >> w3)
                   (Param.PLANAR_DIM, np.float16),  # dimensionalità forma planare (w1 >> w2 .=. w3)
                   (Param.FA, np.float16),  # fractional anisotropy (0-> isotropic, 1-> max anisotropy)
                   (Param.MEAN_INTENSITY, np.float16),  # mean intensity of the block
                   (Param.SUM_SHAPES, np.float16),  # fa + planar_dim + cilindrical_dim
                   (Param.LOCAL_DISARRAY, np.float16),  # local_disarray
                   (Param.LOCAL_DISARRAY_W, np.float16)  # local_disarray using FA as weight for the versors
                   ]
        ).reshape(shape_R)

        # initialize mask of info to False
        R[:, :, :][Param.CELL_INFO] = False
        R[:, :, :][Param.ORIENT_INFO] = False

        print('Dimension of Result Matrix:', R.shape)
        return R, shape_R
    else:
        raise ValueError(' Data array dimension is smaller than dimension of one parallelepiped. \n'
                         ' Ensure which data is loaded or modify analysis parameter')


def create_stats_matrix(shape):
    """
    Define a Matrix with the same structure of R parameters'
    """
    matrix = np.zeros(
        np.prod(shape),
        dtype=[(Stat.N_VALID_VALUES, np.int32),
               (Stat.MIN, np.float16),
               (Stat.MAX, np.float16),
               (Stat.AVG, np.float16),
               (Stat.STD, np.float16),
               (Stat.SEM, np.float16),
               (Stat.MODE, np.float16),
               (Stat.MODALITY, np.float16),
               ]
    ).reshape(shape)
    return matrix


def stats_on_structured_data(input, param, w, invalid_value=None, invalid_par=None, _verb=False):
    """
    :param input: matrix with a structure like R
    :param param: parameter selected
    :param w: weights
    :param invalid_value: the value that indicate an invalid elements; it's used to create the valid_mask
    :param invalid_par: if passed, invalid_mask is created using invalid_value on R[invalid_par] instead on R[param]
    :param _verb:
    :return: structure with statistic results
    """

    if input is not None and param is not None:

        # create mask of valid values (True if valid, False if invalid)
        if invalid_value is not None:
            if invalid_par is not None:
                valid_mask = input[invalid_par] != invalid_value
            else:
                valid_mask = input[param] != invalid_value

        # estimate statistics on selected values
        results = statistics_base(input[param], w=w, valid_mask=valid_mask, _verb=_verb)
        return results
    else:
        return None


def statistics_base(x, w=None, valid_mask=None, invalid_value=None, _verb=False):
    """
    Evaluate statistics (see class Stat) from values ​​in M, using weights if passed.
    :param x: is a numpy ndarray.
    :param w: is a numpy ndarray (same dimension of m) with the weights of m values
    :param valid_mask: ndarray of bool, same shapes of x. if passed, indicates which values use to evaluate the results
    :param invalid_value: if valid_mask is not passed, it is evaluated by invalid_value (for ex: x != -1)
    :param _verb: boolean
    :return: stat: a dictionary containing the results
    """
    # todo: usare numpy mask? <-- se il fatto che comprimo le dimensioni romperà le scatole

    if x is not None:
        if w is not None:
            _w = True
            if _verb:
                print(Bcolors.VERB)
                print('* Statistic estimation with weighted average')
        else:
            w = np.ones_like(x)  # all weights are set as 1 like in an arithmetic average
            _w = False
            if _verb:
                print('* Statistic estimation with arithmetic average')

        if _verb:
            print('matrix.shape: ', x.shape)
            if _w:
                print('weights.shape: ', w.shape)

        # define dictionary of the results
        results = dict()

        # if valid_mask is passed (or created by invalid_value), only valid values are extracted
        # else, all the values are selected
        if valid_mask is None:
            if invalid_value is None:
                valid_mask = np.ones_like(x).astype(np.bool)
            else:
                valid_mask = (x != invalid_value)  # for example, -1

        valid_values = x[valid_mask]  # 1-axis vector
        valid_weights = w[valid_mask]  # 1-axis vector

        if _verb:
            print('Number of values selected: {}'.format(valid_values.shape))

        # collect number of values, min and max
        results[Stat.N_VALID_VALUES] = valid_values.shape[0]  # [0] because is a tuple
        results[Stat.MIN] = valid_values.min()
        results[Stat.MAX] = valid_values.max()

        # collect mean, std, SEM, mode and median
        results[Stat.AVG] = np.average(valid_values, axis=0, weights=valid_weights)
        results[Stat.STD] = np.sqrt(
            np.average((valid_values - results[Stat.AVG]) ** 2, axis=0, weights=valid_weights))
        results[Stat.SEM] = scipy_stats.sem(valid_values)
        results[Stat.MEDIAN] = np.median(valid_values)

        # mode return an object with 2 attributes: mode and count, both as n-array, with n the iniput axis
        results[Stat.MODE] = scipy_stats.mode(valid_values).mode[0]

        # save modality of averaging
        results[Stat.MODALITY] = Mode.ARITH if w is None else Mode.WEIGHT

        if _verb:
            print('Modality:, ', results[Stat.MODALITY])
            print(' - avg:, ', results[Stat.AVG])
            print(' - std:, ', results[Stat.STD])
            print(Bcolors.ENDC)
        return results

    else:
        if _verb:
            print('ERROR: ' + statistics_base.__main__ + ' - Matrix passed is None')
        return None


def estimate_local_disarray(R, parameters, ev_index=2, _verb=True, _verb_deep=False):
    # give R and:
    # - according with the space resolution defined in 'parameters'
    # - using only the eigenvectors with index 'ev_index'
    # estimate the local_disarray (disarray in cluster of versors) in the entire R matrix.
    # Local disarray is defined as: 100 * (1 - align),
    # where local align is estimated by the module of the average vector of the local versors.
    # Two different average are used:
    # an arithmetic average (version 0)
    # a weighted average, where every vector is weighted with its Fractional Anisotropy (version 1).
    # Both versions are saved in the structured var 'matrices_of_disarrays', as
    # matrices_of_disarrays['arithmetic'] and matrices_of_disarrays['weighted']

    # extract parameters
    res_xy = parameters['px_size_xy']
    res_z = parameters['px_size_z']
    resolution_factor = res_z / res_xy

    # estimate space resolution of R
    block_side = int(parameters['roi_xy_pix'])
    num_of_slices_P = block_side / resolution_factor
    shape_P = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)

    if _verb:
        print('\n\n*** Estimate_local_disarray()')
        print('R settings:')
        print('> R.shape: ', R.shape)
        print('> R[ev].shape: ', R['ev'].shape)
        print('> Rapporto fra Pixel Size (z / xy) =', resolution_factor)
        print('> Numero di slice selezionate per ogni ROI in R ({} x {}): {}'.format(block_side, block_side, num_of_slices_P))
        print('> Dimension of Parallelepiped in R:', shape_P, 'pixel  =  [{0:2.2f} {0:2.2f} {1:2.2f}] um'.format(
            block_side * res_xy, num_of_slices_P * res_z))

    # extract disarray space resolution from parameters
    Ng_z = parameters['local_disarray_z_side']
    Ng_xy = parameters['local_disarray_xy_side']
    neighbours_lim = parameters['neighbours_lim']

    # check if disarray space resolution on image plane is valid
    if Ng_xy == 0:
        Ng_xy = int(Ng_z * resolution_factor)
    elif Ng_xy < 2:
        Ng_xy = 2

    if _verb:
        print('Disarray settings:')
        print('> Grane side in XY: ', Ng_xy)
        print('> Grane side in Z: ', Ng_z)
        print('> neighbours_lim for disarray: ', neighbours_lim)

    # shape of grane of disarray analysis
    shape_G = (int(Ng_xy), int(Ng_xy), int(Ng_z))

    # iteration long each axis
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    if _verb:
        print('\n\n> Expected iterations on each axis: ', iterations)
        print('\n\n> Expected total iterations       : ', np.prod(iterations))

    # define global matrix that contains local disarrays and local FA
    # - local disarray is the disarray of the selected cluster of orientation versors
    # - local FA is the mean of the FAs of the selected cluster of orientation versors
    matrices_of_disarray = dict()
    matrices_of_disarray[Mode.ARITH] = np.zeros(iterations).astype(np.float32)
    matrices_of_disarray[Mode.WEIGHT] = np.zeros(iterations).astype(np.float32)
    matrix_of_local_fa = np.zeros(iterations).astype(np.float32)

    if _verb: print('\n *** Start elaboration...')
    if _verb_deep: print(Bcolors.VERB)  # open colored session
    _i = 0
    for z in range(iterations[2]):
        print('- iter: {0:3.0f} ; z = {1})'.format(_i, z))
        for r in range(iterations[0]):
            for c in range(iterations[1]):
                if _verb_deep:
                    print(Bcolors.FAIL + ' *** DEBUGGING MODE ACTIVATED ***')
                    print('\n\n\n\n')
                    print(Bcolors.WARNING +
                          'iter: {0:3.0f} - (z, r, c): ({1}, {2} , {3})'.format(_i, z, r, c) +
                          Bcolors.VERB)

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = tuple(create_slice_coordinate(start_coord, shape_G))
                grane = R[slice_coord]  # extract a sub-volume called 'grane' with size (Gy, Gx, Gz)
                if _verb_deep:
                    print(' 0 - grane -> ', end='')
                    print(grane.shape)

                # N = Gy*Gx*Gz = n' of orientation blocks
                # from shape (gx, gy, gz) --> to --> (N,)
                grane_reshaped = np.reshape(grane, np.prod(grane.shape))
                if _verb_deep:
                    print(' 1 - grane_reshaped -> ', end='')
                    print(grane_reshaped.shape)

                # TODO - BUG corretto: qui è ORIENT_INFO non CELL_INFO !
                n_valid_cells = np.count_nonzero(grane_reshaped[Param.ORIENT_INFO])
                if _verb_deep:
                    print(' valid_cells --> ', n_valid_cells)
                    print(' valid rows: -> ', grane_reshaped[Param.ORIENT_INFO])
                    print(' grane_reshaped[\'orient_info\'].shape:', grane_reshaped[Param.ORIENT_INFO].shape)

                if n_valid_cells > parameters['neighbours_lim']:

                    # (N) -> (N x 3) (select eigenvector with index 'ev_index' from all the N cells)
                    coord = grane_reshaped[Param.EV][:, :, ev_index]
                    if _verb_deep:
                        print(' 2 - coord --> ', coord.shape)
                        print(coord)

                    # extract the fractional anisotropy (shape:(N)) [ALL (for the moment)]
                    fa = grane_reshaped[Param.FA]

                    # for print components, lin.norm and FA of every versors (iv = index_of_versor)
                    if _verb_deep:
                        for iv in range(coord.shape[0]):
                            print(iv, ':', coord[iv, :],
                                  ' --> norm:', np.linalg.norm(coord[iv, :]),
                                  ' --> FA: ', fa[iv])

                    # select only versors and FAs from valid cells:
                    valid_coords = coord[grane_reshaped[Param.ORIENT_INFO]]
                    valid_fa = fa[grane_reshaped[Param.ORIENT_INFO]]
                    if _verb_deep:
                        print(' valid coords - ', valid_coords.shape, ' :')
                        print(valid_coords)
                        print(' valid FAs - ', valid_fa.shape, ' :')
                        print(valid_fa)

                    # order the valid versors by their FA (descending)
                    # [use '-fa' for obtain descending order]
                    ord_coords = valid_coords[np.argsort(-valid_fa)]
                    ord_fa = valid_fa[np.argsort(-valid_fa)]

                    # take the first versor (biggest FA) to
                    # move all other versors in the same half-space
                    v1 = ord_coords[0, :]

                    # move all other versors in the same direction
                    # (by checking the positive or negative result of dot product between
                    # the first versor and the others)
                    if _verb_deep: print('Check if versors have congruent direction (same half-space)')
                    for iv in range(1, ord_coords.shape[0]):
                        scalar = np.dot(v1, ord_coords[iv])
                        if scalar < 0:
                            # change the direction of i-th versor
                            if _verb_deep: print(ord_coords[iv], ' --->>', -ord_coords[iv])
                            ord_coords[iv] = -ord_coords[iv]

                    # for print definitive components, lin.norm and FA of every versors
                    if _verb_deep:
                        print('Definitive versors component in the same half-space:')
                        for iv in range(ord_coords.shape[0]):
                            print(iv, ':', ord_coords[iv, :],
                                  ' --> norm:', np.linalg.norm(ord_coords[iv, :]),
                                  ' --> FA: ', ord_fa[iv])

                    if _verb_deep:
                        print('np.average(ord_coords): \n', np.average(ord_coords, axis=0))
                        print('np.average(ord_coords, weight=fa): \n',
                              np.average(ord_coords, axis=0, weights=ord_fa))

                    # alignment degree: module of the average vector.
                    # The averages estimated are both arithmetical and weighted with the FA
                    alignment = dict()
                    local_disarray = dict()

                    # local disarray ARITHMETIC WAY
                    try:
                        alignment[Mode.ARITH]                     = np.linalg.norm(np.average(ord_coords, axis=0))
                        local_disarray[Mode.ARITH]                = 100 * (1 - alignment[Mode.ARITH])
                        matrices_of_disarray[Mode.ARITH][r, c, z] = local_disarray[Mode.ARITH]
                        R[slice_coord][Param.LOCAL_DISARRAY]      = local_disarray[Mode.ARITH]
                    except:
                        alignment[Mode.ARITH]                     = -1
                        local_disarray[Mode.ARITH]                = -1
                        matrices_of_disarray[Mode.ARITH][r, c, z] = -1
                        R[slice_coord][Param.LOCAL_DISARRAY]      = -1

                    # local disarray WEIGHTED with fractional anisotropy
                    try:
                        alignment[Mode.WEIGHT]                     = np.linalg.norm(np.average(ord_coords, axis=0, weights=ord_fa))
                        local_disarray[Mode.WEIGHT]                = 100 * (1 - alignment[Mode.WEIGHT])
                        matrices_of_disarray[Mode.WEIGHT][r, c, z] = local_disarray[Mode.WEIGHT]
                        R[slice_coord][Param.LOCAL_DISARRAY_W]     = local_disarray[Mode.WEIGHT]
                    except:
                        alignment[Mode.WEIGHT]                     = -1
                        local_disarray[Mode.WEIGHT]                = -1
                        matrices_of_disarray[Mode.WEIGHT][r, c, z] = -1
                        R[slice_coord][Param.LOCAL_DISARRAY_W]     = -1

                    # estimate average Fractional Anisotropy and save results into matrix of local disarray
                    try:
                        matrix_of_local_fa[r, c, z] = np.mean(ord_fa)
                    except:
                        matrix_of_local_fa[r, c, z] = -1

                    if _verb_deep:
                        print('alignment[Mode.ARITH] : ', alignment[Mode.ARITH])
                        print('alignment[Mode.WEIGHT]: ', alignment[Mode.WEIGHT])
                        print('local_disarray[Mode.ARITH] : ', local_disarray[Mode.ARITH])
                        print('local_disarray[Mode.WEIGHT]: ', local_disarray[Mode.WEIGHT])
                        print('mean Fractional Anisotropy : ', matrix_of_local_fa[r, c, z])
                        print('saving.. rcz:({},{},{}):'.format(r, c, z))

                else:
                    # Assign invalid value (-1)
                    # (assume that isolated quiver have no disarray)
                    R[slice_coord][Param.LOCAL_DISARRAY]       = -1
                    R[slice_coord][Param.LOCAL_DISARRAY_W]     = -1
                    matrices_of_disarray[Mode.ARITH][r, c, z]  = -1
                    matrices_of_disarray[Mode.WEIGHT][r, c, z] = -1
                    matrix_of_local_fa[r, c, z]                = -1

                # end iteration
                _i += 1

    # close colored session
    print(Bcolors.ENDC)

    return matrices_of_disarray, matrix_of_local_fa, shape_G, R


def save_in_numpy_file(array, R_prefix, shape_G, parameters,
                       base_path, process_folder, data_prefix=''):

    numpy_filename_endname = '{}_G({},{},{})_limNeig{}.npy'.format(
        R_prefix,
        int(shape_G[0]), int(shape_G[1]), int(shape_G[2]),
        int(parameters['neighbours_lim']))

    disarray_numpy_filename = data_prefix + numpy_filename_endname
    np.save(os.path.join(base_path, process_folder, disarray_numpy_filename), array)
    return disarray_numpy_filename


def compile_results_strings(matrix, name, stats, mode='none_passed', ext=''):
    """

    :param matrix: input values
    :param name: name of the parameter of matrix
    :param stats: statistic results on matrix
    :param mode: aritm or weighted
    :param ext: extension of the results (%, um, none..)
    :return: strings: list of strings
    """
    strings = list()

    strings.append('\n\n *** Results of statistical analysis of {} on accepted points. \n'.format(name))
    strings.append('> Matrix of {} with shape: {}'.format(name, matrix.shape))
    strings.append('> Number of valid values: {}'.format(stats[Stat.N_VALID_VALUES]))
    strings.append('> Modality of statistical evaluation: {}\n'.format(mode))
    strings.append('> Statistical Results:')

    # generation of a string for each statistical parameters
    for att in [att for att in vars(Stat) if str(att)[0] is not '_']:
        # print(att)
        # print(getattr(Stat, att))
        # print(stats)
        # print('{0}'.format(getattr(Stat, att)))
        # print('{0:0.2f}'.format(stats[getattr(Stat, att)]))
        # print('{0}'.format(ext))

        # check if the parameter contains a string (ex: 'ARITH' or 'WEIGHT')
        if isinstance(stats[getattr(Stat, att)], str):
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))

        elif att == Stat.N_VALID_VALUES:
            # print integer and without extension
            strings.append(' - {0}: {1}'.format(getattr(Stat, att), stats[getattr(Stat, att)]))
        else:
            strings.append(' - {0}: {1:0.2f}{2}'.format(getattr(Stat, att), stats[getattr(Stat, att)], ext))
    return strings

