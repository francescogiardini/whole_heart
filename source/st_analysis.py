''' #####################################################################
# SIMPLE EXAMPLE TO ORDER THE EIGENVECTOR AND USE IT

ew = np.array([10, 1, 100])
ev = np.random.randint(low=-10, high=10, size=9, dtype='l').reshape(3, 3)
ev[:, 0] = ev[:, 0] * 10
ev[:, 1] = ev[:, 1] * 1
ev[:, 2] = ev[:, 2] * 100
print(ew)
print(ev)

# ordino autovalori e autovettori t.c. w0 > w1 > w2
order = np.argsort(ew)[::-1];  # decrescent order
ew = np.copy(ew[order]);
ev = np.copy(ev[:, order]);
print()
print(ew)
print(ev)

print()
print(ew[0], ' --> ', ev[:, 0])
print(ew[1], ' --> ', ev[:, 1])
print(ew[2], ' --> ', ev[:, 2])

print('... rotation?')
ev_rotated = np.zeros_like(ev)
for axis in range(ev.shape[1]):
    ev_rotated[:, axis] = check_in_upper_semisphere(ev[:, axis])

print(ew[0], ' --> ', ev_rotated[:, 0])
print(ew[1], ' --> ', ev_rotated[:, 1])
print(ew[2], ' --> ', ev_rotated[:, 2])
##################################################################### '''

# system
import os
import time
import argparse

# general
import numpy as np

from scipy import stats as scipy_stats
# from zetastitcher import InputFile
from tifffile import imread as imread

# custom codes
from custom_tool_kit import manage_path_argument, create_coord_by_iter, create_slice_coordinate, \
    search_value_in_txt, pad_dimension, write_on_txt, Bcolors
from custom_image_base_tool import normalize, print_info, plot_histogram, plot_map_and_save
from disarray_tools import estimate_local_disarray, save_in_numpy_file, compile_results_strings, \
    Param, Mode, Cell_Ratio_mode, statistics_base, create_R, structure_tensor_analysis_3d, \
    sigma_for_uniform_resolution, downsample_2_zeta_resolution, CONST


def block_analysis(parall, shape_P, parameters, sigma, _verbose, fa_threshold, z_comp_threshold, z_boost=1.0):
    # parall : np.uint8

    # initialize empty dictionary and
    results = {}
    there_is_cell = False
    there_is_info = False

    # average intensity of the block
    mean_intensity = np.mean(parall)
    results['mean_intensity'] = mean_intensity
    if _verbose:
        print('   mean_intensity :   ', mean_intensity)

    # check if this block contains cell with the selected methods:
    if parameters['mode_ratio'] == Cell_Ratio_mode.MEAN:
        cell_ratio = np.copy(mean_intensity)
    elif parameters['mode_ratio'] == Cell_Ratio_mode.NON_ZERO_RATIO:
        cell_ratio = np.count_nonzero(parall) / np.prod(shape_P)
    else:
        print(Bcolors.WARNING +
              '** WARNING: parameters[\'mode_ratio\'] is not recognized: all blacks are not analyzed' +
              Bcolors.ENDC)
        cell_ratio = 0

    if cell_ratio >= parameters['threshold_on_cell_ratio']:
        # Orientation Analysis in this data block
        there_is_cell = True

        # save in R
        results['cell_ratio'] = cell_ratio
        if _verbose:
            print('   cell_ratio :   ', cell_ratio)

        # blurring (for isotropic FWHM) and downsampling (for isotropic pixel size)
        # parall is int_8, parall_down is float_32
        parall_down = downsample_2_zeta_resolution(parall,
                                                   parameters['px_size_xy'],
                                                   parameters['px_size_z'],
                                                   sigma=sigma)
        # print('parall_down mean: {}'.format(np.mean(parall_down)))

        # 3D Structure Tensor Analysis - Gradient based
        # - w : descendent ordered eigenvalues
        # - v : ordered eigenvectors
        #       the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        # - shape_parameters : dictionary of shape parameters
        w, v, shape_parameters = structure_tensor_analysis_3d(parall_down, _rotation=False, z_boost=z_boost)

        # save ordered eigenvectors
        results['ev'] = v
        results['ew'] = w

        # save shape parameters
        for key in shape_parameters.keys():
            results[key] = shape_parameters[key]

        # check if there is the orientation information in the block
        ev2z = v[2, 2]  # comp. Z del 3th autovettore (comp Z autovett orientaz)
        # se np.abs(ev2z) > 0.975 vuol dire che vettore perfettamente parallelo asse z (img troppo sfuocata sul pinao xy causa imaging)
        if shape_parameters['fa'] >= fa_threshold and np.abs(ev2z) < z_comp_threshold:
            there_is_info = True
        else:
            if _verbose:
                print('Block rejected ( no info in freq )')
    else:
        if _verbose:
            print('Block rejected ( no cell )')

    return there_is_cell, there_is_info, results


def iterate_orientation_analysis(volume, R, parameters, shape_R, shape_P, _verbose=False, fa_threshold=0.30, z_comp_threshold=0.975, z_boost=1.0):
    # virtually dissect 'volume', perform on each block the analysis implemented in 'block_analysis',
    # and save the results inside R

    # estimate sigma of blurring for isotropic resolution
    sigma_blur = sigma_for_uniform_resolution(FWHM_xy=parameters['fwhm_xy'],
                                              FWHM_z=parameters['fwhm_z'],
                                              px_size_xy=parameters['px_size_xy'])

    perc = 0
    count_iterations = 0  # count iteration
    expected_iterations = np.prod(shape_R)
    print(' > Expected iterations : ', expected_iterations)

    for z in range(shape_R[2]):
        if _verbose:
            print('\n\n')
        print('{0:0.1f} % - z: {1:3}'.format(perc, z))
        for r in range(shape_R[0]):
            for c in range(shape_R[1]):

                start_coord = create_coord_by_iter(r, c, z, shape_P)
                slice_coord = create_slice_coordinate(start_coord, shape_P)

                perc = 100 * (count_iterations / expected_iterations)
                if _verbose:
                    print('\n')

                # save init info in R
                R[r, c, z]['id_block'] = count_iterations
                R[r, c, z][Param.INIT_COORD] = start_coord

                # extract parallelepiped
                parall = volume[tuple(slice_coord)]

                # check dimension (if iteration is on border of volume, add zero_pad)
                parall = pad_dimension(parall, shape_P)

                # If it's not all black...
                if np.max(parall) != 0:

                    # analysis of parallelepiped extracted
                    there_is_cell, there_is_info, results = block_analysis(
                        parall,
                        shape_P,
                        parameters,
                        sigma_blur,
                        _verbose,
                        fa_threshold,
                        z_comp_threshold,
                        z_boost)

                    # save info in R[r, c, z]
                    if there_is_cell: R[r, c, z]['cell_info'] = True
                    if there_is_info: R[r, c, z]['orient_info'] = True

                    # save results in R
                    if _verbose:
                        print(' saved in R:  ')
                    for key in results.keys():
                        R[r, c, z][key] = results[key]
                        if _verbose:
                            print(' > {} : {}'.format(key, R[r, c, z][key]))

                else:
                    if _verbose:
                        print('   block rejected   ')

                count_iterations += 1
    return R, count_iterations


# =================================================== MAIN () ================================================
def main(parser):

    # INPUT HARDCODED FOR DEBUG ==================================================================================
    # source_path = '/home/francesco/LENS/ST_analysis_tests/test_vasi/200116_test_vasi_FA_3.1/stack.tif'
    # parameter_filename = 'parameters_vessels.txt'
    # _verbose = False
    # _deep_verbose = False
    # _save_csv = True
    # _save_hist = True
    # _save_maps = True
    # if _verbose:
    #     print(Bcolors.FAIL + ' *** DEBUGGING MODE *** ' + Bcolors.ENDC)
    # ============================================================================================================


    ## Extract input information FROM TERMINAL =========
    args               = parser.parse_args()
    source_path        = manage_path_argument(args.source_path)
    parameter_filename = args.parameters_filename[0]
    _verbose           = args.verbose
    _deep_verbose      = args.deep_verbose
    _plot_quiver       = args.plot_quiver
    fa_threshold       = args.fa_threshold
    z_comp_threshold   = args.z_comp_threshold
    z_boost            = args.z_boost

    if _verbose:
        print(Bcolors.FAIL + ' *** VERBOSE MODE *** ' + Bcolors.ENDC)
    if _deep_verbose:
        print(Bcolors.FAIL + ' *** DEBUGGING MODE *** ' + Bcolors.ENDC)
    ## =================================================

    # extract filenames and folders
    stack_name = os.path.basename(source_path)
    process_folder = os.path.basename(os.path.dirname(source_path))
    base_path = os.path.dirname(os.path.dirname(source_path))
    parameter_filepath = os.path.join(base_path, process_folder, parameter_filename)
    stack_prefix = stack_name.split('.')[0]

    # create introductiveme information
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n*** ST orientation Analysis ***\n' + Bcolors.ENDC)
    mess_strings.append(' > source path: {}'.format(source_path))
    mess_strings.append(' > stack name: {}'.format(stack_name))
    mess_strings.append(' > process folder: {}'.format(process_folder))
    mess_strings.append(' > base path: {}'.format(base_path))
    mess_strings.append(' > Parameter filename: {}'.format(parameter_filename))
    mess_strings.append(' > Parameter filepath: {}'.format(parameter_filepath))
    mess_strings.append('')
    mess_strings.append(' > PREFERENCES:')
    mess_strings.append('  - _verbose {}'.format(_verbose))
    mess_strings.append('  - _deep_verbose {}'.format(_deep_verbose))
    mess_strings.append('  - _plot_quiver {}'.format(_plot_quiver))
    mess_strings.append('  - fa_threshold {}'.format(fa_threshold))
    mess_strings.append('  - z_boost {}'.format(z_boost))

    # extract parameters
    param_names = ['roi_xy_pix',
                   'px_size_xy', 'px_size_z',
                   'mode_ratio', 'threshold_on_cell_ratio',
                   'fwhm_xy','fwhm_z']

    param_values = search_value_in_txt(parameter_filepath, param_names)

    # create dictionary of parameters
    parameters = {}
    mess_strings.append('\n\n*** Parameters used:')
    mess_strings.append(' > Parameters extracted from {}\n'.format(parameter_filename))
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        mess_strings.append('> {} - {}'.format(p_name, parameters[p_name]))

    # Parameters of Acquisition System:
    # ratio between pixel size in z and xy
    ps_ratio = parameters['px_size_z'] / parameters['px_size_xy']

    # analysis block dimension in z-axis
    shape_P = np.array((int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix']),
                        int(parameters['roi_xy_pix'] / ps_ratio))).astype(np.int32)

    mess_strings.append('\n *** Analysis configuration')
    mess_strings.append(' > Rapporto fra Pixel Size (z / xy) = {0:0.2f}'.format(ps_ratio))
    mess_strings.append(' > Numero di slice selezionate per ogni ROI ({} x {}): {}'.format(
        shape_P[0], shape_P[1], shape_P[2]))
    mess_strings.append(' > Dimension of Parallelepiped: ({0},{1},{2}) pixel  ='
                        '  [{3:2.2f} {4:2.2f} {5:2.2f}] um'.format(
        shape_P[0], shape_P[1], shape_P[2],
        shape_P[0] * parameters['px_size_xy'],
        shape_P[1] * parameters['px_size_xy'],
        shape_P[2] * parameters['px_size_z']))

    # check modality of check of cell content
    mess_strings.append('\n *** Check modality of signal in blocks')
    if parameters['mode_ratio'] == Cell_Ratio_mode.MEAN:
        mess_strings.append(' > Selected mode_ratio = MEAN > {}'.format(parameters['threshold_on_cell_ratio']))
    elif parameters['mode_ratio'] == Cell_Ratio_mode.NON_ZERO_RATIO:
        mess_strings.append(' > Selected mode_ratio = NON_ZERO_RATIO > {}'.format(parameters['threshold_on_cell_ratio']))
    else:
        mess_strings.append(' > ** WARNING: parameters[\'mode_ratio\'] is not recognized: all blacks are not analyzed')

    # create result.txt filename:
    # threshold_on_cell_ratio e z_comp_threshold incise nel nome per poter rilanciare
    # l'analisi piu' volte cambiando solo quelle soglie senza sovrascrivere i risultati precedenti
    txt_info_filename = 'Orientations_INFO_' + stack_prefix + '_thr{0:0.2f}'.format(parameters['threshold_on_cell_ratio']) \
                   + '_z{0:0.2f}'.format(z_comp_threshold) + '_' \
                   + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.txt'
    txt_info_path = os.path.join(os.path.dirname(source_path), txt_info_filename)

    # print to screen, create .txt file and write into .txt file all introductive informations
    write_on_txt(mess_strings, txt_info_path, _print=True, mode='w')
    # clear list of strings
    mess_strings.clear()

    # 1 ----------------------------------------------------------------------------------------------------
    # OPEN STACK

    # extract data - entire Volume: 'V'
    # volume = InputFile(source_path).whole()
    volume = imread(source_path)
    # NB - in futuro va cambiata gestion assi
    volume = np.moveaxis(volume, 0, -1)  # (r, c, z) -> (z, y, x)

    # calculate dimension
    shape_V = np.array(volume.shape)
    pixel_for_slice = shape_V[0] * shape_V[1]
    total_voxel_V = pixel_for_slice * shape_V[2]

    mess_strings.append('\n\n*** Entire loaded Volume dimension:')
    mess_strings.append(' > Dimension if entire Volume : ({}, {}, {})'.format(shape_V[0], shape_V[1], shape_V[2]))
    mess_strings.append(' > Pixel for slice            : {}'.format(pixel_for_slice))
    mess_strings.append(' > Total voxel in Volume      : {}'.format(total_voxel_V))

    # extract list of math informations (as strings) about volume.npy variable
    info = print_info(volume, text='\nVolume informations:', _std=False, _return=True)
    mess_strings = mess_strings + info

    # print and add to .txt
    write_on_txt(mess_strings, txt_info_path, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # 2 ----------------------------------------------------------------------------------------------------
    # CYCLE FOR BLOCKS EXTRACTION and ANALYSIS
    print('\n\n')
    print(Bcolors.OKBLUE + '*** Start Structure Tensor analysis... ' + Bcolors.ENDC)

    t_start = time.time()

    # create empty Result matrix
    R, shape_R = create_R(shape_V, shape_P)

    # real analysis on R
    R, count = iterate_orientation_analysis(volume, R, parameters, shape_R, shape_P, _verbose, fa_threshold, z_comp_threshold, z_boost)
    mess_strings.append('\n > Orientation analysis completed.')

    # extract informations about the data analyzed
    block_with_cell = np.count_nonzero(R[Param.CELL_INFO])
    block_with_info = np.count_nonzero(R[Param.ORIENT_INFO])
    p_rejec_cell = 100 * (1 - (block_with_cell / count))
    p_rejec_info_tot = 100 * (1 - (block_with_info / count))
    p_rejec_info = 100 * (1 - (block_with_info / block_with_cell))

    # end analysis
    t_process = time.time() - t_start

    # create results strings
    mess_strings.append('\n\n*** Results of Orientation analysis:')
    mess_strings.append(' > Expected iterations : {}'.format(np.prod(shape_R)))
    mess_strings.append(' > total_ iteration : {}'.format(count))
    mess_strings.append(' > Time elapsed: {0:.3f} s'.format(t_process))
    mess_strings.append('\n > Total blocks: {}'.format(count))
    mess_strings.append(' > block with cell : {0}, rejected from total: {1} ({2:0.1f}%)'.format(
        block_with_cell,
        count - block_with_cell,
        p_rejec_cell))
    mess_strings.append(' > block with gradient information : {}'.format(block_with_info))
    mess_strings.append(' > rejected from total: {0} ({1:0.1f}%)'.format(count - block_with_info, p_rejec_info_tot))
    mess_strings.append(' > rejected from block with cell: {0} ({1:0.1f}%)'.format(
        block_with_cell - block_with_info, p_rejec_info))

    mess_strings.append('\n > R matrix created with shape: ({}, {}, {}) cells (rcz).'.format(
        R.shape[0], R.shape[1], R.shape[2]))

    # print and write into .txt
    write_on_txt(mess_strings, txt_info_path, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # 3 ----------------------------------------------------------------------------------------------------
    # SAVE R in a NUMPY FILES, write info in the 'Orientation_info.txt'

    # create result matrix (R) filename:
    prefix = 'R_' + stack_prefix + '_fa{0:0.2f}'.format(fa_threshold) + '_zb{0:0.2f}'.format(z_boost) \
             + '_thr{0:0.2f}'.format(parameters['threshold_on_cell_ratio']) + '_z{0:0.2f}'.format(z_comp_threshold)
    R_filename = prefix + '_' + str(int(parameters['roi_xy_pix'] * parameters['px_size_xy'])) + 'um.npy'
    R_filepath = os.path.join(base_path, process_folder, R_filename)

    # Save Results in R.npy
    np.save(R_filepath, R)
    mess_strings.append('\n> R matrix saved in: {}'.format(os.path.dirname(source_path)))
    mess_strings.append('> with name: {}'.format(R_filename))
    mess_strings.append('\n> Informations .txt file saved in: {}'.format(os.path.dirname(txt_info_path)))
    mess_strings.append('> with name: {}'.format(txt_info_filename))
    mess_strings.append('\n')

    # print and write into .txt
    write_on_txt(mess_strings, txt_info_path, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # 4 ----------------------------------------------------------------------------------------------------
    # Call the plot_quiver_on_mosaic_frame script
    if _plot_quiver:
        print(Bcolors.OKBLUE + '*** Run plot_quiver_on_frames... ' + Bcolors.ENDC)
        # build the command to run the script
        plot_vectors_script_path = os.path.join(os.path.dirname(__file__), 'plot_vectors_on_frames.py')
        os.system(f'python3 {plot_vectors_script_path} -s {source_path} -p {parameter_filename}')
# =============================================== END MAIN () ================================================


if __name__ == '__main__':

    # ============================================== START  BY TERMINAL ======================================
    my_parser = argparse.ArgumentParser(description='Orientation analysis - 3D Structure Tensor based')
    my_parser.add_argument('-s', '--source-path', nargs='+',
                           help='absolut path of sample to analyze (3d tiff file or folder of tiff files) ',
                           required=True)
    my_parser.add_argument('-p', '--parameters-filename', nargs='+',
                           help='filename of parameters.txt file (in the same folder of stack)', required=True)
    my_parser.add_argument('-v', action='store_true', default=False, dest='verbose',
                           help='print additional informations')
    my_parser.add_argument('-d', action='store_true', default=False, dest='deep_verbose',
                           help='print a lot of informations - DEBUG MODE')
    my_parser.add_argument('-q', '--plot-quiver', action='store_true', default=False, dest='plot_quiver',
                           help='run "plot_quiver_on_mosaic_frame" at the end of the script')
    # add fractional anisotropy threshold
    my_parser.add_argument('-f', '--fa-threshold', type=float, default=0.30, dest='fa_threshold',
                           help='fractional anisotropy threshold for valid direction (default: 0.30)')
    # add threshold on the z component of vectors
    my_parser.add_argument('-z', '--z-comp-threshold', type=float, default=0.975, dest='z_comp_threshold',
                           help='threshold on the z component of vectors (default: 0.975)')
    # add z_boost factor to compensate a weak z gradient
    my_parser.add_argument('-zb', '--z-boost', type=float, default=1.0, dest='z_boost',
                           help='fattore moltiplicativo applicato al gradiente lungo z prima del calcolo del structure tensor, per compensare un gradiente z debole (default: 1.0, nessun effetto)')

    main(my_parser)
    # ============================================== START  BY TERMINAL ======================================

    # ========= START FOR DEBUG =========
    # my_parser = argparse.ArgumentParser()
    # main(my_parser)  # empty
    # ========= START FOR DEBUG =========
