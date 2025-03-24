import os
import argparse

from scipy.fft import fft2, ifft2, fftshift
import numpy as np
from skimage.filters import threshold_otsu

from custom_tool_kit import search_value_in_txt, write_on_txt, Bcolors, manage_path_argument
from custom_image_base_tool import load_tiff_stack_zyx

import matplotlib.pyplot as plt


def plot_grayscale_image(image, vmin=0, vmax=255, title='Title'):
    """
    Plots a grayscale image using matplotlib's imshow.

    :param image: numpy array of the image to plot
    :param vmin: minimum value for color scaling
    :param vmax: maximum value for color scaling

    Example usage:
    image = np.random.rand(100, 100) * 255  # Example image
    plot_grayscale_image(image, vmin=50, vmax=200)
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    # plt.axis('off')  # Hide axes
    plt.show()
    return None


def segment_backround_otsu(image):
    """
    Segments an image using Otsu's method.

    :param image: numpy array of the image to segment
    :return: segmented image (binary mask) with background as 1 and object as 0
    """
    # Calculate Otsu's threshold
    threshold = threshold_otsu(image)

    # Create a binary mask where pixels above the threshold are set to 0 (foreground)
    # and pixels below the threshold are set to 1 (background)
    background_mask = image <= threshold
    return background_mask


# def fuse_dual_tomograms(tomogram1, tomogram2, output_path):
   # TODO

def prepare_initial_report(left_cam_path, right_cam_path, parameter_filepath, output_folderpath,
                           output_filename, output_filepath, _skip_preprocessing, z_fusion):
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n**************** Start Dual Tomogram Fusion ****************' + Bcolors.ENDC)
    mess_strings.append('Fusion of two DualMesoSPIM tomograms. \n\n Input parameters:')
    mess_strings.append(' > LEFT_CAM source path: {}'.format(left_cam_path))
    mess_strings.append(' > RIGHT_CAM source path: {}'.format(right_cam_path))
    mess_strings.append(' > Parameters filepath: {}'.format(parameter_filepath))
    mess_strings.append(' > Output folder path: {}'.format(output_folderpath))
    mess_strings.append(' > Output filename: {}'.format(output_filename))
    mess_strings.append(' > Output filepath: {}'.format(output_filepath))
    mess_strings.append(' > Skip preprocessing: {}'.format(_skip_preprocessing))
    mess_strings.append(' > Z fusion: {}'.format(z_fusion))
    return mess_strings


def crop_fov(tomogram, fov_portion, axis=1):
    '''
    Crop the field of view of the tomogram
    :param tomogram: numpy array of the tomogram (z, y, x)
    :param fov_portion: (start, end) -> crop tomogram from start(%) to end(%) on the selected axis
    :param axis: axis to crop the field of view
    :return: cropped tomogram

    '''
    # check if fov_portion valuea are between 0 and 1, and start < end
    if not (0 <= fov_portion[0] < fov_portion[1] <= 1):
        raise ValueError('Invalid fov_portion values. fov_portion must be a tuple (start, end) where 0 <= start < end <= 1')

    start = int(tomogram.shape[axis] * fov_portion[0])
    end = int(tomogram.shape[axis] * fov_portion[1])
    if axis == 1:
        return tomogram[:, start:end, :]
    elif axis == 2:
        return tomogram[:, :, start:end]
    else:
        raise ValueError('Invalid axis value. axis must be 1 (y, row) or 2 (x, column)')


def evaluate_snr_of_xy_frame(frame, _segm=True):
    '''
    Evaluate contrast on a single xy frame of the tomogram
    :param frame: numpy array of the frame (y, x)
    :param _segm: if True, evaluate contrast only on the segmentation of the sample
    :return: contrast value
    '''
    if _segm:
        background_mask = segment_backround_otsu(frame)
        frame[background_mask] = 0

    snr = calculate_snr_high_freq(frame)
    snr = (np.max(frame) - np.min(frame)) / (np.max(frame) + np.min(frame))

    return snr


def calculate_snr_high_freq(image):
    """
    Calculate the Signal-to-Noise Ratio (SNR) based on high frequency content of an image.

    :param image: numpy array of the image
    :return: SNR value
    """
    # Perform Fourier transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Define a threshold to separate high frequencies
    threshold = np.percentile(magnitude_spectrum, 95)

    # Calculate the energy of high frequencies
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > threshold] ** 2)

    # Calculate the total energy of the image
    total_energy = np.sum(magnitude_spectrum ** 2)

    # Calculate SNR
    snr = high_freq_energy / total_energy

    return snr


def evaluate_snr_along_z(tomogram, _segm=True, fov_portion=(0, 0.8), z_step=10, _plot=False):
    '''
    Evaluate snr on xy frames (every z_step frames) of the tomogram and return a list of contrast values
    :param tomogram: numpy array of the tomogram (z, y, x)
    :param _segm: if True, evaluate contrast only on the segmentation of the sample
    :param fov_portion: (start, end) -> evaluate contrast only from start(%) to end(%) rows of field of view
        default= from 0 to 80% to discard glue and scattering artifact on the lower part of field of view
    :param _plot: if True, plot the original and cropped tomogram
    :param z_step: step to iterate on z
    :return: list of contrast values
    '''

    # TODO - cambiare per iterare su tutti
    temp_z = int(tomogram.shape[0]/2)

    if _plot: plot_grayscale_image(tomogram[temp_z], title='Before Cropping Fov', vmin=0, vmax=255)

    # if fov_portion is not 0 and 1, evaluate contrast only on the portion of the field of view
    tomogram = crop_fov(tomogram, fov_portion)
    if _plot: plot_grayscale_image(tomogram[temp_z], title='After Cropping Fov', vmin=0, vmax=255)

    snr_values = {}  # dict absolute_z -> contrast value
    # USARE CICLO QUI SOTTO
    # for z in range(0, tomogram.shape[0], z_step):
    #    snr_values[z] = evaluate_snr_of_xy_frame(tomogram[z], _segm=True)

    # TODO - cambiare per iterare su tutti
    for z in range(temp_z, temp_z+1):
        snr_values[z] = evaluate_snr_of_xy_frame(tomogram[z], _segm=True)

    return snr_values




def main(parser):

    # ===============================================================================================
    # ===================================== INITIAL OPERATIONS ======================================
    # ===============================================================================================

    # read parameters
    args = parser.parse_args()
    left_cam_path = manage_path_argument(args.leftCAM_path)
    right_cam_path = manage_path_argument(args.rightCAM_path)
    parameter_filepath = args.parameters_filepath[0]
    if args.output_filepath:
        output_filepath = manage_path_argument(args.output_filepath)
        output_filename = os.path.basename(output_filepath)
        output_dirpath = os.path.dirname(output_filepath)
    else:
        output_dirpath = os.path.dirname(os.path.dirname(left_cam_path))
        output_filename = "fused_tomogram.tif"
        output_filepath = os.path.join(output_dirpath, output_filename)
    z_fusion = args.z_fusion[0] if args.z_fusion else None
    _skip_preprocessing = args.skip_preprocessing

    # check if the output folder exists
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # check if the output file exists
    output_filepath = os.path.join(output_dirpath, output_filename)
    if os.path.exists(output_filepath):
        raise ValueError('Output file already exists. Please remove it before running the script')

    # check if the input files exist
    if not os.path.exists(left_cam_path):
        raise ValueError('Left CAM tomogram not found')
    if not os.path.exists(right_cam_path):
        raise ValueError('Right CAM tomogram not found')

    # prepare initial messages for console and report.txt
    mess_strings = prepare_initial_report(left_cam_path, right_cam_path, parameter_filepath, output_dirpath,
                           output_filename, output_filepath, _skip_preprocessing, z_fusion)

    # create report txt file
    report_filepath = os.path.join(output_dirpath,
                                   '{}_fusion_report.txt'.format(output_filename.split('.')[0]))

    # print to screen, create .txt file and write into .txt file all temporal information
    write_on_txt(mess_strings, report_filepath, _print=True, mode='w')
    # clear list of strings
    mess_strings.clear()

    # extract parameters from parameters.txt file
    parameter_filename = os.path.basename(parameter_filepath)
    param_names = ['px_size_xy', 'px_size_z', 'fwhm_xy', 'fwhm_z']
    param_values = search_value_in_txt(parameter_filepath, param_names)

    # create dictionary of parameters and add info to mess_strings
    parameters = {}
    mess_strings.append(' \n*** Parameters extracted from {}'.format(parameter_filepath))
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        mess_strings.append('> {} - {}'.format(p_name, parameters[p_name]))

    # print to screen, create .txt file and write into .txt file all temporal informations
    write_on_txt(mess_strings, report_filepath, _print=True, mode='w')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ===================================== PREPROCESSING  ==========================================
    # ----------   from original DUAL_mesoSPIM image sequence to 8bit 6um tiffile -------------------
    # ===============================================================================================

    if _skip_preprocessing:
        mess_strings.append(Bcolors.WARNING + '\n\n*** Preprocessing informations:' + Bcolors.ENDC)
        mess_strings.append(' ATTENTION > Skip preprocessing step ')
        print('Loading LEFT_CAM  tomogram....')
        L_ready, shape_L = load_tiff_stack_zyx(left_cam_path) # (r, c, z) -> (z, y, x)
        print('Loading RIGHT_CAM  tomogram....')
        R_ready, shape_R = load_tiff_stack_zyx(right_cam_path) # (r, c, z) -> (z, y, x)
        print('Done.')

        # add info to the report
        mess_strings.append(' > Loaded already preproccesed tomograms:')
        mess_strings.append(' > Dimension of LEFT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_L[0], shape_L[1], shape_L[2]))
        mess_strings.append(' > Dimension of RIGHT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_R[0], shape_R[1], shape_R[2]))
    else:
        # TODO PREPROCESSING FUNCTION
        # L_ready, shape_L = dual_mesospim_preprocessing(leftCAM_path)
        # R_ready, shape_R = dual_mesospim_preprocessing(rightCAM_path)

        # save preprocessed tomograms
        # save_tiff_stack_zyx(L_ready, output_dirpath, 'preprocessed_LEFT_CAM')
        # save_tiff_stack_zyx(R_ready, output_dirpath, 'preprocessed_RIGHT_CAM')
        pass

    # print and add to .txt
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ============================================= FUSION ==========================================
    # ===============================================================================================
    L_contrast = evaluate_snr_along_z(L_ready, _segm=True, fov_portion=(0, 0.8), _plot=True)
    L_contrast = evaluate_snr_along_z(L_ready, _segm=True, fov_portion=(0, 0.8), _plot=True)
    # R_contrast = evaluate_contrast_on_z(R_ready, _segm=True, fov_portion=0.8)

    # z_fusion = evaluate_best_z(L_contrast, R_contrast)

    # fused_tomogram = fuse_dual_tomograms(L_ready, R_ready, z_fusion)

    # save_fused_tomogram(fused_tomogram, output_filepath)






if __name__ == '__main__':
    # define parser
    parser = argparse.ArgumentParser(description='Fuse two DualMesoSPIM tomograms optimizing SNR Ratio')
    parser.add_argument('-l', '--leftCAM_path',
                        type=str, nargs=1, required=True,
                        help='Path to the LEFT Cam1 tomogram')
    parser.add_argument('-r', '--rightCAM_path',
                        type=str, nargs=1, required=True,
                        help='Path to the RIGHT Cam2 tomogram')
    parser.add_argument('-p', '--parameters-filepath',
                        nargs='+', required=True,
                        help='filepath of parameters.txt file')
    parser.add_argument('-o', '--output_filepath',
                        type=str, nargs=1, required=False,
                        help='Output filepath. If not passed, fused_tomogram will be saved in the base directory of the input')
    parser.add_argument('-sp', '--skip_preprocessing',
                        action='store_true',
                        help='Skip preprocessing step. If passed, LEFT and RIGHT input must be path of tiffiles of'
                             '8bit - 6um pixel size preprocessed tomograms')
    parser.add_argument('-z', '--z-fusion',
                        type=int, nargs=1, required=False,
                        help='If passed, fuse tomograms at this z instead of evaluate best z by image quality')

    main(parser)