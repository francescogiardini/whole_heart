import os
import argparse
import json

from tifffile import imsave

from matplotlib.pyplot import xlabel
from scipy.fft import fft2, ifft2, fftshift
import numpy as np
from skimage.filters import threshold_otsu
from scipy.signal import correlate2d
from scipy.signal import fftconvolve
from scipy.ndimage import shift

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
    # print('Otsu threshold:', threshold)

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

# calculate the area of the binary mask
def area_of_mask(mask):
    '''
    Calculate the area of the binary mask
    :param mask: numpy array of the binary mask
    :return: area of the mask
    '''
    return np.sum(mask)


def calculate_sharpness(image, _plot=False, _verb=False):
    """
    Calculate the sharpness of an image using the gradient of the image.

    :param image: numpy array of the image
    :return: sharpness value
    """
    # Calculate the gradient of the image
    gradient_x = np.gradient(image, axis=1)
    gradient_y = np.gradient(image, axis=0)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Calculate the sharpness as the mean of the gradient
    sharpness = np.mean(gradient)

    if _plot:
        plot_grayscale_image(gradient_x, title='Gradient X', vmin=0, vmax=np.max(gradient_x))
        plot_grayscale_image(gradient_y, title='Gradient Y', vmin=0, vmax=np.max(gradient_y))
        plot_grayscale_image(gradient, title='Gradient', vmin=0, vmax=np.max(gradient))

    return sharpness


def evaluate_snr_of_xy_frame(frame, _segm=True, radius=0.2, _plot=False, _verb=False):
    '''
    Evaluate contrast on a single xy frame of the tomogram
    :param frame: numpy array of the frame (y, x)
    :param _segm: if True, evaluate contrast only on the segmentation of the sample
    :return: contrast value
    '''
    if _plot: plot_grayscale_image(frame, title='Original Frame', vmin=0, vmax=np.max(frame))

    if _segm:
        background_mask = segment_backround_otsu(frame)
        if _plot: plot_grayscale_image(background_mask, title='Background Mask', vmin=0, vmax=1)
        frame[background_mask] = 0
        if _plot: plot_grayscale_image(frame, title='Masked Frame', vmin=0, vmax=np.max(frame))

    # Calculate SNR based on high frequency content
    # snr_image = calculate_snr_high_freq(frame, _plot=_plot, _verb=_verb)
    # snr_image = calculate_sharpness(frame, _plot=_plot, _verb=_verb)
    snr_image = calculate_snr_by_high_low_freq_ratio(frame, radius_ratio=radius, _plot=_plot, _verb=_verb)

    # Normalize on the area of the mask
    if _segm:
        area = area_of_mask(background_mask)
        # print('Area of the mask: ', area)
        snr_sample = snr_image / area if _segm else snr_image
        return snr_image, snr_sample

    return snr_image, snr_image


def calculate_snr_by_high_low_freq_ratio(image, radius_ratio=0.5, _plot=False, _verb=False):
    """
    Calculate the Signal-to-Noise Ratio (SNR) based on high frequency content of an image.

    :param image: numpy array of the image
    :return: SNR value
    """
    # Perform Fourier transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    if _verb:
        print("f_transform_shifted.shape:. ", f_transform_shifted.shape)
        print("max of f_transform_shifted:. ", np.max(f_transform_shifted))

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)  # Use log1p to avoid log(0)
    if _verb: print("max of log_magnitude_spectrum: ", np.max(log_magnitude_spectrum))
    if _plot: plot_grayscale_image(log_magnitude_spectrum, title='Log Magnitude Spectrum', vmin=0,
                                   vmax=np.max(log_magnitude_spectrum))

    # Create a circular mask to exclude low frequencies
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
    radius = radius_ratio * min(center_row, center_col)
    high_freq_mask = distance_from_center > radius
    if _plot: plot_grayscale_image(high_freq_mask, title='High Frequencies Mask', vmin=0, vmax=1)

    # Calculate the energy of high frequencies
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask] ** 2)

    # Calculate the total energy of the image
    total_energy = np.sum(magnitude_spectrum ** 2)

    # Calculate SNR
    snr = high_freq_energy / total_energy
    return snr


def calculate_snr_high_freq(image, _plot=False, _verb=False):
    """
    Calculate the Signal-to-Noise Ratio (SNR) based on high frequency content of an image.

    :param image: numpy array of the image
    :return: SNR value
    """
    # Perform Fourier transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    if _verb:
        print("f_transform_shifted.shape:. ", f_transform_shifted.shape)
        print("max of f_transform_shifted:. ", np.max(f_transform_shifted))

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    if _verb: print("max of magnitude_spectrum:. ", np.max(magnitude_spectrum))
    if _plot: plot_grayscale_image(magnitude_spectrum, title='Magnitude Spectrum', vmin=0, vmax=np.max(magnitude_spectrum))

    # Define a threshold to separate high frequencies
    threshold = np.percentile(magnitude_spectrum, 95)
    if _verb: print("high-freq threshold on magnitude_spectrum:. ", threshold)

    # Calculate the energy of high frequencies
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > threshold] ** 2)
    if _verb: print("high_freq_energy:. ", high_freq_energy)
    if _plot: plot_grayscale_image(magnitude_spectrum > threshold, title='High Frequencies Mask', vmin=0, vmax=1)

    # Calculate the total energy of the image
    total_energy = np.sum(magnitude_spectrum ** 2)
    if _verb: print("total_energy:. ", total_energy)

    # Calculate SNR
    snr = high_freq_energy / total_energy
    return snr


def plot_snrs(xl, yl, xr, yr, xlabel, ylabel, title):

    # check if all the list has the same lentgh
    if not len(xl) == len(yl) == len(xr) == len(yr):
        raise ValueError('All the lists must have the same length')

    # plot (xl, yl) and (xr, yr) with different colors
    plt.figure(figsize=(10, 10))
    plt.plot(xl, yl, 'r', label='Left Cam')
    plt.plot(xr, yr, 'b', label='Right Cam')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    return None


def plot_snr(xl, yl, xlabel, ylabel, title):

    # check if all the list has the same lentgh
    if not len(xl) == len(yl):
        raise ValueError('All the lists must have the same length')

    plt.figure(figsize=(10, 10))
    plt.plot(xl, yl, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return None


def evaluate_snr_along_z(tomogram, _segm=True, fov_portion=(0, 0.8), z_step=10, radius=0.2, _plot=False, _verb=False):
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

    # just for visualization before and after cropping step
    z_half = int(tomogram.shape[0] / 2)
    if _plot: plot_grayscale_image(tomogram[z_half], title='Z={} - Before Cropping Fov'.format(z_half), vmin=0,
                                   vmax=255)

    # if fov_portion is not 0 and 1, evaluate contrast only on the portion of the field of view
    tomogram = crop_fov(tomogram, fov_portion)
    if _plot: plot_grayscale_image(tomogram[z_half], title='After Cropping Fov', vmin=0, vmax=255)

    snr_image_values = {}  # dict z -> contrast value of images
    snr_sample_values = {}  # dict absolute_z -> contrast value of the sample area
    # TODO CHANGE HERE
    z_selected = range(0, tomogram.shape[0], z_step)
    # z_selected = (20, 90)
    print('Selected z values: ', list(z_selected))

    for z in z_selected:
        print(' - z = {}'.format(z))
        # print('Evaluating SNR at z={}...'.format(z))
        snr_image_values[z], snr_sample_values[z] = evaluate_snr_of_xy_frame(tomogram[z], _segm=_segm, radius=radius, _plot=_plot, _verb=_verb)
        # print('  > SNR_image: {}'.format(snr_image_values[z]))
        # print('  > SNR_sample: {}\n'.format(snr_sample_values[z]))

    return snr_image_values, snr_sample_values


def plot_snrs_dual(snr_1, snr_2, _log=False, _segm=False, xlabel='x', ylabel='y',
                   labels=('snr1', 'snr21'), title='title', _save=False, _output_dirpath=None, fname=None):

    # check if all the list has the same lentgh
    if not len(snr_1) == len(snr_2):
        raise ValueError('All the lists must have the same length')

    if _log:
        # convert values to log10
        snr_1 = {k: 10 * np.log10(v) for k, v in snr_1.items()}
        snr_2 = {k: 10 * np.log10(v) for k, v in snr_2.items()}

    # prepare figure and plot data
    plt.figure(figsize=(20, 10))
    plt.plot(snr_1.keys(), snr_1.values(), 'o-', color='b', label=labels[0])
    if _segm:
        plt.plot(snr_2.keys(), [v * 10 ** 6 for v in snr_2.values()], 'o-', color='g',  label=labels[1])
    else:
        plt.plot(snr_2.keys(), snr_2.values(), 'o-', color='g',  label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # increase number of x ticks
    plt.xticks(np.arange(min(snr_1.keys()), max(snr_1.keys())+1, 5))

    if _save:
        plt.savefig(os.path.join(_output_dirpath, fname))

    # show plot
    plt.show()
    return None


def find_crossing_z(left_data, right_data):
    z_values = list(map(int, left_data.keys()))
    left_values = list(left_data.values())
    right_values = list(right_data.values())

    for i in range(10, len(z_values) - 10):
        if left_values[i] > right_values[i]:
            return z_values[i]
    return None


def evaluate_best_z(left_data, right_data):

    # extract some frames from the tomograms with


    return find_crossing_z(left_data, right_data)


def plot_rgb_frames(frame_red, frame_green, title='RGB Plot'):
    """
    Plots two 8-bit frames in an RGB image, with one frame in red and the other in green.

    :param frame_red: numpy array of the first frame (8-bit) to be plotted in red
    :param frame_green: numpy array of the second frame (8-bit) to be plotted in green
    :param title: title of the plot
    """
    if frame_red.shape != frame_green.shape:
        raise ValueError('Both frames must have the same dimensions')

    # Create an RGB image
    rgb_image = np.zeros((frame_red.shape[0], frame_red.shape[1], 3), dtype=np.uint8)
    rgb_image[..., 0] = frame_red  # Red channel
    rgb_image[..., 1] = frame_green  # Green channel

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.legend(['Red Frame', 'Green Frame'], loc='upper right')
    plt.axis('off')  # Hide axes
    plt.show()
    return None

'''
def realign_xy_plane(frame1, frame2):
    
    # Realign the xy plane of frame2 to frame1 using cross-correlation.

    # :param frame1: numpy array of the reference frame (8-bit)
    # :param frame2: numpy array of the frame to be aligned (8-bit)
    # :return: aligned frame2
    
    # Calculate the cross-correlation between the two frames
    cross_corr = correlate2d(frame1, frame2, mode='same')
    # TODO questo è lento, vedi sotto
    # https: // stackoverflow.com / questions / 24768222 / how - to - detect - a - shift - between - images

    # Find the peak in the cross-correlation
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    center = np.array(cross_corr.shape) // 2
    shift_yx = np.array(max_idx) - center

    # Apply the shift to frame2
    aligned_frame2 = shift(frame2, shift=shift_yx)

    return aligned_frame2
'''


def realign_xy_plane(frame1, frame2):
    '''
    Realign the xy plane of frame2 to frame1 using cross-correlation with fftconvolve.

    :param frame1: numpy array of the reference frame (8-bit)
    :param frame2: numpy array of the frame to be aligned (8-bit)
    :return: aligned frame2
    '''
    # Calculate the cross-correlation using fftconvolve
    cross_corr = fftconvolve(frame1, frame2[::-1, ::-1], mode='same')

    # Find the peak in the cross-correlation
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    center = np.array(cross_corr.shape) // 2
    shift_yx = np.array(max_idx) - center

    # Apply the shift to frame2
    aligned_frame2 = shift(frame2, shift=shift_yx)

    return aligned_frame2, shift_yx


def fuse_dual_tomograms(top, bottom, z_switch):
    """
    Fuse two tomograms along the z-axis.

    :param L_ready_zyx: numpy array of the left tomogram (z, y, x)
    :param R_zyx_realigned: numpy array of the right tomogram (z, y, x), realigned on xy plane
    :param z_fusion: integer, the z index where the fusion occurs
    :return: fused tomogram (z, y, x)
    """
    # Use R_zyx_realigned from 0 to z_fusion
    top_part = top[:z_switch + 1, ...]

    # Use L_ready_zyx from z_fusion + 1 to the end
    bottom_part = bottom[z_switch + 1:, ...]

    # Concatenate the two parts along the z-axis
    fused_tomogram = np.concatenate((top_part, bottom_part), axis=0)

    return fused_tomogram


def save_fused_tomogram(fused_tomogram, output_filepath):
    """
    Save the 3D fused tomogram as a TIFF file with correct axis order.

    :param fused_tomogram: numpy array of the fused tomogram (z, y, x)
    :param output_filepath: string, path to save the output TIFF file
    """
    # Ensure the data is in the correct format (z, y, x)
    if len(fused_tomogram.shape) != 3:
        raise ValueError("Fused tomogram must be a 3D numpy array (z, y, x).")

    # Save the 3D array as a TIFF file
    imsave(output_filepath, fused_tomogram, imagej=True)


def main(parser):

    # https://dsp.stackexchange.com/questions/61818/what-are-the-measurable-factors-of-image-sharpness


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
    # output_filepath = os.path.join(output_dirpath, output_filename)
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
        L_ready_zyx, shape_L = load_tiff_stack_zyx(left_cam_path) # (r, c, z) -> (z, y, x)
        print('Loading RIGHT_CAM  tomogram....')
        R_ready_zyx, shape_R = load_tiff_stack_zyx(right_cam_path) # (r, c, z) -> (z, y, x)
        print('Done.')

        # add info to the report
        mess_strings.append(' > Loaded already preproccesed tomograms:')
        mess_strings.append(' > Dimension of LEFT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_L[0], shape_L[1], shape_L[2]))
        mess_strings.append(' > Dimension of RIGHT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_R[0], shape_R[1], shape_R[2]))
    else:
        # TODO PREPROCESSING FUNCTION
        # L_ready_zyx, shape_L = dual_mesospim_preprocessing(leftCAM_path)
        # R_ready_zyx, shape_R = dual_mesospim_preprocessing(rightCAM_path)

        # save preprocessed tomograms
        # save_tiff_stack_zyx(L_ready_zyx, output_dirpath, 'preprocessed_LEFT_CAM')
        # save_tiff_stack_zyx(R_ready_zyx, output_dirpath, 'preprocessed_RIGHT_CAM')
        pass

    # print and add to .txt
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ===================================== BEST Z POSITION FOR FUSION ==========================================
    # TODO: add info to reports txt
    # ===============================================================================================
    _plot             = False
    _save_final_plots = True
    _verb             = False

    _segm  = False
    z_step = 50
    radius = 0.1

    print(Bcolors.OKBLUE + '\n\n*** Evaluating SNR with radius ratio: {} '.format(radius) + Bcolors.ENDC)
    L_snr_image, L_snr_sample = evaluate_snr_along_z(L_ready_zyx, _segm=_segm, fov_portion=(0, 0.8), z_step=z_step, radius=radius, _plot=_plot, _verb=_verb)
    R_snr_image, R_snr_sample = evaluate_snr_along_z(R_ready_zyx, _segm=_segm, fov_portion=(0, 0.8), z_step=z_step, radius=radius, _plot=_plot, _verb=_verb)

    # save results in a file
    snr_values_out_path = os.path.join(output_dirpath, 'snr_values')
    if not os.path.exists(snr_values_out_path):
        os.makedirs(snr_values_out_path)
    with open(os.path.join(snr_values_out_path, 'L_snr_image_radius{}_zstep{}.txt'.format(radius, z_step)), 'w') as file:
        json.dump(L_snr_image, file)
    with open(os.path.join(snr_values_out_path, 'R_snr_image_radius{}_zstep{}.txt'.format(radius, z_step)), 'w') as file:
        json.dump(R_snr_image, file)

    # extract the best z value to fuse the two tomograms
    z_fusion = evaluate_best_z(L_snr_image, R_snr_image)
    print('Best z value to fuse the two tomograms: ', z_fusion)

    # save the best z value in the report
    mess_strings.append(' \n*** Best z value to fuse the two tomograms: {}'.format(z_fusion))


    # ===============================================================================================
    # ===================================== REALIGNMENT ON XY PLANE  ==========================================

    # extract frames from L and R tomogram at best z for furion
    L_switch_frame = L_ready_zyx[z_fusion]
    R_switch_frame = R_ready_zyx[z_fusion]

    # plot the two frames in red and green on the same plot with two different color to check the alignment
    # plot_rgb_frames(L_switch_frame, R_switch_frame, title='Alignment Check - Best Z: {}'.format(z_fusion))

    # perform the realignment on the xy plane on R tomogram
    print("Evaluating the xy translation of Right tomogram respect to left...")
    R_switch_frame_realigned, right_shift_yx = realign_xy_plane(L_switch_frame, R_switch_frame)
    mess_strings.append(' \n*** Shift (y, x) of right cam to left: ({}, {})'.format(right_shift_yx[0], right_shift_yx[1]))

    # check realignment
    # plot_rgb_frames(L_switch_frame, R_switch_frame_realigned, title='Realignment Check - Best Z: {}'.format(z_fusion))

    # translate right tomogram
    print('Realigning the xy plane of the RIGHT tomogram...')
    # accurato anche per shift non interi ma lento
    # R_zyx_realigned = shift(R_ready_zyx, shift=(0, right_shift_yx[0], right_shift_yx[1]))
    # più veloce (solo shift interi)
    R_zyx_realigned = np.roll(R_ready_zyx, shift=(0, right_shift_yx[0], right_shift_yx[1]), axis=(0, 1, 2))

    # fuse the two tomograms
    print('Fusing the two tomograms...')
    plot_rgb_frames(L_ready_zyx[z_fusion], R_zyx_realigned[z_fusion], title='3D Realignment Check - Best Z: {}'.format(z_fusion))
    fused_tomogram = fuse_dual_tomograms(top=R_zyx_realigned, bottom=L_ready_zyx, z_switch=z_fusion)

    print('Saving the fused tomogram...')
    save_fused_tomogram(fused_tomogram, output_filepath)

    print(Bcolors.OKBLUE + '\n\n**************** End Dual Tomogram Fusion ****************' + Bcolors.ENDC)
    return None


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