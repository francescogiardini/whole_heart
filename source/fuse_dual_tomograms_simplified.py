import os
import argparse
import json

import cv2
from numpy.ma.core import outer
from scipy.fft import fft2, fftshift
from scipy.signal import fftconvolve
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import shift
from skimage.filters.tests.test_median import image

from custom_tool_kit import search_value_in_txt, write_on_txt, Bcolors, manage_path_argument
from custom_image_base_tool import load_tiff_stack_zyx

import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from scipy.optimize import minimize

from tifffile import imread, imsave

from source.custom_tool_kit import create_fldr


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


def prepare_initial_report(args, output_folderpath, report_filepath):
    mess_strings = list()
    mess_strings.append(Bcolors.OKBLUE + '\n\n**************** Start Dual Tomogram Fusion ****************' + Bcolors.ENDC)
    mess_strings.append('Fusion of two DualMesoSPIM tomograms. \n Input parameters:')

    for param, value in vars(args).items():
        # if value is a list, convert it to a string
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        # add parameter and value to the message strings
        mess_strings.append(' > {}: {}'.format(param, value))
     # add output folder path and report filepath to the message strings
    mess_strings.append(' > Output folder path: {}'.format(output_folderpath))
    mess_strings.append(' > Report filepath: {}'.format(report_filepath))
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


def evaluate_snr_along_z(tomogram, _segm=True, fov_portion=(0, 0.8), z_slicing=10, radius=0.2, _plot=False, _verb=False):
    '''
    Evaluate snr on xy frames (every z_step frames) of the tomogram and return a list of contrast values
    :param tomogram: numpy array of the tomogram (z, y, x)
    :param _segm: if True, evaluate contrast only on the segmentation of the sample
    :param fov_portion: (start, end) -> evaluate contrast only from start(%) to end(%) rows of field of view
        default= from 0 to 80% to discard glue and scattering artifact on the lower part of field of view
    :param _plot: if True, plot the original and cropped tomogram
    :param z_slicing: step to iterate on z
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
    z_selected = range(0, tomogram.shape[0], z_slicing)
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
    return 1


def evaluate_best_z(left_data, right_data):

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


def estimate_isotropic_scaling(img1, img2):

    # Rileva feature con ORB o SIFT
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Matcher (puoi usare anche FLANN per SIFT)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ottieni i punti corrispondenti
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Trova la trasformazione di similitudine (rotazione, scala, traslazione)
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1)

    # Il fattore di scala è la norma delle prime due colonne (ignora rotazione)
    scale_x = np.linalg.norm(M[0, :2])
    scale_y = np.linalg.norm(M[1, :2])
    scale = (scale_x + scale_y) / 2

    return scale


def scale_image_for_best_fit(image1, image2):
    """
    Estimate the isotropic scaling factor to best fit image2 to image1.

    :param image1: Reference image (numpy array).
    :param image2: Image to be scaled (numpy array).
    :return: Optimal scaling factor and the scaled image2.
    """
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")

    optimal_xy_scale = estimate_isotropic_scaling(image1, image2)

    # Apply the optimal scaling to image2
    scaled_image2 = zoom(image2, (optimal_xy_scale, optimal_xy_scale), order=1)
    scaled_image2 = match_image_size(scaled_image2, image1.shape)

    return scaled_image2, optimal_xy_scale

def match_image_size(image, target_shape):
    """
    Crop or pad an image to match the target shape.

    :param image: Input image (numpy array).
    :param target_shape: Desired shape (tuple).
    :return: Image cropped or padded to the target shape.
    """
    output = np.zeros(target_shape, dtype=image.dtype)
    min_shape = np.minimum(image.shape, target_shape)
    slices_input = tuple(slice(0, s) for s in min_shape)
    slices_output = tuple(slice(0, s) for s in min_shape)
    output[slices_output] = image[slices_input]
    return output


def estimate_isotropic_scaling_and_translation(img1, img2):
    # Rileva feature con ORB
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ottieni punti corrispondenti
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Trova la trasformazione di similitudine (include traslazione e scala isotropica + rotazione)
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.LMEDS)

    if M is None:
        raise ValueError("Non è stato possibile stimare la trasformazione.")

    # Calcola la scala isotropica (media della norma delle due righe)
    scale_x = np.linalg.norm(M[0, :2])
    scale_y = np.linalg.norm(M[1, :2])
    scale = (scale_x + scale_y) / 2

    # Traslazione
    tx = M[0, 2]
    ty = M[1, 2]

    return scale, tx, ty


def apply_scale_and_translation(img, scale, tx, ty):
    """
    Applica una trasformazione di similitudine (scala isotropica + traslazione) all'immagine.

    Parameters:
        img   : immagine di input (numpy array)
        scale : fattore di scala isotropico
        tx    : traslazione lungo x
        ty    : traslazione lungo y

    Returns:
        Immagine trasformata
    """
    # Costruisci matrice affine: solo scala e traslazione, niente rotazione
    M = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)

    # Applica la trasformazione affine
    h, w = img.shape[:2]
    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return transformed


def single_cam_preprocessing(imgseq_path, parameters, output_voxel_size=None, max_intensity=None,
                             _save_preprocessed_tomograms=False, outfolderpath=None, cam_name=None, _eight_bit=False):
    """
    Preprocess a single camera tomogram.

    :param imgseq_path: Path to the folder containing the TIFF sequence.
    :param parameters: Dictionary containing 'px_size_xy' and 'px_size_z'.
    :param output_voxel_size: Desired voxel size for scaling.
    :param max_intensity: Maximum intensity value for equalization.
    :param _save_preprocessed_tomograms: If True, saves the preprocessed tomogram.
    :param outpath: Path to save the preprocessed tomogram if saving is enabled.
    :return: Preprocessed tomogram and its shape.
    """
    # Load TIFF sequence into a 3D array
    print('Loading TIFF sequence from {}...'.format(imgseq_path))
    tiff_files = sorted([os.path.join(imgseq_path, f) for f in os.listdir(imgseq_path) if f.endswith('.tif')])
    tomogram = np.stack([imread(f) for f in tiff_files], axis=0)  # (z, y, x)

    # ===============================
    # SCALING
    # ===============================

    # Calculate scaling factors
    px_size_xy = parameters['px_size_xy']
    px_size_z = parameters['px_size_z']

    if output_voxel_size:
        scale_xy = px_size_xy / output_voxel_size
        scale_z = px_size_z / output_voxel_size
        # Apply scaling
        print('Scaling tomogram to voxel size {}...'.format(output_voxel_size))
        tomogram_scaled = zoom(tomogram, (scale_z, scale_xy, scale_xy), order=1)  # Linear interpolation
    else:
        # Not Apply scaling
        print('No scaling applied. Using original voxel size.')
        tomogram_scaled = tomogram

    # ===============================
    # EQUALIZATION
    # ===============================
    if max_intensity:
        # Rescale the histogram to the specified maximum intensity
        print('Rescaling histogram to maximum intensity {} '.format(max_intensity))
        tomogram_scaled = equalize_max_with_clip(tomogram_scaled, max_intensity, dtype=np.uint16)
    else:
        # Not Apply equalization
        print('No equalization applied. Using original intensity range')

    if _eight_bit:
        print('Rescaling histogram to maximum intensity 255 and convert to 8bit ')
        tomogram_scaled = convert_16bit_to_8bit(img16=tomogram_scaled)

    # Save the preprocessed tomogram if required
    if _save_preprocessed_tomograms and outfolderpath and cam_name:
        outfname = cam_name + '_ds{}um_eq{}.tif'.format(int(output_voxel_size), int(max_intensity))
        outfolderpath = os.path.join(outfolderpath, outfname)
        print('Saving preprocessed tomogram to {}...'.format(outfolderpath))
        imsave(outfolderpath, tomogram_scaled, imagej=True)

    return tomogram_scaled, tomogram_scaled.shape


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


def save_fused_tomogram(fused_tomogram, output_folderpath):
    """
    Save the 3D fused tomogram as a TIFF file with correct axis order.

    :param fused_tomogram: numpy array of the fused tomogram (z, y, x)
    :param output_filepath: string, path to save the output TIFF file
    """
    # Ensure the data is in the correct format (z, y, x)
    if len(fused_tomogram.shape) != 3:
        raise ValueError("Fused tomogram must be a 3D numpy array (z, y, x).")

    # Save the 3D array as a TIFF file
    output_filepath = os.path.join(output_folderpath, 'fused_tomogram.tif')
    imsave(output_filepath, fused_tomogram, imagej=True)


def zoom_and_center_crop(input_array_zyx, scale_yx):
    """
    Apply zoom in the xy plane to a 3D array and return an output with the same shape as the input.

    :param input_array_zyx: 3D numpy array (z, y, x)
    :param scale_yx: Scaling factor for the y and x axes
    :return: 3D numpy array with the same shape as the input
    """
    # Apply zoom
    zoomed_array = zoom(input_array_zyx, (1, scale_yx, scale_yx), order=1)

    # Calculate cropping or padding to match the original shape
    original_shape = input_array_zyx.shape
    zoomed_shape = zoomed_array.shape

    output_array = np.zeros(original_shape, dtype=input_array_zyx.dtype)

    for axis in (1, 2):  # Process y and x axes
        start = (zoomed_shape[axis] - original_shape[axis]) // 2
        end = start + original_shape[axis]

        if start < 0:  # Padding needed
            pad_start = -start
            pad_end = original_shape[axis] - (end - zoomed_shape[axis])
            output_array[:, :, :] = np.pad(zoomed_array, ((0, 0), (pad_start, pad_end), (pad_start, pad_end)), mode='constant')
        else:  # Cropping needed
            output_array = zoomed_array[:, start:end, start:end]

    return output_array


def transform_volume_xy(volume, scale, tx, ty, interpolation=cv2.INTER_LINEAR):
    """
    Applica una trasformazione affine (scala + traslazione) lungo XY a un volume 3D ZYX.

    Parameters:
        volume        : array numpy 3D con shape (Z, Y, X)
        scale         : fattore di scala isotropico da applicare in XY
        tx, ty        : traslazione lungo gli assi X e Y
        interpolation : tipo di interpolazione (default: bilineare)

    Returns:
        volume trasformato con stessa shape
    """
    z_dim, y_dim, x_dim = volume.shape
    transformed_volume = np.zeros_like(volume)

    # Matrice di trasformazione affine per XY
    M = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)

    for z in range(z_dim):
        slice_xy = volume[z]
        transformed = cv2.warpAffine(slice_xy, M, (x_dim, y_dim), flags=interpolation)
        transformed_volume[z] = transformed

    return transformed_volume


def equalize_max_with_clip(image, max_intensity, dtype=np.uint16):
    max_val = np.iinfo(dtype).max  # es. 65535 for uint16
    scale_factor = max_val / max_intensity

    # linear transformation
    rescaled = np.clip(image, 0, max_intensity) * scale_factor
    return rescaled.astype(dtype)


def convert_16bit_to_8bit(img16):
    return (img16/256).astype('uint8')


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
    base_dirpath = os.path.dirname(left_cam_path)
    output_folderpath = args.output_folderpath[0] if args.output_folderpath else os.path.join(base_dirpath, "preprocessing")
    z_fusion = args.z_fusion[0] if args.z_fusion else None
    _skip_preprocessing = args.skip_preprocessing
    output_voxel_size = args.voxel_size[0] if args.voxel_size else None
    max_intensity = args.max_intensity[0] if args.max_intensity else None
    _save_preprocessed_tomograms = args.save_preprocessed_tomograms
    z_slicing = args.z_slicing[0] if args.z_slicing else None
    _eight_bit = args.eight_bit

    # check if the output folder exists
    create_fldr(output_folderpath)

    # check if the input files exist
    if not os.path.exists(left_cam_path):
        raise ValueError('Left CAM tomogram not found')
    if not os.path.exists(right_cam_path):
        raise ValueError('Right CAM tomogram not found')

    # create report txt file
    report_filepath = os.path.join(output_folderpath, 'fusion_report.txt')

    # prepare initial messages for console and report.txt
    mess_strings = prepare_initial_report(args, output_folderpath, report_filepath)

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
    mess_strings.append(' \n*** Parameters extracted from {}'.format(parameter_filename))
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
        mess_strings.append(Bcolors.OKBLUE + '\n*** Preprocessing informations:' + Bcolors.ENDC)
        mess_strings.append(' > Preprocessing steps:')
        mess_strings.append(' > Scaling to voxel size {} um'.format(output_voxel_size))
        mess_strings.append(' > Rescaling histogram to maximum intensity {}'.format(max_intensity))

        print('Preprocessing is running....')
        L_ready_zyx, shape_L = single_cam_preprocessing(left_cam_path, parameters,
                                                        output_voxel_size=output_voxel_size,
                                                        max_intensity=max_intensity,
                                                        _save_preprocessed_tomograms=_save_preprocessed_tomograms,
                                                        outfolderpath=output_folderpath,
                                                        cam_name='left_cam',
                                                        _eight_bit=_eight_bit)
        R_ready_zyx, shape_R = single_cam_preprocessing(right_cam_path, parameters,
                                                        output_voxel_size=output_voxel_size,
                                                        max_intensity=max_intensity,
                                                        _save_preprocessed_tomograms=_save_preprocessed_tomograms,
                                                        outfolderpath=output_folderpath,
                                                        cam_name='right_cam',
                                                        _eight_bit=_eight_bit)
        print('Preprocessing done.')
        if _save_preprocessed_tomograms:
            mess_strings.append('Preprocessed tomograms saved in {}'.format(output_folderpath))

    # print and add to .txt
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ===================================== BEST Z POSITION FOR FUSION ==============================
    # TODO: add info to reports txt
    # ===============================================================================================
    _plot             = False
    _save_final_plots = False
    _verb             = False
    _segm             = False
    z_slicing         = z_slicing if z_slicing else 1
    radius            = 0.1

    print(Bcolors.OKBLUE + '\n\n*** Evaluating SNR with radius ratio: {} '.format(radius) + Bcolors.ENDC)
    L_snr_image, L_snr_sample = evaluate_snr_along_z(L_ready_zyx, _segm=_segm, fov_portion=(0, 0.8), z_slicing=z_slicing, radius=radius, _plot=_plot, _verb=_verb)
    R_snr_image, R_snr_sample = evaluate_snr_along_z(R_ready_zyx, _segm=_segm, fov_portion=(0, 0.8), z_slicing=z_slicing, radius=radius, _plot=_plot, _verb=_verb)

    '''
    # double tomogram, double snr
    # plot_snrs_dual(L_snr_image, R_snr_image, _log=False, _segm=False, xlabel='Z', ylabel='SNR',
    #                labels=('LeftCAM SNR (Image - no segm)', 'RightCAM SNR (Image - no segm)'),
    #                title='SNR along Z - step: {}; FFT ratio: {}'.format(z_slicing, radius),
    #                _save=_save_final_plots, _output_dirpath=base_dirpath,
    #                fname='snr_no_segm_step{}_radius{}.png'.format(z_slicing, radius))

    #plot log
    plot_snrs_dual(L_snr_image, R_snr_image, _log=True, _segm=False, xlabel='Z', ylabel='SNR',
                   labels=('LeftCAM SNR (Image - no segm)', 'RightCAM SNR (Image - no segm)'),
                   title='Log10(SNR) along Z - step: {}; FFT ratio: {}'.format(z_slicing, radius),
                   _save=_save_final_plots, _output_dirpath=base_dirpath,
                   fname='log10_snr_no_segm_step{}_radius{}.png'.format(z_slicing, radius))

    # plot_snrs_dual(L_snr_image, L_snr_sample, _segm=_segm, xlabel='z', ylabel='SNR',
    #                labels=('LeftCAM SNR (Image - no segm)', 'LeftCAM SNR (Sample - segmented)'),
    #                title='LeftCAM - SNR along Z - step: {}; FFT ratio 0.2'.format(z_slicing))
    #
    # plot_snrs_dual(R_snr_image, R_snr_sample, _segm=_segm, xlabel='z', ylabel='SNR',
    #                  labels=('RightCAM SNR (Image - no segm)', 'RightCAM SNR (Sample - segmented)'),
    #                  title='RightCAM - SNR along Z - step: {}; FFT ratio 0.2'.format(z_slicing))
    '''

    # save results in a file
    snr_values_out_path = os.path.join(base_dirpath, 'snr_values')
    if not os.path.exists(snr_values_out_path):
        os.makedirs(snr_values_out_path)
    with open(os.path.join(snr_values_out_path, 'L_snr_image_radius{}_zstep{}.txt'.format(radius, z_slicing)), 'w') as file:
        json.dump(L_snr_image, file)
    with open(os.path.join(snr_values_out_path, 'R_snr_image_radius{}_zstep{}.txt'.format(radius, z_slicing)), 'w') as file:
        json.dump(R_snr_image, file)

    # extract the best z value to fuse the two tomograms
    z_fusion = evaluate_best_z(L_snr_image, R_snr_image)
    mess_strings.append(' \n*** Evaluated SNR with radius ratio: {} '.format(radius))
    mess_strings.append('Best z value to fuse the two tomograms: {}'.format(z_fusion))
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # ===============================================================================================
    # ========================== REALIGNMENT ON XY PLANE and SCALE between CAMS  ====================

    # extract frames from L and R tomogram at best z for furion
    L_switch_zframe = L_ready_zyx[z_fusion]
    R_switch_zframe = R_ready_zyx[z_fusion]

    # plot the two frames in red and green on the same plot with two different color to check the alignment
    plot_rgb_frames(L_switch_zframe, R_switch_zframe, title='Alignment Check - Best Z: {}'.format(z_fusion))

    # estimate trasnaltion and scale between L_switch_zframe and R_switch_zframe
    print("Evaluating the xy translation and scale factor of Right tomogram respect to left...")
    right_scale_xy, right_tx, right_ty = estimate_isotropic_scaling_and_translation(L_switch_zframe, R_switch_zframe)
    print('Estimated scale factor: {}'.format(right_scale_xy))
    print('Estimated translation (tx, ty): ({}, {})'.format(right_tx, right_ty))
    mess_strings.append(
        ' \n*** Scaling factor of right cam to left: {}'.format(right_scale_xy))
    mess_strings.append(
        ' \n*** Translation (tx, ty) of right cam to left: ({}, {})'.format(right_tx, right_ty))

    # apply translation and scale to the right tomogram
    R_switch_frame_adjusted = apply_scale_and_translation(R_switch_zframe, right_scale_xy, right_tx, right_ty)
    plot_rgb_frames(L_switch_zframe, R_switch_frame_adjusted, title='Fitting Check')

    # apply the same transformation to the whole tomogram
    print('Realigning the whole tomogram...')
    R_zyx_adjusted = transform_volume_xy(R_ready_zyx, right_scale_xy, right_tx, right_ty, interpolation=cv2.INTER_LINEAR)
    print('Done.')

    # ===============================================================================================
    # ===================================== ACTUAL FUSION OF READY TOMOGRAMS  ==========================================
    print('Fusing the two tomograms...')
    plot_rgb_frames(L_ready_zyx[z_fusion], R_zyx_adjusted[z_fusion],
                    title='3D Realignment Check before fusion - Best Z: {}'.format(z_fusion))
    fused_tomogram = fuse_dual_tomograms(top=R_zyx_adjusted, bottom=L_ready_zyx, z_switch=z_fusion)

    print('Saving the fused tomogram...')
    save_fused_tomogram(fused_tomogram, output_folderpath)

    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    print(Bcolors.OKBLUE + '\n\n**************** End Dual Tomogram Fusion ****************' + Bcolors.ENDC)
    return None


if __name__ == '__main__':
    # define parser
    parser = argparse.ArgumentParser(description='Fuse two DualMesoSPIM tomograms optimizing SNR Ratio')
    parser.add_argument('-l', '--leftCAM_path',
                        type=str, nargs=1, required=True,
                        help='Path to the LEFT Cam1 tomogram (img seq)')
    parser.add_argument('-r', '--rightCAM_path',
                        type=str, nargs=1, required=True,
                        help='Path to the RIGHT Cam2 tomogram (img seq)')
    parser.add_argument('-p', '--parameters-filepath',
                        nargs='+', required=True,
                        help='filepath of parameters.txt file')
    # add output folder if needed
    parser.add_argument('-o', '--output_folderpath',
                        type=str, nargs=1, required=False,
                        help='Path to the output folder')
    parser.add_argument('-sp', '--skip_preprocessing',
                        action='store_true',
                        help='Skip preprocessing step. If passed, LEFT and RIGHT input must be path of tiffiles of'
                             '8bit - 6um pixel size preprocessed tomograms')
    parser.add_argument('-z', '--z-fusion',
                        type=int, nargs=1, required=False, default=[50],
                        help='If passed, fuse tomograms at this z instead of evaluate best z by image quality')
    # add the desidered voxel size of the preprocessed tomograms
    parser.add_argument('-vs', '--voxel_size', default=[6],
                        type=float, nargs=1, required=False,
                        help='Voxel size of the preprocessed tomograms')
    # add parameter of max intensity to rescale histogram
    parser.add_argument('-mi', '--max_intensity', default=[5000],
                        type=float, nargs=1, required=False,
                        help='Max intensity to rescale histogram')
    # addarameter boolean if save preprocessed single views or not
    parser.add_argument('-spt', '--save_preprocessed_tomograms',
                        action='store_true', default=True,
                        help='Save preprocessed tomograms')
    # add parameter of z_slicing for contrast evaluation
    parser.add_argument('-zs', '--z_slicing',
                        type=int, nargs=1, required=False, default=[50],
                        help='Z slicing step for contrast evaluation')
    # add boolean parameter if convert to 8 bit or not
    parser.add_argument('-bit', '--eight_bit',
                        action='store_true', default=True,
                        help='Convert to 8 bit')


    main(parser)