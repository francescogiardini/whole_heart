import os
import argparse
import json
import gc

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import fftconvolve
from scipy.ndimage import shift, zoom
from skimage.filters import threshold_otsu
from tifffile import imread, imwrite
# from skimage.filters.tests.test_median import image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from custom_tool_kit import search_value_in_txt, write_on_txt, Bcolors, manage_path_argument, create_fldr
from custom_image_base_tool import load_tiff_stack_zyx


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

    if z_slicing is None or z_slicing <= 0:
        z_slicing = int(tomogram.shape[0] / 10)  # default slicing every 50 frames
    print('Selected z slicing: ', z_slicing)

    z_selected = range(0, tomogram.shape[0], z_slicing)

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

    plt.close()
    return None


def find_crossing_z(left_data, right_data):
    z_values = list(map(int, left_data.keys()))
    left_values = list(left_data.values())
    right_values = list(right_data.values())

    # scansiona i valori di z (partendo dal 20% e fermandosi all'80%) e trova il primo punto in cui left_values supera right_values
    for i in range(int(len(z_values) * 0.2), int(len(z_values) * 0.8)):
        if left_values[i] > right_values[i]:
            return z_values[i]
    # fail
    return -1


def evaluate_best_z(left_data, right_data):

    return find_crossing_z(left_data, right_data)


def plot_rgb_frames(frame_red, frame_green, title='RGB Plot', red_ch=None, green_ch=None, outpath=None, _save=False):
    """
    Plots two 8-bit frames in an RGB image, with one frame in red and the other in green.
    Optionally adds labels to the legend and saves the plot as a PNG.

    :param frame_red: numpy array of the first frame (8-bit) to be plotted in red
    :param frame_green: numpy array of the second frame (8-bit) to be plotted in green
    :param title: title of the plot
    :param red_ch: label for the red channel (optional)
    :param green_ch: label for the green channel (optional)
    :param outpath: path to save the plot if _save is True
    :param _save: if True, saves the plot as a PNG in the specified outpath
    """
    if frame_red.shape != frame_green.shape:
        raise ValueError('Both frames must have the same dimensions')

    # Aggiorna il titolo con le etichette dei canali
    if red_ch and green_ch:
        title += f" (R: {red_ch}, G: {green_ch})"

    # Create an RGB image
    rgb_image = np.zeros((frame_red.shape[0], frame_red.shape[1], 3), dtype=np.uint8)
    rgb_image[..., 0] = frame_red  # Red channel
    rgb_image[..., 1] = frame_green  # Green channel

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(title)

    plt.axis('off')  # Hide axes

    # Save the plot if _save is True
    if _save and outpath:
        save_path = os.path.join(outpath, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved at: {save_path}")

    plt.close()
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


def plot_matches(img1, img2, kp1, kp2, matches, title='Matches'):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def estimate_isotropic_scaling_and_translation(img1, img2, _plot_matches=False, plot_inliers_only=True, _verb=False):
    # Rileva feature con ORB
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Matcher con ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    if _verb: print("Number of good matches found: {}".format(len(good_matches)))

    if len(good_matches) < 6:
        raise ValueError("Too few good matches ({}) to estimate the transformation.".format(len(good_matches)))

    # Ottieni i punti corrispondenti
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Stima la trasformazione con RANSAC
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3)

    if M is None:
        raise ValueError("Non è stato possibile stimare la trasformazione.")

    # Filtra i match inlier per il plot
    if _plot_matches:
        if plot_inliers_only and inliers is not None:
            inlier_matches = [m for i, m in enumerate(good_matches) if inliers[i]]
            plot_matches(img1, img2, kp1, kp2, inlier_matches, title='Inlier Matches (RANSAC)')
        else:
            plot_matches(img1, img2, kp1, kp2, good_matches, title='Good Matches (ratio test)')

    # Calcola la scala isotropica
    scale_x = np.linalg.norm(M[0, :2])
    scale_y = np.linalg.norm(M[1, :2])
    scale = (scale_x + scale_y) / 2

    # Estrai traslazione
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

# resample tomogram but saving RAM
def resample_volume_slicewise(tomogram, scale_z, scale_xy):
    # Prima: scala XY per ogni slice
    slices_scaled = [
        zoom(slice_, zoom=scale_xy, order=1)  # slice_ è 2D
        for slice_ in tomogram
    ]
    # Stack lungo Z
    stack_xy = np.stack(slices_scaled)

    # Poi: riscalo lungo Z
    if scale_z != 1.0:
        rescaled = zoom(stack_xy, (scale_z, 1, 1), order=1)
    else:
        rescaled = stack_xy

    return rescaled


def single_cam_preprocessing(imgseq_path, parameters, output_voxel_size=None, _scale_is_needed=False, max_intensity=None,
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
    tiff_files = sorted([os.path.join(imgseq_path, f) for f in os.listdir(imgseq_path) if f.endswith(('.tif', '.tiff'))])
    tomogram = np.stack([imread(f) for f in tiff_files], axis=0)  # (z, y, x)

    # ===============================
    # SCALING DIMENSIONS
    # ===============================
    if _scale_is_needed:
        scale_xy = parameters['px_size_xy'] / output_voxel_size[1]
        scale_z = parameters['px_size_z'] / output_voxel_size[0]
        # Apply scaling
        print('Scaling tomogram to voxel size: ', output_voxel_size, '  ...')
        # DEPRECATED (Memory intensive), see below
        # tomogram_scaled = zoom(tomogram, (scale_z, scale_xy, scale_xy), order=1)  # Linear interpolation
        tomogram_scaled = resample_volume_slicewise(tomogram, scale_z, scale_xy)
        del tomogram  # Free memory
        gc.collect()
    else:
        # Not Apply scaling
        print('No scaling applied. Using original voxel size: ', output_voxel_size)
        tomogram_scaled = tomogram

    # ===============================
    # EQUALIZATION
    # ===============================
    if max_intensity:
        # Rescale the histogram to the specified maximum intensity
        print('Rescaling histogram to maximum intensity {} '.format(max_intensity))
        tomogram_scaled = equalize_max_with_clip_inplace(tomogram_scaled, max_intensity, dtype=np.uint16)
    else:
        # Not Apply equalization
        print('No equalization applied. Using original intensity range')

    if _eight_bit:
        print('Rescaling histogram to maximum intensity 255 and convert to 8bit ')
        tomogram_scaled = convert_16bit_to_8bit_inplace(img16=tomogram_scaled)

    # Save the preprocessed tomogram if required
    if _save_preprocessed_tomograms and outfolderpath and cam_name:

        # prepare the out fname
        if _scale_is_needed:
            ds_suffix = '_ps{:.2f}_{:.2f}_{:.2f}um'.format(output_voxel_size[0], output_voxel_size[1], output_voxel_size[2])
        else:
            ds_suffix = ''
        outfname = cam_name + ds_suffix + '_eq{}.tif'.format(
            int(max_intensity))
        outfolderpath = os.path.join(outfolderpath, outfname)
        print('Saving preprocessed tomogram to {}...'.format(outfolderpath))
        imwrite(outfolderpath, tomogram_scaled, imagej=True)

    return tomogram_scaled, tomogram_scaled.shape


'''
DEPREACTED BECAUSE OF MEMORY
top[...] e bottom[...] → creano due nuove copie in RAM
np.concatenate(...) → crea una terza copia finale
Totale: RAM = ~3× slice size

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
'''


def fuse_dual_tomograms(top: np.ndarray, bottom: np.ndarray, z_switch: int) -> np.ndarray:
    """
    Fuse two tomograms along Z, using:
    - top[:z_switch+1]
    - bottom[z_switch+1:]
    Avoids duplication of z_switch frame.

    Returns:
        np.ndarray: fused tomogram (Z, Y, X)
    """
    # define the depths of top and bottom tomograms
    z_top = z_switch + 1
    z_bottom = bottom.shape[0] - z_top

    # define the output shape and prepare empty array
    output_shape = (z_top + z_bottom, *top.shape[1:])
    fused = np.empty(output_shape, dtype=top.dtype)

    # Fill the fused tomogram
    fused[:z_top] = top[:z_top]
    fused[z_top:] = bottom[z_top:]

    return fused


def save_tomogram_for_fiji(tomogram, output_folderpath, voxel_size, fname='sample', axes='ZYX', _verb=False):
    """
    Salva un tomogramma 3D come file TIFF compatibile con Fiji.

    :param tomogram: numpy array del tomogramma (z, y, x)
    :param output_folderpath: stringa, percorso della cartella di output
    :param voxel_size: dimensione del voxel in micrometri (float o tuple per ZYX)
    :param parameters: dizionario opzionale con parametri aggiuntivi (es. 'px_size_xy', 'px_size_z')
    :param fname: nome del campione per il file di output (opzionale)
    :param axes: stringa che specifica l'ordine degli assi (default: 'ZYX')
    :param dtype: tipo di dato per il salvataggio (default: np.uint16)
    :return: percorso del file salvato
    """
    # Verifica che il tomogramma sia un array 3D
    if len(tomogram.shape) != 3:
        raise ValueError("Il tomogramma deve essere un array numpy 3D (z, y, x).")

    # Determina la dimensione del voxel
    if isinstance(voxel_size, (float, int)):
        voxel_size = (voxel_size, voxel_size, voxel_size)
    elif len(voxel_size) != 3:
        raise ValueError("voxel_size deve essere un singolo valore o una tupla di 3 valori (Z, Y, X).")

    # Salva l'array 3D come file TIFF
    fname = '{}_fused_ps{}_{}_{}um.tif'.format(fname, voxel_size[0], voxel_size[1], voxel_size[2])
    output_filepath = os.path.join(output_folderpath, fname)

    imwrite(output_filepath,
            tomogram,
            imagej=True,
            metadata={
                'axes': axes,
                'spacing': voxel_size[0],  # distanza tra slice (Z)
                'unit': 'um',  # unità di misura
            },
            resolution=(1.0 / voxel_size[2], 1.0 / voxel_size[1]),  # XY resolution (from ZYX format)
            )
    if _verb:
        print(f"Tomogram saved with voxel size {voxel_size} and axes {axes} in: \n{output_filepath}")

    return output_filepath


'''
def save_fused_tomogram(fused_tomogram, output_folderpath, _skip_preprocessing,
                                      output_voxel_size, parameters, sample_name=''):
    """
    Save the 3D fused tomogram as a TIFF file with correct axis order.

    :param fused_tomogram: numpy array of the fused tomogram (z, y, x)
    :param output_filepath: string, path to save the output TIFF file
    """
    # Ensure the data is in the correct format (z, y, x)
    if len(fused_tomogram.shape) != 3:
        raise ValueError("Fused tomogram must be a 3D numpy array (z, y, x).")

    # create metadata for the TIFF file
    if _skip_preprocessing:
        metadata = {
            'axes': 'ZYX',
            'spacing': parameters['px_size_z'],  # distanza tra slice (Z)
            'unit': 'um',  # unità di misura, può essere anche 'micron'
            'resolution': (1.0 / parameters['px_size_xy'], 1.0 / parameters['px_size_xy']),  # in pixel per µm
        }
    else:
        metadata = {
            'axes': 'ZYX',
            'spacing': output_voxel_size,  # distanza tra slice (Z)
            'unit': 'um',  # unità di misura, può essere anche 'micron'
            'resolution': (1.0 / output_voxel_size, 1.0 / output_voxel_size),  # in pixel per µm
        }

    # Save the 3D array as a TIFF file
    fname = '{}_fused_tomogram.tif'.format(sample_name) if sample_name else 'fused_tomogram.tif'
    output_filepath = os.path.join(output_folderpath, fname)
    imsave(output_filepath, fused_tomogram, imagej=True, metadata=metadata)
    return output_filepath
'''


def save_fused_tomogram(fused_tomogram, output_folderpath,
                        out_voxel_size, parameters, sample_name='', _verb=False):
    """
    Salva il tomogramma fuso utilizzando la funzione save_tomogram_for_fiji.

    :param fused_tomogram: numpy array del tomogramma fuso (z, y, x)
    :param output_folderpath: stringa, percorso della cartella di output
    :param _skip_preprocessing: booleano, indica se saltare il preprocessing
    :param out_voxel_size: dimensione del voxel in micrometri
    :param parameters: dizionario con parametri aggiuntivi
    :param sample_name: nome del campione per il file di output (opzionale)
    :return: percorso del file salvato
    """


    # print("voxel_size for saving: ", out_voxel_size)

    # Salva il tomogramma fuso come file TIFF compatibile con Fiji
    output_filepath = save_tomogram_for_fiji(
        tomogram=fused_tomogram,
        output_folderpath=output_folderpath,
        voxel_size=out_voxel_size,
        fname=sample_name,
        axes='ZYX',
        _verb=_verb
    )

    return output_filepath

'''
NOT USED ANYMORE
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
'''


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

'''
==================================
DEPRECATED BECAUSE OF MEMORY ISSUES   -   see below for the inplace version
np.clip(image, 0, max_intensity) → nuova copia in RAM
* scale_factor → altra copia temporanea (probabilmente in float64)
.astype(dtype) → altra copia finale

def equalize_max_with_clip(image, max_intensity, dtype=np.uint16):
    max_val = np.iinfo(dtype).max  # es. 65535 for uint16
    scale_factor = max_val / max_intensity

    # linear transformation
    rescaled = np.clip(image, 0, max_intensity) * scale_factor
    return rescaled.astype(dtype)
==================================
'''
def equalize_max_with_clip_inplace(volume, max_intensity, dtype=np.uint16):
    max_val = np.iinfo(dtype).max
    scale_factor = max_val / max_intensity
    output = np.empty_like(volume, dtype=dtype)

    for z in range(volume.shape[0]):
        # Clipping + scaling per slice
        slice_ = np.clip(volume[z], 0, max_intensity)
        slice_ = slice_ * scale_factor
        output[z] = slice_.astype(dtype)

    return output


'''
# DEPRECATED BECAUSE OF MEMORY ISSUES
# (img16 / 256) → crea un array temporaneo in float64
# .astype('uint8') → nuova copia finale in uint8
def convert_16bit_to_8bit(img16):
    return (img16/256).astype('uint8')
'''

def convert_16bit_to_8bit_inplace(img16):
    z, y, x = img16.shape
    img8 = np.empty((z, y, x), dtype=np.uint8)

    for i in range(z):
        # Cast direttamente in uint8 evitando float64 temporanei
        img8[i] = (img16[i] >> 8).astype(np.uint8)  # divide by 256, safer and faster

    return img8

def manage_desired_voxel_size(voxel_size, skip_preprocessing, parameters, voxel_size_fpath=None):
    """
    Manage the input voxel size based on the provided arguments.

    :param voxel_size: list or None, input voxel size
    :param skip_preprocessing: boolean, whether to skip preprocessing
    :return: voxel size as a float or tuple of floats
    """

    if skip_preprocessing:
        # load dict from voxel_size_fpath
        with open(voxel_size_fpath, 'r') as f:
            voxel_size_dict = json.load(f)
        out_voxel_size = (
        float(voxel_size_dict['px_size_z']), float(voxel_size_dict['px_size_xy']), float(voxel_size_dict['px_size_xy']))
        _scale_is_needed = False
        return out_voxel_size, _scale_is_needed

    else:
        if voxel_size is None:
            # use the original voxel size from parameters
            out_voxel_size = (float(parameters['px_size_z']), float(parameters['px_size_xy']), float(parameters['px_size_xy']))
            _scale_is_needed = False
            return out_voxel_size, _scale_is_needed

        else:
            if len(voxel_size) == 1:
                 # isotropic voxel size
                 out_voxel_size = (float(voxel_size[0]), float(voxel_size[0]), float(voxel_size[0]))
                 _scale_is_needed = True
                 return out_voxel_size, _scale_is_needed

            if len(voxel_size) == 3:
                # check if voxel_size is the same of input data
                if voxel_size[0] != parameters['px_size_z'] or voxel_size[1] != parameters['px_size_xy'] or voxel_size[2] != parameters['px_size_xy']:
                    # scaling is needed
                    _scale_is_needed = True
                    return voxel_size, _scale_is_needed
                else:
                    # no scaling needed
                    _scale_is_needed = False
                    return voxel_size, _scale_is_needed


def save_output_voxel_size_as_dict(output_voxel_size, preproc_output_fpath):
    """    Save the output voxel size in a JSON file.
    :param output_voxel_size: tuple, voxel size in (px_size_z, px_size_xy, px_size_xy)
    :param preproc_output_fpath: string, path to the preprocessing output folder
    :param mess_strings: list, messages to be logged
    """

    # save output_voxel_size in a file as dict
    voxel_size_dict = {'px_size_xy': output_voxel_size[1], 'px_size_z': output_voxel_size[0]}
    voxel_size_fpath = os.path.join(preproc_output_fpath, 'preprocess_voxel_size.json')
    with open(voxel_size_fpath, 'w') as f:
        json.dump(voxel_size_dict, f)
    return voxel_size_fpath



def main(parser):

    # ===============================================================================================
    # ===================================== INITIAL OPERATIONS ======================================
    # ===============================================================================================

    # read parameters
    args = parser.parse_args()
    left_cam_path = manage_path_argument(args.leftCAM_path)
    right_cam_path = manage_path_argument(args.rightCAM_path)
    parameter_filepath = args.parameters_filepath[0]
    sample_name = args.sample_name
    base_dirpath = os.path.dirname(left_cam_path)
    output_folderpath = args.output_folderpath[0] if args.output_folderpath else base_dirpath
    z_fusion = args.z_fusion[0] if args.z_fusion else None
    _skip_preprocessing = args.skip_preprocessing
    max_intensity = args.max_intensity[0] if args.max_intensity else None
    _save_preprocessed_tomograms = args.save_preprocessed_tomograms
    z_slicing = args.z_slicing[0] if args.z_slicing else None
    _eight_bit = args.eight_bit
    fft_radius = args.fft_radius[0]
    fov_perc = args.field_of_view[0]

    if not _skip_preprocessing:
        # create preprocessing folder
        preproc_output_fpath = os.path.join(output_folderpath, 'preprocessing')
        create_fldr(preproc_output_fpath)
        voxel_size_fpath = None
    else:
        preproc_output_fpath = base_dirpath
        voxel_size_fpath = os.path.join(preproc_output_fpath, 'preprocess_voxel_size.json')

    # check if the input files exist
    if not os.path.exists(left_cam_path):
        raise ValueError('Left CAM tomogram not found')
    if not os.path.exists(right_cam_path):
        raise ValueError('Right CAM tomogram not found')

    # create report txt file
    report_filepath = os.path.join(preproc_output_fpath, 'fusion_report.txt')

    # prepare initial messages for console and report.txt
    mess_strings = prepare_initial_report(args, preproc_output_fpath, report_filepath)

    # print to screen, create .txt file and write into .txt file all temporal information
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
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

    output_voxel_size, _scale_is_needed = manage_desired_voxel_size(args.voxel_size, _skip_preprocessing, parameters, voxel_size_fpath)

    # print to screen, create .txt file and write into .txt file all temporal informations
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ===================================== PREPROCESSING  ==========================================
    # ----------   from original DUAL_mesoSPIM image sequence to 8bit 6um tiffile -------------------
    # ===============================================================================================

    if _skip_preprocessing:
        mess_strings.append(Bcolors.WARNING + '\n\n*** Preprocessing informations:' + Bcolors.ENDC)
        mess_strings.append(' ATTENTION > Skip preprocessing step')
        print('Loading LEFT_CAM  tomogram from {}...'.format(left_cam_path))
        L_ready_zyx, shape_L = load_tiff_stack_zyx(left_cam_path) # (r, c, z) -> (z, y, x)
        print('Loading RIGHT_CAM  tomogram from {}...'.format(right_cam_path))
        R_ready_zyx, shape_R = load_tiff_stack_zyx(right_cam_path) # (r, c, z) -> (z, y, x)

        print('Done.')

        # add info to the report
        mess_strings.append(' > Loaded already preproccesed tomograms:')
        mess_strings.append(' > LEFT CAM tomogram: {}'.format(left_cam_path))
        mess_strings.append(' > RIGHT CAM tomogram: {}'.format(right_cam_path))
        mess_strings.append(' > Dimension of LEFT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_L[0], shape_L[1], shape_L[2]))
        mess_strings.append(' > Dimension of RIGHT CAM tomogram [pixel]: ({}, {}, {})'.
                            format(shape_R[0], shape_R[1], shape_R[2]))
    else:
        mess_strings.append(Bcolors.OKBLUE + '\n*** Preprocessing informations:' + Bcolors.ENDC)
        if _scale_is_needed:
            mess_strings.append(' > Scaling to voxel size [ZYX, um]: {}, {}, {}'.format(output_voxel_size[0], output_voxel_size[1], output_voxel_size[2]))
        else:
            mess_strings.append(' > No scaling applied, using original voxel size [ZYX, um]: {}, {}, {}'.format(output_voxel_size[0], output_voxel_size[1], output_voxel_size[2]))
        mess_strings.append(' > Rescaling histogram to maximum intensity {}'.format(max_intensity))
        # print and add to .txt
        write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
        # clear list of strings
        mess_strings.clear()

        print('Preprocessing is running....')
        L_ready_zyx, shape_L = single_cam_preprocessing(left_cam_path, parameters,
                                                        output_voxel_size=output_voxel_size,
                                                        _scale_is_needed=_scale_is_needed,
                                                        max_intensity=max_intensity,
                                                        _save_preprocessed_tomograms=_save_preprocessed_tomograms,
                                                        outfolderpath=preproc_output_fpath,
                                                        cam_name='left_cam',
                                                        _eight_bit=_eight_bit)
        R_ready_zyx, shape_R = single_cam_preprocessing(right_cam_path, parameters,
                                                        output_voxel_size=output_voxel_size,
                                                        _scale_is_needed=_scale_is_needed,
                                                        max_intensity=max_intensity,
                                                        _save_preprocessed_tomograms=_save_preprocessed_tomograms,
                                                        outfolderpath=preproc_output_fpath,
                                                        cam_name='right_cam',
                                                        _eight_bit=_eight_bit)

        print('Preprocessing done.')
        if _save_preprocessed_tomograms:
            voxel_size_fpath = save_output_voxel_size_as_dict(output_voxel_size, preproc_output_fpath)
            mess_strings.append(
                ' Voxel size of preprocessed LEFT and RIGHT tomograms saved as: \n{}'.format(voxel_size_fpath))
            mess_strings.append('Preprocessed tomograms saved in:  \n{}'.format(preproc_output_fpath))

    # print and add to .txt
    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    # clear list of strings
    mess_strings.clear()

    # ===============================================================================================
    # ===================================== BEST Z POSITION FOR FUSION ==============================
    _plot             = False
    _save_final_plots = True
    _verb             = False
    _segm             = False

    print(Bcolors.OKBLUE + '\n*** Evaluating SNR with radius ratio: {} '.format(fft_radius) + Bcolors.ENDC)
    L_snr_image, L_snr_sample = evaluate_snr_along_z(L_ready_zyx, _segm=_segm, fov_portion=(0, fov_perc/100), z_slicing=z_slicing, radius=fft_radius, _plot=_plot, _verb=_verb)
    R_snr_image, R_snr_sample = evaluate_snr_along_z(R_ready_zyx, _segm=_segm, fov_portion=(0, fov_perc/100), z_slicing=z_slicing, radius=fft_radius, _plot=_plot, _verb=_verb)

    # double tomogram, double snr
    plot_snrs_dual(L_snr_image, R_snr_image, _log=False, _segm=False, xlabel='Z', ylabel='SNR',
                   labels=('LeftCAM SNR (Image - no segm)', 'RightCAM SNR (Image - no segm)'),
                   title='SNR along Z - step: {}; FFT ratio: {}'.format(z_slicing, fft_radius),
                   _save=_save_final_plots, _output_dirpath=preproc_output_fpath,
                   fname='snr_no_segm_step{}_radius{}.png'.format(z_slicing, fft_radius))

    #plot log
    # plot_snrs_dual(L_snr_image, R_snr_image, _log=True, _segm=False, xlabel='Z', ylabel='SNR',
    #                labels=('LeftCAM SNR (Image - no segm)', 'RightCAM SNR (Image - no segm)'),
    #                title='Log10(SNR) along Z - step: {}; FFT ratio: {}'.format(z_slicing, radius),
    #                _save=_save_final_plots, _output_dirpath=base_dirpath,
    #                fname='log10_snr_no_segm_step{}_radius{}.png'.format(z_slicing, radius))

    # plot_snrs_dual(L_snr_image, L_snr_sample, _segm=_segm, xlabel='z', ylabel='SNR',
    #                labels=('LeftCAM SNR (Image - no segm)', 'LeftCAM SNR (Sample - segmented)'),
    #                title='LeftCAM - SNR along Z - step: {}; FFT ratio 0.2'.format(z_slicing))
    #
    # plot_snrs_dual(R_snr_image, R_snr_sample, _segm=_segm, xlabel='z', ylabel='SNR',
    #                  labels=('RightCAM SNR (Image - no segm)', 'RightCAM SNR (Sample - segmented)'),
    #                  title='RightCAM - SNR along Z - step: {}; FFT ratio 0.2'.format(z_slicing))

    # save results in a file
    snr_values_out_path = os.path.join(preproc_output_fpath, 'snr_values')
    if not os.path.exists(snr_values_out_path):
        os.makedirs(snr_values_out_path)
    with open(os.path.join(snr_values_out_path, 'L_snr_image_radius{}_zstep{}.txt'.format(fft_radius, z_slicing)), 'w') as file:
        json.dump(L_snr_image, file)
    with open(os.path.join(snr_values_out_path, 'R_snr_image_radius{}_zstep{}.txt'.format(fft_radius, z_slicing)), 'w') as file:
        json.dump(R_snr_image, file)

    if z_fusion is None:
    # extract the best z value to fuse the two tomograms
        z_fusion = evaluate_best_z(L_snr_image, R_snr_image) if z_fusion is None else z_fusion
        mess_strings.append(' \n*** Evaluated SNR with radius ratio: {} '.format(fft_radius))
        mess_strings.append('Best z value to fuse the two tomograms: {}'.format(z_fusion))
    else:
        mess_strings.append(' \n*** Using z_fusion value passed as argument: {}'.format(z_fusion))

    write_on_txt(mess_strings, report_filepath, _print=True, mode='a')
    mess_strings.clear()

    # ===============================================================================================
    # ========================== REALIGNMENT ON XY PLANE and SCALE between CAMS  ====================

    # extract frames from L and R tomogram at best z for furion
    L_switch_zframe = L_ready_zyx[z_fusion]
    R_switch_zframe = R_ready_zyx[z_fusion]

    # plot the two frames in red and green on the same plot with two different color to check the alignment
    plot_rgb_frames(L_switch_zframe, R_switch_zframe, title='Alignment Check - Best Z for Fusion: {}'.format(z_fusion),
                    red_ch='Left CAM', green_ch='Right CAM', outpath=preproc_output_fpath, _save=_save_final_plots)

    # estimate translation and scale between L_switch_zframe and R_switch_zframe
    print("Evaluating the xy translation and scale factor of Right tomogram respect to left...")
    right_scale_xy, right_tx, right_ty = estimate_isotropic_scaling_and_translation(L_switch_zframe, R_switch_zframe)
    print('Estimated scale factor: {}'.format(right_scale_xy))
    print('Estimated translation (tx, ty)[um]: ({}, {}) um'.format(right_tx * output_voxel_size[2], right_ty * output_voxel_size[2]))
    mess_strings.append(
        ' \n*** Scaling factor of right cam to left: {}'.format(right_scale_xy))
    mess_strings.append(
        ' \n*** Translation (tx, ty) [um] of right cam to left: ({}, {}) um'.format(right_tx * output_voxel_size[2], right_ty * output_voxel_size[2]))

    # apply translation and scale to the right frame
    R_switch_frame_adjusted = apply_scale_and_translation(R_switch_zframe, right_scale_xy, right_tx, right_ty)
    plot_rgb_frames(L_switch_zframe, R_switch_frame_adjusted, title='After registration - frames at z: {}'.format(z_fusion),
                    red_ch='Left CAM', green_ch='Right CAM', outpath=preproc_output_fpath, _save=_save_final_plots)

    # apply the same transformation to the whole tomogram
    print('Realigning the whole tomogram...', end=' ')
    R_zyx_adjusted = transform_volume_xy(R_ready_zyx, right_scale_xy, right_tx, right_ty, interpolation=cv2.INTER_LINEAR)
    del R_ready_zyx  # Free memory
    print('Done.')

    # ===============================================================================================
    # ===================================== ACTUAL FUSION OF READY TOMOGRAMS  ==========================================
    print('Fusing the two tomograms...', end=' ')
    plot_rgb_frames(L_ready_zyx[z_fusion], R_zyx_adjusted[z_fusion],
                    title='3D Realignment Check before fusion - Z: {}'.format(z_fusion),
                    red_ch='Left CAM', green_ch='Right CAM', outpath=preproc_output_fpath, _save=_save_final_plots)
    fused_tomogram = fuse_dual_tomograms(top=R_zyx_adjusted, bottom=L_ready_zyx, z_switch=z_fusion)
    print('Done.')
    del L_ready_zyx, R_zyx_adjusted  # Free memory

    print('Saving the fused tomogram...')
    fused_fpath = save_fused_tomogram(fused_tomogram=fused_tomogram, output_folderpath=output_folderpath,
                                      out_voxel_size=output_voxel_size,
                                      parameters=parameters, sample_name=sample_name, _verb=True)
    mess_strings.append(
        'Fused tomogram saved in: \n{}'.format(fused_fpath))

    write_on_txt(mess_strings, report_filepath, _print=False, mode='a')
    mess_strings.clear()

    print(Bcolors.OKBLUE + '\n**************** End Dual Tomogram Fusion ****************\n\n' + Bcolors.ENDC)
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
    parser.add_argument('-sm', '--sample_name',
                        type=str, nargs=1, required=False, default='sample',
                        help='Name of the sample. Default: "sample". ')
    # add output folder if needed
    parser.add_argument('-o', '--output_folderpath',
                        type=str, nargs=1, required=False,
                        help='Path to the output folder')
    # add FOV_portion
    parser.add_argument('-fov', '--field_of_view', default=[100],
                        type=float, nargs=1, required=False,
                        help='Portion of the FoV where estimate image contrast. Example: "80" means the 80% of FoV from the top part. Default: 100')
    parser.add_argument('-sp', '--skip_preprocessing',
                        action='store_true',
                        help='Skip preprocessing step. If passed, LEFT and RIGHT input must be path of tiffiles of'
                             '8bit preprocessed tomograms')
    parser.add_argument('-z', '--z-fusion',
                        type=int, nargs=1, required=False, default=None,
                        help='If passed, fuse tomograms at this z instead of evaluate best z by image quality')
    # add the desidered voxel size of the preprocessed tomograms
    parser.add_argument('-vs', '--voxel_size', default=None,
                        type=float, nargs='+', required=False,
                        help='Output voxel size (ZYX, in um). If not passed, no scaling is applied. Example: "6.0" for isotropic scaling or "6.0 6.0 10.0" for anisotropic scaling.')
    # add parameter of max intensity to rescale histogram
    parser.add_argument('-mi', '--max_intensity', default=[65000],
                        type=float, nargs=1, required=False,
                        help='Max intensity to rescale histogram')
    # addarameter boolean if save preprocessed single views or not
    parser.add_argument('-spt', '--save_preprocessed_tomograms',
                        action='store_true', default=True,
                        help='Save preprocessed tomograms')
    # add parameter of z_slicing for contrast evaluation
    parser.add_argument('-zs', '--z_slicing',
                        type=int, nargs=1, required=False, default=None,
                        help='Z slicing step for contrast evaluation')
    # add boolean parameter if convert to 8 bit or not
    parser.add_argument('-bit', '--eight_bit',
                        action='store_true', default=True,
                        help='Convert to 8 bit')
    # add radius for SNR evaluation
    parser.add_argument('--fft_radius',
                        type=float, nargs=1, required=False, default=[0.1],
                        help='Radius for SNR evaluation (default: 0.1)')
    main(parser)