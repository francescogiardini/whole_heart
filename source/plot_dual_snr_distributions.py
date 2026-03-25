# python
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Hardcoded controls
INIZIO = 4               # skip first N samples
FINE = 5                # skip last N samples
TICK_STEP_SAMPLES = 5    # show one xtick every N samples
MICRON_PER_Z = 60        # conversion factor z -> microns

def read_snr_files(directory):
    left_cam_data = {}
    right_cam_data = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt') and (filename.startswith('L_') or filename.startswith('R_')):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                if filename.startswith('L_'):
                    radius = filename.split('_')[3].replace('radius', '')
                    left_cam_data[float(radius)] = data
                elif filename.startswith('R_'):
                    radius = filename.split('_')[3].replace('radius', '')
                    right_cam_data[float(radius)] = data

    return dict(sorted(left_cam_data.items())), dict(sorted(right_cam_data.items()))

def find_crossing_z(left_data, right_data):
    z_values = list(map(int, left_data.keys()))
    left_values = list(left_data.values())
    right_values = list(right_data.values())

    for i in range(10, len(z_values) - 10):
        if left_values[i] > right_values[i]:
            return z_values[i]
    return None

def save_crossing_z(crossing_z_dict, output_dir):
    with open(os.path.join(output_dir, 'crossing_z.txt'), 'w') as file:
        for radius, crossing_z in crossing_z_dict.items():
            file.write(f'Radius {radius}: {crossing_z}\n')

def _compute_global_window(left_cam_data):
    # choose reference z list from the first radius
    first_left = next(iter(left_cam_data.values()))
    all_z_samples = sorted(map(int, first_left.keys()))
    start_idx = INIZIO
    end_idx = max(start_idx + 1, len(all_z_samples) - FINE)  # ensure at least one sample
    return np.array(all_z_samples), start_idx, end_idx

def plot_snr_data(left_cam_data, right_cam_data, output_dir, crossing_z_dict):
    plt.figure(figsize=(20, 10))
    symbols = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    all_z_samples, start_idx, end_idx = _compute_global_window(left_cam_data)
    all_z_microns = all_z_samples * MICRON_PER_Z

    # xticks: every TICK_STEP_SAMPLES samples within the plotted window
    tick_indices = np.arange(start_idx, end_idx, TICK_STEP_SAMPLES)
    xtick_positions = all_z_microns[tick_indices]

    for idx, (radius, left_data) in enumerate(left_cam_data.items()):
        right_data = right_cam_data[radius]
        z_samples = sorted(map(int, left_data.keys()))
        # slice according to INIZIO / FINE
        z_sliced = z_samples[start_idx:end_idx]
        if len(z_sliced) == 0:
            continue
        z_microns = np.array(z_sliced) * MICRON_PER_Z

        left_values = [left_data[str(z)] for z in z_sliced]
        right_values = [right_data[str(z)] for z in z_sliced]
        left_snr_values = np.log10(left_values)
        right_snr_values = np.log10(right_values)

        crossing_z = crossing_z_dict.get(radius)
        if crossing_z:
            print(f'Radius {radius}: L exceeds R at z = {crossing_z}')

        symbol = symbols[idx % len(symbols)]
        plt.plot(z_microns, left_snr_values, f'{symbol}-', color='b', label=f'LeftCAM radius {radius}')
        plt.plot(z_microns, right_snr_values, f'{symbol}-', color='g', label=f'RightCAM radius {radius}')

        # annotate crossing only if inside plotted window
        if crossing_z and (crossing_z in z_sliced):
            x_text = crossing_z * MICRON_PER_Z + MICRON_PER_Z  # offset by one sample in microns
            plt.text(x_text, np.log10(left_data[str(crossing_z)]) + 0.03, str(crossing_z), color='black', fontsize=12, ha='right')

    plt.xlabel('z (µm)')
    plt.ylabel('log10(SNR)')
    plt.title('DualMesoSPIM: Image Quality along z by High Frequency analysis')
    plt.legend()
    plt.xticks(xtick_positions)
    plt.xlim(all_z_microns[start_idx] - 0.1 * MICRON_PER_Z, all_z_microns[end_idx - 1] + 0.1 * MICRON_PER_Z)

    _save_figure(output_dir, 'snr_ratio_dual_camera_by_radius')
    plt.show()


def _save_figure(output_dir, basename, dpi_png=200, dpi_pdf=300):
    """
    Save current matplotlib figure as both PNG and PDF in `output_dir` using `basename`.
    """
    png_path = os.path.join(output_dir, f'{basename}.png')
    pdf_path = os.path.join(output_dir, f'{basename}.pdf')
    plt.savefig(png_path, dpi=dpi_png, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=dpi_pdf, bbox_inches='tight')


# python
def plot_ratio_data(left_cam_data, right_cam_data, output_dir, crossing_z_dict):
    plt.figure(figsize=(20, 10))
    symbols = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.plasma(np.linspace(0.9, 0, len(left_cam_data)))

    # get global z sample list and defaults
    all_z_samples, default_start_idx, default_end_idx = _compute_global_window(left_cam_data)
    all_z_microns = all_z_samples * MICRON_PER_Z

    # Desired plotting window in microns (user request)
    Z_MIN_MICRON = 1000
    Z_MAX_MICRON = 6000

    # convert micron window to sample indices on the global z grid
    start_sample_idx = int(np.searchsorted(all_z_microns, Z_MIN_MICRON, side='left'))
    end_sample_idx = int(np.searchsorted(all_z_microns, Z_MAX_MICRON, side='right'))

    # if the converted window is invalid, fall back to defaults from _compute_global_window
    if end_sample_idx <= start_sample_idx:
        start_idx = default_start_idx
        end_idx = default_end_idx
    else:
        # also respect the INIZIO / FINE defaults (don't go outside them)
        start_idx = max(default_start_idx, start_sample_idx)
        end_idx = min(default_end_idx, end_sample_idx)

    # xticks: every TICK_STEP_SAMPLES samples within the plotted window
    tick_indices = np.arange(start_idx, end_idx, TICK_STEP_SAMPLES)
    xtick_positions = all_z_microns[tick_indices]

    for idx, (radius, left_data) in enumerate(left_cam_data.items()):
        right_data = right_cam_data[radius]
        z_samples = sorted(map(int, left_data.keys()))
        # slice according to the computed sample-window (converted from microns)
        z_sliced = z_samples[start_idx:end_idx]
        if len(z_sliced) == 0:
            continue
        z_microns = np.array(z_sliced) * MICRON_PER_Z

        # calculate normalized contrast (R-L)/(R+L) with safe handling of zero/NaN
        ratio_values = []
        for z in z_sliced:
            r = right_data.get(str(z), np.nan)
            l = left_data.get(str(z), np.nan)
            denom = r + l
            ratio_values.append((r - l) / denom if denom != 0 and not (np.isnan(r) or np.isnan(l)) else np.nan)

        symbol = symbols[idx % len(symbols)]
        plt.plot(z_microns, ratio_values, '-', color=colors[idx], label=f'R = {radius}')

    # reference for equal intensities in normalized scale
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Depth in the sample from CAM2 to CAM1 (µm)')
    plt.ylabel('Contrast Ratio (C2-C1)/(C2+C1)')
    plt.title('Contrast Ratio (CAM2/CAM1) by High Frequency analysis')
    plt.legend(loc='lower left')
    plt.xticks(xtick_positions)
    # set x-limits according to the chosen sample window
    plt.xlim(all_z_microns[start_idx] - 0.1 * MICRON_PER_Z, all_z_microns[end_idx - 1] + 0.1 * MICRON_PER_Z)

    # Create inset plot around crossing points if any valid crossings inside the plotted window
    valid_crossings = [v for v in crossing_z_dict.values() if v is not None]
    valid_crossings_in_window = [v for v in valid_crossings if start_idx <= v < end_idx]

    if valid_crossings_in_window:
        ax_inset = plt.gca().inset_axes([0.65, 0.5, 0.3, 0.4])  # x0, y0, width, height
        min_crossing = min(valid_crossings_in_window)
        max_crossing = max(valid_crossings_in_window)
        inset_start_idx = max(start_idx, min_crossing - 4)
        inset_end_idx = min(end_idx, max_crossing + 4)
        global_inset_indices = np.arange(inset_start_idx, inset_end_idx)
        inset_xticks = global_inset_indices * MICRON_PER_Z

        for idx, (radius, left_data) in enumerate(left_cam_data.items()):
            right_data = right_cam_data[radius]
            z_samples = sorted(map(int, left_data.keys()))
            ratio_map = {}
            for z in z_samples:
                rs = str(z)
                if rs in right_data and rs in left_data:
                    r = right_data[rs]
                    l = left_data[rs]
                    denom = r + l
                    ratio_map[z] = (r - l) / denom if denom != 0 else np.nan

            y_vals = [ratio_map.get(g, np.nan) for g in global_inset_indices]
            x_vals = global_inset_indices * MICRON_PER_Z

            symbol = symbols[idx % len(symbols)]
            ax_inset.plot(x_vals, y_vals, '-', color=colors[idx], label=f'R/L radius {radius}')

        ax_inset.axhline(y=0, color='k', linestyle='--')
        ax_inset.set_xticks(inset_xticks)
        ax_inset.set_xlim(global_inset_indices[0] * MICRON_PER_Z, global_inset_indices[-1] * MICRON_PER_Z)
        ax_inset.set_title('Zoomed In', fontsize=10)

    _save_figure(output_dir, 'rl_normalized_ratio_dual_camera_by_radius')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot SNR values from text files in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing the SNR text files.')
    args = parser.parse_args()

    left_cam_data, right_cam_data = read_snr_files(args.directory)
    crossing_z_dict = {radius: find_crossing_z(left_data, right_cam_data[radius]) for radius, left_data in left_cam_data.items()}
    save_crossing_z(crossing_z_dict, args.directory)
    plot_snr_data(left_cam_data, right_cam_data, args.directory, crossing_z_dict)
    plot_ratio_data(left_cam_data, right_cam_data, args.directory, crossing_z_dict)

if __name__ == '__main__':
    main()