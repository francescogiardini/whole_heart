import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

def plot_snr_data(left_cam_data, right_cam_data, output_dir, crossing_z_dict):
    plt.figure(figsize=(20, 10))
    symbols = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    for idx, (radius, left_data) in enumerate(left_cam_data.items()):
        right_data = right_cam_data[radius]
        z_values = list(map(int, left_data.keys()))
        left_snr_values = np.log10(list(left_data.values()))
        right_snr_values = np.log10(list(right_data.values()))

        crossing_z = crossing_z_dict[radius]
        if crossing_z:
            print(f'Radius {radius}: L exceeds R at z = {crossing_z}')

        symbol = symbols[idx % len(symbols)]
        plt.plot(z_values, left_snr_values, f'{symbol}-', color='b', alpha=radius, label=f'LeftCAM radius {radius}')
        plt.plot(z_values, right_snr_values, f'{symbol}-', color='g', alpha=radius, label=f'RightCAM radius {radius}')

        if crossing_z:
            plt.text(crossing_z + 1, np.log10(left_data[str(crossing_z)]) + 0.03, str(crossing_z), color='black', fontsize=12, ha='right')

    plt.xlabel('z')
    plt.ylabel('log10(SNR)')
    plt.title('DualMesoSPIM: Image Quality along z by High Frequency analysis')
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values) + 1, 2))
    plt.savefig(os.path.join(output_dir, 'snr_ratio_dual_camera_by_radius.png'))
    plt.show()

def plot_ratio_data(left_cam_data, right_cam_data, output_dir, crossing_z_dict):
    plt.figure(figsize=(20, 10))
    symbols = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    for idx, (radius, left_data) in enumerate(left_cam_data.items()):
        right_data = right_cam_data[radius]
        z_values = list(map(int, left_data.keys()))
        ratio_values = [right_data[str(z)] / left_data[str(z)] for z in z_values]

        symbol = symbols[idx % len(symbols)]
        plt.plot(z_values, ratio_values, f'{symbol}-', color='k', alpha=radius, label=f'R/L ratio radius {radius}')

    plt.axhline(y=1, color='k', linestyle='--')
    plt.xlabel('z')
    plt.ylabel('R/L')
    plt.title('DualMesoSPIM: R/L Ratio along z by High Frequency analysis')
    plt.legend(loc='lower left')
    plt.xticks(np.arange(min(z_values), max(z_values) + 1, 2))

    # Create inset plot
    ax_inset = plt.gca().inset_axes([0.50, 0.40, 0.3, 0.5])
    min_crossing_z = min(crossing_z_dict.values())
    max_crossing_z = max(crossing_z_dict.values())
    inset_xticks = np.arange(min_crossing_z - 5, max_crossing_z + 6, 1)

    for idx, (radius, left_data) in enumerate(left_cam_data.items()):
        right_data = right_cam_data[radius]
        z_values = list(map(int, left_data.keys()))
        ratio_values = [right_data[str(z)] / left_data[str(z)] for z in z_values]

        crossing_z = crossing_z_dict[radius]
        if crossing_z:
            inset_z_values = z_values[max(0, z_values.index(crossing_z) - 5):min(len(z_values), z_values.index(crossing_z) + 6)]
            inset_ratio_values = ratio_values[max(0, z_values.index(crossing_z) - 5):min(len(z_values), z_values.index(crossing_z) + 6)]

            symbol = symbols[idx % len(symbols)]
            ax_inset.plot(inset_z_values, inset_ratio_values, f'{symbol}-', color='k', alpha=radius)

    ax_inset.axhline(y=1, color='k', linestyle='--')
    ax_inset.set_xticks(inset_xticks)
    ax_inset.set_title('Zoomed In', fontsize=10)

    plt.savefig(os.path.join(output_dir, 'rl_ratio_dual_camera_by_radius.png'))
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