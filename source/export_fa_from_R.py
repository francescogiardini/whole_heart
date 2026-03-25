import numpy as np
import os
import argparse

from tifffile import imwrite

from custom_tool_kit import search_value_in_txt


def export_fa_from_R(R_filepath, parameter_filepath, fa_threshold):
    """
    Export FA field from R matrix as 3D TIFF and mean projections along 3 axes.
    - 3D TIFF: each voxel = FA value if cell_info is True, 0 otherwise.
    - Mean projections: average along each axis, considering only non-zero voxels.
    """

    # load R
    R_filename = os.path.basename(R_filepath)
    R = np.load(R_filepath)
    shape_R = R.shape
    base_path = os.path.dirname(R_filepath)

    print('R loaded from: ', R_filepath)
    print('shape_R (r, c, z): ', shape_R)

    # build 3D FA matrix: 0 where cell_info is False, fa value where cell_info is True
    fa_3d = np.zeros(shape_R, dtype=np.float32)
    cell_mask = R['cell_info'] == True
    fa_3d[cell_mask] = R['fa'][cell_mask].astype(np.float32)

    print('Total cells with cell_info=True: {}'.format(np.count_nonzero(cell_mask)))
    print('FA range in valid cells: [{:.4f}, {:.4f}]'.format(
        fa_3d[cell_mask].min() if np.any(cell_mask) else 0,
        fa_3d[cell_mask].max() if np.any(cell_mask) else 0))

    # create output folder
    out_folder = os.path.join(base_path, 'FA_export_{}'.format(R_filename.split('.')[0]))
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # ---- Save 3D TIFF ----
    # tifffile expects (z, y, x) for 3D TIFF
    # our fa_3d is (r, c, z) -> moveaxis to (z, r, c)
    fa_3d_tiff = np.moveaxis(fa_3d, 2, 0)  # (r, c, z) -> (z, r, c)
    tiff_3d_path = os.path.join(out_folder, 'FA_3D.tiff')
    imwrite(tiff_3d_path, fa_3d_tiff)
    print('\n3D FA TIFF saved: {}'.format(tiff_3d_path))
    print('  shape (z, r, c): {}'.format(fa_3d_tiff.shape))

    # ---- Mean projections (only over non-zero voxels) ----

    # XY projection: mean along z axis (axis=2)
    # fa_3d shape: (r, c, z)
    nonzero_count_xy = np.count_nonzero(fa_3d, axis=2).astype(np.float32)  # (r, c)
    fa_sum_xy = np.sum(fa_3d, axis=2)  # (r, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_xy = np.where(nonzero_count_xy > 0, fa_sum_xy / nonzero_count_xy, 0.0).astype(np.float32)

    xy_path = os.path.join(out_folder, 'FA_mean_XY.tiff')
    imwrite(xy_path, mean_xy)
    print('FA mean XY saved: {}  shape: {}'.format(xy_path, mean_xy.shape))

    # XZ projection: mean along c axis (axis=1)
    # fa_3d shape: (r, c, z) -> collapse c -> result (r, z)
    nonzero_count_xz = np.count_nonzero(fa_3d, axis=1).astype(np.float32)  # (r, z)
    fa_sum_xz = np.sum(fa_3d, axis=1)  # (r, z)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_xz = np.where(nonzero_count_xz > 0, fa_sum_xz / nonzero_count_xz, 0.0).astype(np.float32)

    xz_path = os.path.join(out_folder, 'FA_mean_XZ.tiff')
    imwrite(xz_path, mean_xz)
    print('FA mean XZ saved: {}  shape: {}'.format(xz_path, mean_xz.shape))

    # YZ projection: mean along r axis (axis=0)
    # fa_3d shape: (r, c, z) -> collapse r -> result (c, z)
    nonzero_count_yz = np.count_nonzero(fa_3d, axis=0).astype(np.float32)  # (c, z)
    fa_sum_yz = np.sum(fa_3d, axis=0)  # (c, z)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_yz = np.where(nonzero_count_yz > 0, fa_sum_yz / nonzero_count_yz, 0.0).astype(np.float32)

    yz_path = os.path.join(out_folder, 'FA_mean_YZ.tiff')
    imwrite(yz_path, mean_yz)
    print('FA mean YZ saved: {}  shape: {}'.format(yz_path, mean_yz.shape))

    print('\n ** Process finished - OK')
    print('All files saved in: {}'.format(out_folder))
    return out_folder


def main(args):
    export_fa_from_R(
        R_filepath=args.R_path,
        parameter_filepath=args.parameter_filename,
        fa_threshold=args.fa
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export FA from R matrix as 3D TIFF and mean projections.')
    parser.add_argument('-r', '--R_path', type=str, required=True,
                        help='Complete path of the R .npy matrix.')
    parser.add_argument('-p', '--parameter_filename', type=str, required=True,
                        help='Parameter filename.')
    parser.add_argument('-f', '--fa', type=float, required=True,
                        help='FA threshold (stored for reference, not used for filtering in this script).')
    args = parser.parse_args()
    main(args)
