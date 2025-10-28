#!/usr/bin/env python3
import argparse
import numpy as np
import tifffile as tiff
import math
import os

def process_large_tiff(input_path, down_factor):
    print(f"Opening: {input_path}")
    arr = tiff.memmap(input_path)  # memory-mapped read
    z_size, y_size, x_size = arr.shape
    print(f"Original shape: Z={z_size}, Y={y_size}, X={x_size}")

    # Dimensioni finali
    out_z = z_size // down_factor
    out_y = y_size // down_factor
    out_x = x_size // down_factor
    print(f"Downsampled shape: Z={out_z}, Y={out_y}, X={out_x}")

    # Array di accumulo piccoli
    nerve_sum = np.zeros((out_z, out_y, out_x), dtype=np.uint32)
    tissue_sum = np.zeros((out_z, out_y, out_x), dtype=np.uint32)

    # Elaborazione a blocchi lungo Z
    block_slices = down_factor * 10  # es. processa 10 macrovoxel alla volta
    for z_start in range(0, z_size, block_slices):
        z_end = min(z_start + block_slices, z_size)
        print(f"Processing Z: {z_start} -> {z_end}")

        block = arr[z_start:z_end, :, :]  # lettura da memmap

        # Mask nervi e tessuto
        mask_nerve = (block == 2)
        mask_tissue = (block == 1) | (block == 2)

        # Riduzione 3D del blocco corrente
        # Calcola gli indici di output
        out_z_start = z_start // down_factor
        out_z_end = math.ceil(z_end / down_factor)

        # Riduci blocco nervi
        reduced_nerve = block_reduce_and_sum_3d(mask_nerve, down_factor)  # numero di voxel “nervo” in ogni macrovoxel del blocco ridotto.
        reduced_tissue = block_reduce_and_sum_3d(mask_tissue, down_factor)  # numero di voxel “tessuto+nervo” in ogni macrovoxel del blocco ridotto.

        # Somma ai risultati finali
        nerve_sum[out_z_start:out_z_end, :, :] += reduced_nerve
        tissue_sum[out_z_start:out_z_end, :, :] += reduced_tissue

    # Percentuali finali (0-100)
    with np.errstate(divide='ignore', invalid='ignore'):
        percent = (nerve_sum / tissue_sum) * 100.0
        percent[np.isnan(percent)] = 0  # dove non c'è tessuto

    # Clipping e conversione a 8 bit
    percent_8bit = np.clip(percent, 0, 100).astype(np.uint8)

    # Salvataggio
    out_path = os.path.splitext(input_path)[0] + f"_nerve_percent_{down_factor}x.tif"
    tiff.imwrite(out_path, percent_8bit)
    print(f"Saved: {out_path}")


def block_reduce_and_sum_3d(mask, factor):
    """Riduce un array 3D booleano per fattore isotropo calcolando la somma"""
    z, y, x = mask.shape
    z_new = z // factor
    y_new = y // factor
    x_new = x // factor

    # Taglia per multipli esatti
    mask = mask[:z_new*factor, :y_new*factor, :x_new*factor]

    # Reshape e somma (trucco per fare la somma in modo efficiente)
    mask = mask.reshape(z_new, factor, y_new, factor, x_new, factor)
    summed = mask.sum(axis=(1, 3, 5)).astype(np.uint32)
    return summed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample 3D segmentation to nerve percentage map")
    parser.add_argument("--input", required=True, help="Path to segmented TIFF (uint8: 0=bg, 1=tissue, 2=nerve)")
    parser.add_argument("--downsampling", type=int, required=True, help="Isotropic downsampling factor (e.g., 100)")
    args = parser.parse_args()

    process_large_tiff(args.input, args.downsampling)
