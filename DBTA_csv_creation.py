#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:44:36 2024

@author: tgodelaine
"""

import large_image
import pandas as pd
import os
import tifffile
import zarr
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.insert(0, '/auto/home/users/m/d/mdausort/Software/large_image_source_tifffile')


def create_patches(image_path, patch_w, patch_h, stride_percent, magnification):

    image_slide = large_image.getTileSource(image_path)
    xc, yc = [], []

    for slide_info in image_slide.tileIterator(

            scale=dict(magnification=magnification),
            tile_size=dict(width=patch_w, height=patch_w),
            tile_overlap=dict(x=0, y=0),
            format=large_image.tilesource.TILE_FORMAT_NUMPY):

        im_tile = np.array(slide_info['tile'])
        tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))

        if np.mean(tile_mean_rgb) < 220. and im_tile.shape == (patch_w, patch_h, 3):
            xc.append(slide_info["x"])
            yc.append(slide_info["y"])

    return xc, yc


def create_annotation_csv_file(csv_dir, images_dir, patch_w, patch_h, stride_percent, magnification, set_percent=[0.7, 0.1, 0.2]):
    csv_path = os.path.join(csv_dir, 'annotation_patches_' + str(patch_w) + '_' + str(patch_h)
                            + '_' + str(stride_percent)
                            + '_' + str(magnification) + '.csv')

    header = ['uuid', 'xc', 'yc', 'w', 'h', 'diagnosis']

    annotation_path = os.path.join(csv_dir, 'annotation.csv')
    df = pd.read_csv(annotation_path)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        images = [f for f in os.listdir(images_dir) if f.endswith('.ndpi')]

        for image in images:
            image_path = os.path.join(images_dir, image)
            idx = df.index[df['uuid'] == image.split('.ndpi')[0]].tolist()

            xc, yc = create_patches(image_path, patch_w, patch_h, stride_percent, magnification)

            diagnosis = df.loc[idx, 'diagnosis']

            for x, y in zip(xc, yc):  # xc, yc coordinate of the upper left corner
                new_line = [image, x, y, patch_w, patch_h, diagnosis.values[0]]
                writer.writerow(new_line)

    df_patches = pd.read_csv(csv_path)

    sorted_df = df_patches.sort_values(by='diagnosis')

    classes = np.unique(sorted_df['diagnosis'])

    diagnosis_df = sorted_df['diagnosis']

    train_df = pd.DataFrame({})
    val_df = pd.DataFrame({})
    test_df = pd.DataFrame({})

    set_percent = [0.7, 0.1, 0.2]

    for i, c in enumerate(classes):
        df_c = sorted_df[diagnosis_df == c]
        anchor = len(df_c)

        train_df = pd.concat([train_df, df_c.iloc[:int(anchor * set_percent[0]), :]])
        val_df = pd.concat([val_df, df_c.iloc[int(anchor * set_percent[0]):int(anchor * set_percent[0]) + int(anchor * set_percent[1]), :]])
        test_df = pd.concat([test_df, df_c.iloc[int(anchor * set_percent[0]) + int(anchor * set_percent[1]):, :]])

    # Save the selected rows to a new CSV file
    train_df.to_csv(csv_path.split('.csv')[0] + '_train.csv', index=False)
    val_df.to_csv(csv_path.split('.csv')[0] + '_val.csv', index=False)
    test_df.to_csv(csv_path.split('.csv')[0] + '_test.csv', index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for the specified arguments")

    parser.add_argument("--csv_dir", type=str, help="Directory of the csv file")
    parser.add_argument("--images_dir", type=str, help="Directory of the images")
    parser.add_argument("-pw", "--patch_w", type=int, default=416, help="Width of the patch")
    parser.add_argument("-ph", "--patch_h", type=int, default=416, help="Height of the patch")
    parser.add_argument("-sp", "--stride_percent", type=float, default=1.0, help="Stride percentage")
    parser.add_argument("-m", "--magnification", type=float, default=20, help="Magnification level")
    parser.add_argument("--set_percent", type=float, nargs='+', default=[0.7, 0.1, 0.2], help="Distribution percentage")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    create_annotation_csv_file(args.csv_dir, args.images_dir,
                               args.patch_w, args.patch_h, args.stride_percent, args.magnification,
                               args.set_percent)
