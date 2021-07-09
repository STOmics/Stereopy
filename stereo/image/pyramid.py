#!/usr/bin/env python3
# coding: utf-8
"""
@file: pyramid.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/11  create file.
"""
import time
import h5py
import numpy as np
import tifffile as tifi
from PIL import Image


def _write_attrs(gp, d):
    """ Write dict to hdf5.Group as attributes. """
    for k, v in d.items():
        gp.attrs[k] = v


def split_image(im, img_size, h5_path, bin_size):
    """ Split image into patches with img_size and save to h5 file. """
    t0 = time.time()
    # get number of patches
    height, width = im.shape
    num_x = int(width / img_size) + 1
    num_y = int(height / img_size) + 1

    with h5py.File(h5_path, 'a') as out:
        group = out.require_group(f'bin_{bin_size}')

        # write attributes
        attrs = {'sizex': width,
                 'sizey': height,
                 'XimageNumber': num_x,
                 'YimageNumber': num_y}
        _write_attrs(group, attrs)

        # write dataset
        for x in range(0, num_x):
            for y in range(0, num_y):
                # deal with last row/column images
                x_end = min(((x + 1) * img_size), width)
                y_end = min(((y + 1) * img_size), height)
                small_im = im[y * img_size:y_end, x * img_size:x_end]

                data_name = f'{x}/{y}'
                try:
                    # normal dataset creation
                    group.create_dataset(data_name, data=small_im)
                except Exception as e:
                    # if dataset already exists, replace it with new data
                    print(e)
                    del group[data_name]
                    group.create_dataset(data_name, data=small_im)
    t1 = time.time()
    print(f"bin_{bin_size} split: {t1 - t0:.2f} seconds")


def merge_pyramid(h5_path, bin_size, out_path):
    """ Merge image patches back to large image. """
    t0 = time.time()
    h5 = h5py.File(h5_path, 'r')
    # get attributes
    img_size = h5['metaInfo'].attrs['imgSize']
    group = h5[f'bin_{bin_size}']
    width = group.attrs['sizex']
    height = group.attrs['sizey']
    # initialize image
    im = np.zeros((height, width), dtype=group['0/0'][()].dtype)

    # recontruct image
    for i in range(group.attrs['XimageNumber']):
        for j in range(group.attrs['YimageNumber']):
            small_im = group[f'{i}/{j}'][()]
            x_end = min(((i + 1) * img_size), width)
            y_end = min(((j + 1) * img_size), height)
            im[j * img_size:y_end, i * img_size:x_end] = small_im
    h5.close()
    t1 = time.time()
    print(f"Merge image: {t1 - t0:.2f} seconds.")
    tifi.imsave(out_path + '.tiff', im)
    image = Image.open(out_path + '.tiff')
    image.mode = 'I'
    image.point(lambda i: i * (1. / 256)).convert('L').save(out_path + '.jpeg')
    return im


def create_pyramid(img_path, h5_path, img_size, x_start, y_start, mag):
    """ Create image pyramid and save to h5. """
    t0 = time.time()
    img = tifi.imread(img_path)
    t1 = time.time()
    print(f"Load image: {t1 - t0:.2f} seconds.")

    # get height and width
    height, width = img.shape
    # im = np.rot90(im, 1)  ## 旋转图片，配准后的图片应该不用旋转了

    # write image metadata
    with h5py.File(h5_path, 'a') as h5_out:
        meta_group = h5_out.require_group('metaInfo')
        info = {'imgSize': img_size,
                'x_start': x_start,
                'y_start': y_start,
                'sizex': width,
                'sizey': height}
        _write_attrs(meta_group, info)

    # write image pyramid of bin size
    for bin_size in mag:
        im_downsample = img[::bin_size, ::bin_size]
        split_image(im_downsample, img_size, h5_path, bin_size)

    t2 = time.time()
    print(f"Save h5: {t2 - t1:.2f} seconds.")
