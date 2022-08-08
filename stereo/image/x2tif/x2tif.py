"""
Tools for converting gef/gem(txt) to tif image.
"""
import os
import sys
import gzip
from PIL import Image

import numpy as np
import pandas as pd

from stereo.log_manager import logger

sys.path.append("..")

_IMAGE_MAGNIFY = 0.5

USING_CROP = False
CROP_SIZE = 10000


def gef2image(gef_file_path, dump_to_disk: bool = False, out_dir: str = "./", bin_size=20):
    """
    Convert gef file to tif image.

    :param gef_file_path:
    :param dump_to_disk: `True` for dumping tif to local disk
    :param out_dir: When `dump_to_disk` is `True`, the image will dump to this directory path
    :param bin_size: default is `bin100`, set 1 using `bin1` for high quality, which will cost more memory and CPU
    :return: image_nd_array, a nd_array for describe (x, y) with `UMI_sum`
    """
    df = pd.read_hdf(gef_file_path, f'/geneExp/bin{bin_size}/expression', columns=['x', 'y', 'count'])
    new_df = df.groupby(['x', 'y'], as_index=False, sort=False).agg(UMI_sum=('count', 'sum'))
    image_nd_array = np.zeros(shape=(df['y'].max() + 1, df['x'].max() + 1), dtype=np.uint8)
    image_nd_array[new_df['y'], new_df['x']] = new_df['UMI_sum']
    if dump_to_disk:
        _save_result(gef_file_path, out_dir, image_nd_array)
    return image_nd_array


def txt2image(gem_file_path, dump_to_disk=False, out_dir: str = "./"):
    """
    Convert expression matrix data to image.

    :param gem_file_path:
    :param dump_to_disk: `True` for dumping tif to local disk
    :param out_dir: When `dump_to_disk` is `True`, the image will dump to this directory path
    :return: image_nd_array, a nd_array for describe (x, y) with `UMI_sum`
    """
    f, num_of_header_lines, header = _parse_head(gem_file_path)
    df = pd.read_csv(f, sep='\t', header=0, usecols=['x', 'y', 'MIDCount'], engine='pyarrow', dtype=np.uint32)
    df['x'] = df['x'] - df['x'].min()
    df['y'] = df['y'] - df['y'].min()
    df.rename(columns={'UMICount': 'MIDCount'}, inplace=True)
    new_df = df.groupby(['x', 'y'], as_index=False, sort=False).agg(UMI_sum=('MIDCount', 'sum'))
    image_nd_array = np.zeros(shape=(df['y'].max() + 1, df['x'].max() + 1), dtype=np.uint8)
    image_nd_array[new_df['y'], new_df['x']] = new_df['UMI_sum']
    if dump_to_disk:
        _save_result(gem_file_path, out_dir, image_nd_array)
    return image_nd_array


def _save_result(source_file_path, out_dir, image):
    filename = os.path.splitext(os.path.basename(source_file_path))[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = Image.fromarray(image)
    img.save(os.path.join(out_dir, filename + '.tif'), compression="tiff_lzw")

    if USING_CROP:
        crop_dir = os.path.join(out_dir, 'crop')
        if not os.path.exists(crop_dir):
            os.mkdir(crop_dir)
        name = os.path.join(crop_dir, filename)
        _save_patches(img, CROP_SIZE, CROP_SIZE, name)


def _parse_head(gem):
    """
    Parse additional header info
    """
    if gem.endswith('.gz'):
        f = gzip.open(gem, 'rb')
    else:
        f = open(gem, 'rb')

    header = ''
    num_of_header_lines = 0
    eoh = 0
    for i, l in enumerate(f):
        decoded_l = l.decode("utf-8")  # read in as binary, decode first
        if decoded_l.startswith('#'):  # header lines always start with '#'
            header += decoded_l
            num_of_header_lines += 1
            eoh = f.tell()  # get end-of-header position
        else:
            break

    logger.debug("Number of header lines: %s" % str(num_of_header_lines))
    logger.debug("Header info: %s" % str(header))

    # find start of expression matrix
    f.seek(eoh)
    return f, num_of_header_lines, header


def _save_patches(img, patch_w, patch_h, name):
    # calculate num of patches
    width, height = img.size
    num_w = int(np.ceil(width / patch_w))
    num_h = int(np.ceil(height / patch_h))
    logger.debug("{} col, {} row, total {} patches".format(num_w, num_h, num_w * num_h))
    for i in range(num_w):
        for j in range(num_h):
            # NOTE: last patches will have different patch size
            x = i * patch_w
            y = j * patch_h

            if i == num_w - 1:
                p_w = width - x
            else:
                p_w = patch_w
            if j == num_h - 1:
                p_h = height - y
            else:
                p_h = patch_h

            box = (x, y, x + p_w, y + p_h)
            im_crop = img.crop(box)

            im_name = name + '_{}_{}_{}_{}'.format(height, width, y, x)
            im_crop.save(im_name + '.tif', compression='tiff_lzw')
