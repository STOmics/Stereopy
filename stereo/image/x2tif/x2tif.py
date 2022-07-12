"""
Tools for converting gef/gem(txt) to tif image.
"""
import os
import sys
import gzip
from PIL import Image

import h5py
import numpy as np
import pandas as pd

from stereo.log_manager import logger

sys.path.append("..")

_IMAGE_MAGNIFY = 0.5

USING_CROP = False
CROP_SIZE = 10000


def gef2image(gef_file_path, dump_to_disk: bool = False, out_dir: str = "./"):
    """
    Convert gef file to tif image.

    :param gef_file_path:
    :param dump_to_disk: `True` for saving output to local disk
    :param out_dir: When `dump_to_disk` is `True`, the image will dump to this directory path
    :return: x_record, y_record, x_start, y_start
    """
    with h5py.File(gef_file_path, 'r') as f:
        data = f['/geneExp/bin1/expression'][:]
        # x_start = f['/geneExp/bin1/expression'].attrs['minX']
        # y_start = f['/geneExp/bin1/expression'].attrs['minY']

    df = pd.DataFrame(data, columns=['x', 'y', 'count'], dtype=int)
    logger.debug("min x: {} min y: {}".format(df['x'].min(), df['y'].min()))

    max_x = df['x'].max() + 1
    max_y = df['y'].max() + 1
    logger.debug("image dimension: {} x {} (width x height)".format(max_x, max_y))

    new_df = df.groupby(['x', 'y']).agg(UMI_sum=('count', 'sum')).reset_index()
    image = np.zeros(shape=(max_y, max_x), dtype=np.uint8)
    image[new_df['y'], new_df['x']] = new_df['UMI_sum']

    if dump_to_disk:
        _save_result(gef_file_path, out_dir, image)
    return image
    # return np.sum(image, 0).astype('float64'), np.sum(image, 1).astype('float64'), x_start, y_start


def txt2image(gem_file_path, dump_to_disk=False, out_dir: str = "./"):
    """
    Convert expression matrix data to image.

    :param gem_file_path:
    :param dump_to_disk: `True` for saving output to local disk
    :param out_dir: When `dump_to_disk` is `True`, the image will dump to this directory path
    :return: x_record, y_record, x_start, y_start
    """
    # Read from txt
    f, num_of_header_lines, header = _parse_head(gem_file_path)
    logger.debug("Number of header lines: {}".format(num_of_header_lines))
    logger.debug("Header info: \n{}".format(header))

    df = pd.read_csv(f, sep='\t', header=0)
    # Get image dimension and count each pixel's gene numbers
    logger.info("min x: {} min y: {}".format(df['x'].min(), df['y'].min()))

    df['x'] = df['x'] - df['x'].min()
    df['y'] = df['y'] - df['y'].min()
    max_x = df['x'].max() + 1
    max_y = df['y'].max() + 1
    logger.info("image dimension: {} x {} (width x height)".format(max_x, max_y))

    if gem_file_path.endswith('.txt'):
        # x_start = df['x'].min()
        # y_start = df['y'].min()
        pass
    else:
        _list = header.split("\n#")[-2:]
        # x_start = _list[0].split("=")[1]
        # y_start = _list[1].split("=")[1]

    try:
        new_df = df.groupby(['x', 'y']).agg(UMI_sum=('UMICount', 'sum')).reset_index()
    except Exception as e:
        logger.debug("try group-by with `x-y` and agg with `UMICount` failed, exception=%s" % str(e))
        new_df = df.groupby(['x', 'y']).agg(UMI_sum=('MIDCount', 'sum')).reset_index()

    # Set image pixel to gene counts
    # from uint16 to uint8
    image = np.zeros(shape=(max_y, max_x), dtype=np.uint8)
    image[new_df['y'], new_df['x']] = new_df['UMI_sum']
    # Save image (thumbnail image & crop image) to file
    if dump_to_disk:
        _save_result(gem_file_path, out_dir, image)
    return image


def _save_result(source_file_path, out_dir, image):
    filename = os.path.splitext(os.path.basename(source_file_path))[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = Image.fromarray(image)
    img.save(os.path.join(out_dir, filename + '.tif'), compression="tiff_lzw")

    if isinstance(_IMAGE_MAGNIFY, float) and _IMAGE_MAGNIFY > 0:
        img_re = img.resize(
            (int(img.size[0] * _IMAGE_MAGNIFY), int(img.size[1] * _IMAGE_MAGNIFY)),
            resample=Image.NEAREST
        )
        img_re.save(os.path.join(out_dir, filename + '_' + str(_IMAGE_MAGNIFY) + '.tif'), compression="tiff_lzw")

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
    logger.info("gem %s %d" % (gem, gem.endswith('.gz')))
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

    logger.debug("%s" % str(num_of_header_lines))
    logger.debug("%s" % str(header))
    logger.debug("%s" % str(eoh))

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
