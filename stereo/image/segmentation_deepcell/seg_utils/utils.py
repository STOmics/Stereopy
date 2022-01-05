import numpy as np
import cv2
import slideio
import os
import tifffile
from skimage import transform
from scipy import signal
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, find_boundaries, relabel_sequential


def resize(data, shape, data_format='channels_last', labeled_image=False):
    """Resize the data to the given shape.
    Uses openCV to resize the data if the data is a single channel, as it
    is very fast. However, openCV does not support multi-channel resizing,
    so if the data has multiple channels, use skimage.

    Args:
        data (np.array): data to be reshaped. Must have a channel dimension
        shape (tuple): shape of the output data in the form (x,y).
            Batch and channel dimensions are handled automatically and preserved.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
        labeled_image (bool): flag to determine how interpolation and floats are handled based
         on whether the data represents raw images or annotations

    Raises:
        ValueError: ndim of data not 3 or 4
        ValueError: Shape for resize can only have length of 2, e.g. (x,y)

    Returns:
        numpy.array: data reshaped to new shape.
    """
    if len(data.shape) not in {3, 4}:
        raise ValueError('Data must have 3 or 4 dimensions, e.g. '
                         '[batch, x, y], [x, y, channel] or '
                         '[batch, x, y, channel]. Input data only has {} '
                         'dimensions.'.format(len(data.shape)))

    if len(shape) != 2:
        raise ValueError('Shape for resize can only have length of 2, e.g. (x,y).'
                         'Input shape has {} dimensions.'.format(len(shape)))

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == 'channels_first' else -1
    batch_axis = -1 if data_format == 'channels_first' else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == 'channels_first':
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        _resize = lambda d: transform.resize(d, shape, mode='constant', preserve_range=True,
                                             order=order, anti_aliasing=anti_aliasing)
    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype('float32')

        _resize = lambda d: np.expand_dims(cv2.resize(np.squeeze(d), shape,
                                                      interpolation=interpolation),
                                           axis=channel_axis)

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)


def normalize(image, epsilon=1e-07):
    """Normalize image data by dividing by the maximum pixel value

    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.

    Returns:
        numpy.array: normalized image data
    """

    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image


def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
    """

    if mask.ndim not in {2, 3}:
        raise ValueError('erode_edges expects arrays of ndim 2 or 3.'
                         'Got ndim: {}'.format(mask.ndim))
    if erosion_width:
        new_mask = np.copy(mask)
        for _ in range(erosion_width):
            boundaries = find_boundaries(new_mask, mode='inner')
            new_mask[boundaries > 0] = 0
        return new_mask

    return mask


def cell_watershed(outputs,
                   min_distance=10,
                   detection_threshold=0.1,
                   distance_threshold=0.01,
                   exclude_border=False,
                   small_objects_threshold=0):
    """Postprocessing function for deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the outer distance prediction.

    Args:
        outputs (list): DeepWatershed model output. A list of
            [inner_distance, outer_distance, fgbg].

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Prediction for the foregound/background transform.

        min_distance (int): Minimum allowable distance between two cells.
        detection_threshold (float): Threshold for the inner distance.
        distance_threshold (float): Threshold for the outer distance.
        exclude_border (bool): Whether to include centroid detections
            at the border.
        small_objects_threshold (int): Removes objects smaller than this size.

    Returns:
        numpy.array: Uniquely labeled mask.
    """
    inner_distance_batch = outputs[0][:, ..., 0]
    outer_distance_batch = outputs[1][:, ..., 0]

    label_images = []
    for batch in range(inner_distance_batch.shape[0]):
        inner_distance = inner_distance_batch[batch]
        outer_distance = outer_distance_batch[batch]

        coords = peak_local_max(inner_distance,
                                min_distance=min_distance,
                                threshold_abs=detection_threshold,
                                exclude_border=exclude_border)

        markers = np.zeros(inner_distance.shape)
        markers[coords[:, 0], coords[:, 1]] = 1
        markers = label(markers)
        label_image = watershed(-outer_distance,
                                markers,
                                mask=outer_distance > distance_threshold)
        label_image = erode_edges(label_image, 1)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)

    return label_images


def tile_image(image, model_input_shape=(512, 512),
               stride_ratio=0.75, pad_mode='constant'):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The image to tile, must be rank 4.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): A tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, pad_mode)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_x = []
    overlaps_y = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                y_axis = 2

                # Compute the start and end for each tile
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                # Compute the overlaps for each tile
                if i == 0:
                    overlap_x = (0, tile_size_x - stride_x)
                elif i == rep_number_x - 2:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - image.shape[x_axis] + x_end)
                elif i == rep_number_x - 1:
                    overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                else:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                if j == 0:
                    overlap_y = (0, tile_size_y - stride_y)
                elif j == rep_number_y - 2:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - image.shape[y_axis] + y_end)
                elif j == rep_number_y - 1:
                    overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                else:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                overlaps_x.append(overlap_x)
                overlaps_y.append(overlap_y)
                counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y

    return tiles, tiles_info


def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (signal.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (signal.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window


def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(np.expand_dims(window_x, -1), -1)
    window_y = np.expand_dims(np.expand_dims(window_y, -1), -1)

    window = window_x * window_y.transpose(1, 0, 2)
    return window



def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']
    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    stride_ratio = tiles_info['stride_ratio']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
    window_size = (tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=np.float)

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles, batches, x_starts, x_ends, y_starts, y_ends, overlaps_x, overlaps_y):

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[1] and
                min_tile_size <= tile_size_y < image_shape[2] and
                stride_ratio >= min_stride_ratio):
            window = window_2D(window_size, overlap_x=overlap_x,
                               overlap_y=overlap_y, power=power)
            image[batch, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = x_pad[0]
    y_start = y_pad[0]
    x_end = image_shape[1] - x_pad[1]
    y_end = image_shape[2] - y_pad[1]

    image = image[:, x_start:x_end, y_start:y_end, :]

    return image


def view_bar(message, id, total, end=''):
    rate = id / total
    rate_num = int(rate * 40)
    print('\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num,
                                    "=" * (40 - rate_num), np.round(rate * 100), id, total,), end=end)


def split(image, cut_size, overlap=100):
    shapes = image.shape
    x_nums = int(shapes[0] / (cut_size - overlap))
    y_nums = int(shapes[1] / (cut_size - overlap))
    img_list = []
    x_list = []
    y_list = []
    for x_temp in range(x_nums + 1):
        for y_temp in range(y_nums + 1):
            x_begin = max(0, x_temp * (cut_size - overlap))
            y_begin = max(0, y_temp * (cut_size - overlap))
            x_end = min(x_begin + cut_size, shapes[0])
            y_end = min(y_begin + cut_size, shapes[1])
            i = image[x_begin: x_end, y_begin: y_end]
            # tifffile.imsave(os.path.join(outpath, file + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' + str(x_begin) + '_' + str(y_begin) + '.tif'), i)  #, r'white_5000'r'20210326_other_crop'
            x_list.append(x_begin)
            y_list.append(y_begin)
            img_list.append(i)
    return img_list, x_list, y_list


def merge(label_list, x_list, y_list, shapes,  overlap = 100):

    if len(label_list) == 1:
        return label_list[0]

    if not isinstance(label_list, list):
        return label_list

    image = np.zeros((int(shapes[0]), int(shapes[1])), dtype=np.uint8)
    for index, temp_img in enumerate(label_list):
        info = [x_list[index], y_list[index]]
        h, w = temp_img.shape
        x_begin = int(info[0]) + overlap // 2
        y_begin = int(info[1]) + overlap // 2
        if overlap == 0:
            image[int(x_begin): int(x_begin) + h - overlap, int(y_begin): int(y_begin) + w - overlap] = temp_img
        else:
            image[int(x_begin): int(x_begin) + h - overlap, int(y_begin): int(y_begin) + w - overlap] = temp_img[
                                                                                                        overlap // 2: - overlap // 2,
                                                                                                        overlap // 2: - overlap // 2]
    return image


def czi_save_tif(path, outpath):
    _, file = os.path.split(path)
    slide = slideio.open_slide(path, "CZI")

    scene = slide.get_scene(0)
    print(dir(scene))
    print(scene.size, scene.resolution, scene.origin)
    im = scene.read_block((0, 0, 0, 0), size=(scene.size[0], scene.size[1]))
    shapes = im.shape
    print(shapes)
    if len(shapes) == 2:
        tifffile.imsave(os.path.join(outpath, os.path.splitext(file)[0]) + '_ssdna.tif', im)
    else:
        tifffile.imsave(os.path.join(outpath, os.path.splitext(file)[0]) + '_ssdna.tif', im[:, :, 0])
        tifffile.imsave(os.path.join(outpath, os.path.splitext(file)[0]) + '_cona.tif', im[:, :, 1])


def czi2tif(path):
    if os.path.isdir(path):
        file_list = os.listdir(path)
        path_file, _ = os.path.split(path)
        outpath = os.path.join(path_file, _ + '_tif')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        for file in file_list:
            print('*' * 50)
            print(file)
            czi = os.path.join(path, file)
            czi_save_tif(czi, outpath)
    else:
        path_file, _ = os.path.split(path)
        outpath = os.path.join(path_file, _ + '_tif')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        print('*' * 50)
        czi_save_tif(path, outpath)
    print('*' * 50)
    print('save done!')


def outline(image):
    image = np.where(image != 0, 1, 0).astype(np.uint8)
    edge = np.zeros((image.shape), dtype=np.uint8)
    contours, hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = cv2.drawContours(edge, contours, -1, (255, 255, 255), 1)
    return r


def hole_fill(binary_image):
    ''' 孔洞填充 '''
    hole = binary_image.copy()  ## 空洞填充
    hole = cv2.copyMakeBorder(hole, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0])  # 首先将图像边缘进行扩充，防止空洞填充不完全
    hole2 = hole.copy()
    cv2.floodFill(hole, None, (0, 0), 255)  # 找到洞孔
    hole = cv2.bitwise_not(hole)
    binary_hole = cv2.bitwise_or(hole2, hole)[1:-1, 1:-1]
    return binary_hole


def transfer_16bit_to_8bit(image_16bit):

    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)

    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

    return image_8bit