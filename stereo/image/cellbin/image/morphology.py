"""Functions for pre- and post-processing image data"""
from _warnings import warn

import numpy as np
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects, h_maxima
from skimage.morphology import disk, ball, square, cube, dilation
from skimage.segmentation import relabel_sequential, watershed
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops


def f_deep_watershed(outputs,
                     radius=10,
                     maxima_threshold=0.1,
                     interior_threshold=0.01,
                     maxima_smooth=0,
                     interior_smooth=1,
                     maxima_index=0,
                     interior_index=-1,
                     label_erosion=0,
                     small_objects_threshold=0,
                     fill_holes_threshold=0,
                     pixel_expansion=None,
                     watershed_line=1,
                     maxima_algorithm='h_maxima',
                     **kwargs):
    """
    Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    :param outputs:(list): List of [maximas, interiors] model outputs.
            Use `maxima_index` and `interior_index` if list is longer than 2,
            or if the outputs are in a different order.
    :param radius: (int): Radius of disk used to search for maxima
    :param maxima_threshold:(float): Threshold for the maxima prediction.
    :param interior_threshold:(float): Threshold for the interior prediction.
    :param maxima_smooth:(int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
    :param interior_smooth:(int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
    :param maxima_index:(int): The index of the maxima prediction in ``outputs``.
    :param interior_index:(int): The index of the interior prediction in ``outputs``.
    :param label_erosion:(int): Number of pixels to erode segmentation labels.
    :param small_objects_threshold:(int): Removes objects smaller than this size.
    :param fill_holes_threshold:(int): Maximum size for holes within segmented
            objects to be filled.
    :param pixel_expansion:(int): Number of pixels to expand ``interiors``.
    :param watershed_line:(int): If need watershed line.
    :param maxima_algorithm:(str): Algorithm used to locate peaks in ``maximas``.
            One of ``h_maxima`` (default) or ``peak_local_max``.
            ``peak_local_max`` is much faster but seems to underperform when
            given regious of ambiguous maxima.
    :param kwargs:
    :return:numpy.array: Integer label mask for instance segmentation.

    Raises:
    ValueError: ``outputs`` is not properly formatted.
    """

    try:
        maximas = outputs[maxima_index]
        interiors = outputs[interior_index]
    except (TypeError, KeyError, IndexError):
        raise ValueError('`outputs` should be a list of at least two '
                         'NumPy arryas of equal shape.')

    valid_algos = {'h_maxima', 'peak_local_max'}
    if maxima_algorithm not in valid_algos:
        raise ValueError('Invalid value for maxima_algorithm: {}. '
                         'Must be one of {}'.format(maxima_algorithm, valid_algos))

    total_pixels = maximas.shape[1] * maximas.shape[2]
    if maxima_algorithm == 'h_maxima' and total_pixels > 5000 ** 2:
        print('h_maxima peak finding algorithm was selected, '
                 'but the provided image is larger than 5k x 5k pixels.'
                 'This will lead to slow prediction performance.')
    # Handle deprecated arguments
    min_distance = kwargs.pop('min_distance', None)
    if min_distance is not None:
        radius = min_distance
        warn('`min_distance` is now deprecated in favor of `radius`. '
                     'The value passed for `radius` will be used.')

    # distance_threshold vs interior_threshold
    distance_threshold = kwargs.pop('distance_threshold', None)
    if distance_threshold is not None:
        interior_threshold = distance_threshold
        warn('`distance_threshold` is now deprecated in favor of '
                     '`interior_threshold`. The value passed for '
                     '`distance_threshold` will be used.',
                     DeprecationWarning)

    # detection_threshold vs maxima_threshold
    detection_threshold = kwargs.pop('detection_threshold', None)
    if detection_threshold is not None:
        maxima_threshold = detection_threshold
        warn('`detection_threshold` is now deprecated in favor of '
                     '`maxima_threshold`. The value passed for '
                     '`detection_threshold` will be used.',
                     DeprecationWarning)

    if maximas.shape[:-1] != interiors.shape[:-1]:
        raise ValueError('All input arrays must have the same shape. '
                         'Got {} and {}'.format(
                            maximas.shape, interiors.shape))

    if maximas.ndim not in {4, 5}:
        raise ValueError('maxima and interior tensors must be rank 4 or 5. '
                         'Rank 4 is 2D data of shape (batch, x, y, c). '
                         'Rank 5 is 3D data of shape (batch, frames, x, y, c).')

    input_is_3d = maximas.ndim > 4

    # fill_holes is not supported in 3D
    if fill_holes_threshold and input_is_3d:
        warn('`fill_holes` is not supported for 3D data.')
        fill_holes_threshold = 0

    label_images = []
    for maxima, interior in zip(maximas, interiors):
        # squeeze out the channel dimension if passed
        maxima = nd.gaussian_filter(maxima[..., 0], maxima_smooth)
        interior = nd.gaussian_filter(interior[..., 0], interior_smooth)

        if pixel_expansion:
            fn = cube if input_is_3d else square
            interior = dilation(interior, selem=fn(pixel_expansion * 2 + 1))

        # peak_local_max is much faster but has poorer performance
        # when dealing with more ambiguous local maxima
        if maxima_algorithm == 'peak_local_max':
            coords = peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get('exclude_border', False))

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1
        else:
            # Find peaks and merge equal regions
            fn = ball if input_is_3d else disk
            # markers = h_maxima(image=maxima,
            #                    h=maxima_threshold,
            #                    selem=fn(radius))
            markers = h_maxima(image=maxima,
                               h=maxima_threshold,
                               footprint=fn(radius))

        markers = label(markers)
        label_image = watershed(-1 * interior, markers,
                                mask=interior > interior_threshold,
                                watershed_line=watershed_line)

        if label_erosion:
            label_image = f_erode_edges(label_image, label_erosion)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(label_image,
                                               min_size=small_objects_threshold)

        # fill in holes that lie completely within a segmentation label
        if fill_holes_threshold > 0:
            label_image = f_fill_holes(label_image, size=fill_holes_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images


def f_erode_edges(mask, erosion_width):
    """
    Erode edge of objects to prevent them from touching

    :param mask: (numpy.array): uniquely labeled instance mask
    :param erosion_width: erosion_width (int): integer value for pixel width to erode edges
    :return: numpy.array: mask where each instance has had the edges eroded

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


def f_fill_holes(label_img, size=10, connectivity=1):
    """
    Fills holes located completely within a given label with pixels of the same value

    :param label_img: (numpy.array): a 2D labeled image
    :param size: (int): maximum size for a hole to be filled in
    :param connectivity: (int): the connectivity used to define the hole
    :return:numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """

    output_image = np.copy(label_img)

    props = regionprops(np.squeeze(label_img.astype('int')), cache=False)
    for prop in props:
        if prop.euler_number < 1:
            patch = output_image[prop.slice]

            filled = remove_small_holes(
                ar=(patch == prop.label),
                area_threshold=size,
                connectivity=connectivity)

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image
