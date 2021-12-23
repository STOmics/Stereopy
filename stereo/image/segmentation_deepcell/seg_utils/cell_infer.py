import numpy as np
import os
import logging
import tensorflow as tf
from .utils import normalize, cell_watershed, resize, tile_image, untile_image, split, merge, outline, view_bar


class CellInfer(object):
    #: Metadata for the dataset used to train the model
    dataset_metadata = {}

    #: Metadata for the model and training process
    model_metadata = {}

    def __init__(self,
                 model=None,
                 model_mpp=0.65,
                 preprocessing_fn=normalize,
                 postprocessing_fn=cell_watershed,
                 format_model_output_fn=None,
                 dataset_metadata=None,
                 model_metadata=None):
        model_path = os.path.join(os.path.split(__file__)[0], 'model') if model is None else model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_image_shape = self.model.input_shape[1:]
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1

        self.required_channels = self.model_image_shape[-1]

        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self.format_model_output_fn = format_model_output_fn
        self.dataset_metadata = dataset_metadata
        self.model_metadata = model_metadata
        self.model_mpp = model_mpp

        self.logger = logging.getLogger(self.__class__.__name__)

        # Test that pre and post processing functions are callable
        if self.preprocessing_fn is not None and not callable(self.preprocessing_fn):
            raise ValueError('Preprocessing_fn must be a callable function.')
        if self.postprocessing_fn is not None and not callable(self.postprocessing_fn):
            raise ValueError('Postprocessing_fn must be a callable function.')
        if self.format_model_output_fn is not None and not callable(self.format_model_output_fn):
            raise ValueError('Format_model_output_fn must be a callable function.')


    def _resize_input(self, image, image_mpp):
        """Checks if there is a difference between image and model resolution
        and resizes if they are different. Otherwise returns the unmodified
        image.

        Args:
            image (numpy.array): Input image to resize.
            image_mpp (float): Microns per pixel for the ``image``.

        Returns:
            numpy.array: Input image resized if necessary to match ``model_mpp``
        """
        # Don't scale the image if mpp is the same or not defined
        if image_mpp not in {None, self.model_mpp}:
            shape = image.shape
            scale_factor = image_mpp / self.model_mpp
            new_shape = (int(shape[1] * scale_factor),
                         int(shape[2] * scale_factor))
            image = resize(image, new_shape, data_format='channels_last')
            self.logger.debug('Resized input from %s to %s', shape, new_shape)

        return image


    def _preprocess(self, image, **kwargs):
        """Preprocess ``image`` if ``preprocessing_fn`` is defined.
        Otherwise return ``image`` unmodified.

        Args:
            image (numpy.array): 4D stack of images
            kwargs (dict): Keyword arguments for ``preprocessing_fn``.

        Returns:
            numpy.array: The pre-processed ``image``.
        """
        if self.preprocessing_fn is not None:

            image = self.preprocessing_fn(image, **kwargs)

        return image

    def _tile_input(self, image, pad_mode='constant'):
        """Tile the input image to match shape expected by model

        Args:
            image (numpy.array): Input image to tile
            pad_mode (str): The padding mode, one of "constant" or "reflect".

        Raises:
            ValueError: Input images must have only 4 dimensions

        Returns:
            (numpy.array, dict): Tuple of tiled image and dict of tiling
            information.
        """
        if len(image.shape) != 4:
            raise ValueError('deepcell_toolbox.tile_image only supports 4d images.'
                             'Image submitted for predict has {} dimensions'.format(
                                 len(image.shape)))

        # Check difference between input and model image size
        x_diff = image.shape[1] - self.model_image_shape[0]
        y_diff = image.shape[2] - self.model_image_shape[1]

        # Check if the input is smaller than model image size
        if x_diff < 0 or y_diff < 0:
            # Calculate padding
            x_diff, y_diff = abs(x_diff), abs(y_diff)
            x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
            y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

            image = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')

            tiles, tiles_info = tile_image(image, model_input_shape=self.model_image_shape,
                                           stride_ratio=0.75, pad_mode=pad_mode)
            tiles_info['padding'] = True
            tiles_info['padding_x_pad'] = x_pad
            tiles_info['padding_y_pad'] = y_pad

        # Otherwise tile images larger than model size
        else:
            # Tile images, needs 4d
            tiles, tiles_info = tile_image(image, model_input_shape=self.model_image_shape,
                                           stride_ratio=0.75, pad_mode=pad_mode)

        return tiles, tiles_info


    def _untile_output(self, output_tiles, tiles_info):
        """Untiles either a single array or a list of arrays
        according to a dictionary of tiling specs

        Args:
            output_tiles (numpy.array or list): Array or list of arrays.
            tiles_info (dict): Tiling specs output by the tiling function.

        Returns:
            numpy.array or list: Array or list according to input with untiled images
        """
        # If padding was used, remove padding
        if tiles_info.get('padding', False):
            def _process(im, tiles_info):
                out = untile_image(im, tiles_info, model_input_shape=self.model_image_shape)
                x_pad, y_pad = tiles_info['padding_x_pad'], tiles_info['padding_y_pad']
                out = out[:, x_pad[0]:-x_pad[1], y_pad[0]:-y_pad[1], :]
                return out
        # Otherwise untile
        else:
            def _process(im, tiles_info):
                out = untile_image(im, tiles_info, model_input_shape=self.model_image_shape)
                return out

        if isinstance(output_tiles, list):
            output_images = [_process(o, tiles_info) for o in output_tiles]
        else:
            output_images = _process(output_tiles, tiles_info)

        return output_images

    def _format_model_output(self, output_images):
        """Applies formatting function the output from the model if one was
        provided. Otherwise, returns the unmodified model output.

        Args:
            output_images: stack of untiled images to be reformatted

        Returns:
            dict or list: reformatted images stored as a dict, or input
            images stored as list if no formatting function is specified.
        """
        if self.format_model_output_fn is not None:
            formatted_images = self.format_model_output_fn(output_images)
            return formatted_images
        else:
            return output_images


    def _run_model(self,
                   image,
                   batch_size=4,
                   pad_mode='constant',
                   preprocess_kwargs={}):
        """Run the model to generate output probabilities on the data.

        Args:
            image (numpy.array): Image with shape ``[batch, x, y, channel]``
            batch_size (int): Number of images to predict on per batch.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to
                the preprocessing function.

        Returns:
            numpy.array: Model outputs
        """
        # Preprocess image if function is defined
        image = self._preprocess(image, **preprocess_kwargs)

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = self._tile_input(image, pad_mode=pad_mode)

        # Run images through model
        output_tiles = self.model.predict(tiles, batch_size=batch_size)


        # Untile images
        output_images = self._untile_output(output_tiles, tiles_info)

        # restructure outputs into a dict if function provided
        formatted_images = self._format_model_output(output_images)

        return formatted_images


    def _postprocess(self, image, **kwargs):
        """Applies postprocessing function to image if one has been defined.
        Otherwise returns unmodified image.

        Args:
            image (numpy.array or list): Input to postprocessing function
                either an ``numpy.array`` or list of ``numpy.arrays``.

        Returns:
            numpy.array: labeled image
        """
        if self.postprocessing_fn is not None:

            image = self.postprocessing_fn(image, **kwargs)

            # Restore channel dimension if not already there
            if len(image.shape) == self.required_rank - 1:
                image = np.expand_dims(image, axis=-1)

        elif isinstance(image, list) and len(image) == 1:
            image = image[0]

        return image


    def _resize_output(self, image, original_shape):
        """Rescales input if the shape does not match the original shape
        excluding the batch and channel dimensions.

        Args:
            image (numpy.array): Image to be rescaled to original shape
            original_shape (tuple): Shape of the original input image

        Returns:
            numpy.array: Rescaled image
        """
        if not isinstance(image, list):
            image = [image]

        for i in range(len(image)):
            img = image[i]
            # Compare x,y based on rank of image
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]

            # Resize if same is false
            if not same:
                # Resize function only takes the x,y dimensions for shape
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape,
                             data_format='channels_last',
                             labeled_image=True)
            image[i] = img

        if len(image) == 1:
            image = image[0]

        return image


    def predict_image(self,
                image,
                batch_size=4,
                image_mpp=None,
                pad_mode='reflect',
                preprocess_kwargs=None,
                postprocess_kwargs=None):
        if preprocess_kwargs is None:
            preprocess_kwargs = {}

        if postprocess_kwargs is None:
            postprocess_kwargs = {
                'min_distance': 10,
                'detection_threshold': 0.1,
                'distance_threshold': 0.01,
                'exclude_border': False,
                'small_objects_threshold': 0
            }

        # Check input size of image
        if len(image.shape) != self.required_rank:
            raise ValueError('Input data must have {} dimensions. '
                             'Input data only has {} dimensions'.format(
                self.required_rank, len(image.shape)))

        if image.shape[-1] != self.required_channels:
            raise ValueError('Input data must have {} channels. '
                             'Input data only has {} channels'.format(
                self.required_channels, image.shape[-1]))

        # Resize image, returns unmodified if appropriate
        resized_image = self._resize_input(image, image_mpp)

        # Generate model outputs
        output_images = self._run_model(
            image=resized_image, batch_size=batch_size,
            pad_mode=pad_mode, preprocess_kwargs=preprocess_kwargs
        )

        # Postprocess predictions to create label image
        label_image = self._postprocess(output_images, **postprocess_kwargs)

        # Resize label_image back to original resolution if necessary
        label_image = self._resize_output(label_image, image.shape)


        return label_image


def cellInfer(file, size, overlap=100, model_path=None):

    # split -> predict -> merge
    if isinstance(file, list):
        file_list = file
    else:
        file_list = [file]

    result = []

    for idx, image in enumerate(file_list):

        # split
        img_list, x_list, y_list = split(image, size, overlap)

        # load model
        seg_app = CellInfer(model=model_path)

        label_list = []
        label_list_outline = []
        # deepcell predict
        total_num = len(img_list)
        for idy, img in enumerate(img_list):


            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            if len(img.shape) != 4:
                print('input data shape wrong!!')

            labeled_img = seg_app.predict_image(img)
            label = np.squeeze(labeled_img)

            label_list.append(label)
            # label_list_outline.append(outline(label))
            view_bar('【image %d/%d】 batch' % (idx + 1, len(file_list)), idy + 1, total_num, end='\n' if idy + 1 == total_num else '')


        # merge mask and save
        # merge_label_outline = merge(label_list_outline, x_list, y_list, image.shape)
        merge_label = merge(label_list, x_list, y_list, image.shape)

        merge_label = np.where(merge_label !=0, 1, 0).astype(np.uint8)
        result.append(merge_label) #, merge_label_outline

    return result