import tifffile
import numpy as np
import os
import cv2


class Image(object):
    def __init__(self):
        self.suffix: str = ''
        self.image = None
        self.channel: int = 1
        self.dtype = None
        self.width: int = 0
        self.height: int = 0
        self.depth: int = 8
        self.ndim = 1
        # self.channel_first: bool = False

    def read(self, image, buffer=None):
        """
        update by dengzhonghan on 2023/3/1
        - support zeiss 2 channel image (channel at first)
        - support get specific channel

        Args:
            image (): image path in string format or image in numpy array format

        Returns:
            1: Fail
            0: Success

        """
        if type(image) is str:
            self.suffix = os.path.splitext(image)[1]
            if self.suffix in ['.tif', '.tiff']:
                self.image = tifffile.imread(image)  # 3 channel is RGB??
            elif self.suffix in ['.png']:
                self.image = cv2.imread(image, -1)
            else:
                return 1
        elif type(image) is np.ndarray:
            self.image = image
        elif type(image) is list and len(image) == 4:
            assert buffer is not None
            y0, y1, x0, x1 = image
            if buffer.ndim == 3:
                self.image = buffer[y0: y1, x0: x1, :]
            else:
                self.image = buffer[y0: y1, x0: x1]
        else:
            return 1
        if self.image is None or len(self.image) == 0:
            raise Exception(f"Reading {image} error!")
        self.ndim = self.image.ndim
        self.dtype = self.image.dtype
        if self.dtype == 'uint8':
            self.depth = 8
        elif self.dtype == 'uint16':
            self.depth = 16

        if self.ndim == 3:
            shape = self.image.shape
            if shape[0] in [1, 2, 3, 4]:
                self.image = self.image.transpose(1, 2, 0)
            self.height, self.width, self.channel = self.image.shape
        else:
            self.height, self.width = self.image.shape
            self.channel = 1

        return 0

    @staticmethod
    def write_s(image, output_path: str, compression=False):
        try:
            if compression:
                tifffile.imwrite(output_path, image, compression="zlib", compressionargs={"level": 8})
            else:
                tifffile.imwrite(output_path, image)
        except Exception as e:
            print(e)
            print("Write image has some error, will write without compression.")
            tifffile.imwrite(output_path, image)

    def get_channel(self, ch):
        if self.channel == 1 or ch == -1:
            return
        else:
            self.image = np.array(self.image[:, :, ch])  # cv circle raise error if no np.array
            self.channel = 1
        return