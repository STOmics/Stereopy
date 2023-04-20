from ...image.wsi_split import SplitWSI
from . import CellSegmentation
from .predict import CellPredict
from .processing import f_prepocess, f_preformat, f_postformat, f_preformat_mesmer, \
    f_postformat_mesmer, f_padding, f_fusion
from ...dnn.onnx_net import OnnxNet

import numpy as np


# TensorRT/ONNX
# HE/DAPI/mIF
class Segmentation(CellSegmentation):

    def __init__(self, model_path="", net="bcdu", mode="onnx", gpu="-1", num_threads=0,
                 win_size=(256, 256), intput_size=(256, 256, 1), overlap=16):
        """

        :param model_path:
        :param net:
        :param mode:
        :param gpu:
        :param num_threads:
        """
        # self.PREPROCESS_SIZE = (8192, 8192)

        self._win_size = win_size
        self._input_size = intput_size
        self._overlap = overlap

        self._net = net
        self._gpu = gpu
        self._mode = mode
        # self._model_path = model_path
        self._model = None
        self._sess = None
        self._num_threads = num_threads
        # self._f_init_model()

    def f_init_model(self, model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)

        if self._net == "mesmer":
            self._sess = CellPredict(self._model, f_preformat_mesmer, f_postformat_mesmer)
        else:
            self._sess = CellPredict(self._model, f_preformat, f_postformat)

    def f_predict(self, img):
        """

        :param img:CHANGE
        :return:
        """
        img = f_prepocess(img)
        sp_run = SplitWSI(img, self._win_size, self._overlap, 100, True, True, False, np.uint8)
        sp_run.f_set_run_fun(self._sess.f_predict)
        sp_run.f_set_pre_fun(f_padding, self._win_size)
        sp_run.f_set_fusion_fun(f_fusion)
        _, _, pred = sp_run.f_split2run()
        pred[pred > 0] = 1
        return pred
