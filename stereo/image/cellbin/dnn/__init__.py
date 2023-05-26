# Deep Neural Networks (dnn module)
# (Pytorch, TensorFlow) models with ONNX.
# In this section you will find the functions, which describe how to run classification, segmentation and detection
# DNN models with ONNX.

from abc import ABC, abstractmethod


class BaseNet(ABC):
    @abstractmethod
    def _f_load_model(self):
        return

    @abstractmethod
    def f_predict(self, img):
        return
