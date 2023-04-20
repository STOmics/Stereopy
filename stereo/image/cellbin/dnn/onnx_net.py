from os import path

import onnxruntime
from . import BaseNet


class OnnxNet(BaseNet):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(OnnxNet, self).__init__()
        self._providers = ['CPUExecutionProvider']
        self._providers_id = [{'device_id': -1}]
        self._model = None
        self._gpu = int(gpu)
        self._model_path = model_path
        self._input_name = 'input_1'
        self._output_name = None
        self._num_threads = num_threads
        self._f_init()

    def _f_init(self):
        if self._gpu > -1:
            self._providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._providers_id = [{'device_id': self._gpu}, {'device_id': -1}]
        self._f_load_model()

    def _f_load_model(self):
        if path.exists(self._model_path):
            sessionOptions = onnxruntime.SessionOptions()
            if (self._gpu < 0) and (self._num_threads > 0):
                sessionOptions.intra_op_num_threads = self._num_threads
            self._model = onnxruntime.InferenceSession(self._model_path, providers=self._providers,
                                                       provider_options=self._providers_id, sess_options=sessionOptions)
            self._input_name = self._model.get_inputs()[0].name
        else:
            raise Exception(f"Weight path '{self._model_path}' does not exist")

    def f_predict(self, data):
        pred = self._model.run(self._output_name, {self._input_name: data})
        return pred
