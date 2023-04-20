from tifffile import tifffile

from . import CellBinElement
from ..dnn.cseg.detector import Segmentation
from ..dnn.cseg.cell_trace import get_trace as get_t


class CellSegmentation(CellBinElement):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(CellSegmentation, self).__init__()

        self._MODE = "onnx"
        self._NET = "bcdu"
        self._WIN_SIZE = (256, 256)
        self._INPUT_SIZE = (256, 256, 1)
        self._OVERLAP = 16

        self._gpu = gpu
        self._model_path = model_path
        self._num_threads = num_threads

        self._cell_seg = Segmentation(
            net=self._NET,
            mode=self._MODE,
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            intput_size=self._INPUT_SIZE,
            overlap=self._OVERLAP
        )
        self._cell_seg.f_init_model(model_path=self._model_path)

    def run(self, img):
        mask = self._cell_seg.f_predict(img)
        return mask

    @staticmethod
    def get_trace(mask):
        return get_t(mask)


def cell_seg_v3(input_path, out_path, model_path, gpu="0", num_threads=0):
    cell_bcdu = CellSegmentation(
        model_path=model_path,
        gpu=gpu,
        num_threads=num_threads
    )
    img = tifffile.imread(input_path)
    mask = cell_bcdu.run(img)
    CellSegmentation.get_trace(mask)
    tifffile.imwrite(out_path, mask)
