import copy
import unittest

from stereo.core.ms_data import MSData
from stereo.io import read_gef
from stereo.io.reader import read_h5ms
from stereo.io.writer import write_h5ms
from stereo.utils._download import _download

class TestIOH5ms(unittest.TestCase):

    DEMO_DATA_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804a837dc46f018382c40ca51af0&code='

    def setUp(self, *args, **kwargs):
        super().setUp()

        self.file_path = _download(TestIOH5ms.DEMO_DATA_URL)

        self.ms_data = MSData()
        self.ms_data += read_gef(self.file_path)
        self.ms_data += copy.deepcopy(self.ms_data[0])

    def test_1_write(self):
        write_h5ms(self.ms_data, "./stereopy_data/SS200000135TL_D1.h5ms")

    def test_2_read(self):
        ms_data = read_h5ms("./stereopy_data/SS200000135TL_D1.h5ms")
        print(ms_data)
