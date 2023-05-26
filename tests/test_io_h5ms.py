import copy
import pytest
import unittest

from stereo.core.ms_data import MSData
from stereo.io import read_gef
from stereo.io.reader import read_h5ms
from stereo.io.writer import write_h5ms
from stereo.utils._download import _download

from settings import TEST_DATA_PATH, DEMO_DATA_URL


class TestIOH5ms(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp()

        self.file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)

        self.ms_data = MSData()
        self.ms_data += read_gef(self.file_path)
        self.ms_data += copy.deepcopy(self.ms_data[0])

    def test_write_and_read(self):
        write_h5ms(self.ms_data, TEST_DATA_PATH + "SS200000135TL_D1.h5ms")
        ms_data = read_h5ms(TEST_DATA_PATH + "SS200000135TL_D1.h5ms")
        print(ms_data)
