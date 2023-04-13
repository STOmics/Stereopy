import unittest

from stereo.io.reader import read_gef
from stereo.utils._download import _download

from settings import DEMO_DATA_URL, TEST_DATA_PATH


class TestSCTransform(unittest.TestCase):
    DATA = None

    @classmethod
    def setUpClass(cls) -> None:
        file_path = _download(DEMO_DATA_URL, dir_str=TEST_DATA_PATH)
        cls.DATA = read_gef(file_path)

    def test_scTransform(self):
        self.DATA.tl.sctransform()

