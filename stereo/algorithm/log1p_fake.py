# python core module
import time
from copy import deepcopy

# third part module
import numpy as np

from .algorithm_base import AlgorithmBase
from .algorithm_base import ErrorCode
# module in self project
from ..log_manager import logger
from ..plots.plot_base import PlotBase


# plot example
class PlotLog1pFake(PlotBase):

    # all methods will be auto-registered to plot_collection when method name is called
    def log1p_plot_1(self, test=123):
        logger.info(f'test_log1p_plot_1 {test}')

    def log1p_plot_2(self, **kwargs):
        logger.info(f'test_log1p_plot_2 {kwargs}')


class Log1pFake(AlgorithmBase):
    DEMO_DATA_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804a837dc46f018382c40ca51af0&code='

    def main(self, log_fast=True, inplace=True, verbose=False):
        """
            This is a fake log1p method.

            :param log_fast:
            :param inplace:
            :param verbose: TODO: verbose not finished
            :return:
        """

        the_very_beginning_time = time.time()

        if inplace:
            stereo_exp_data = self.stereo_exp_data
        else:
            stereo_exp_data = deepcopy(self.stereo_exp_data)

        if not log_fast:
            # FIXME: use time.sleep will stuck when this method is using in a web-api
            time.sleep(3.14159)
        stereo_exp_data.exp_matrix = np.log1p(stereo_exp_data.exp_matrix)

        if not inplace:
            self.pipeline_res['log1p'] = stereo_exp_data

        logger.info('log1p cost %.4f seconds', time.time() - the_very_beginning_time)
        return ErrorCode.Success

    def test_copy_safety(self):
        stereo_exp_data = deepcopy(self.stereo_exp_data)
        assert id(stereo_exp_data) != id(self.stereo_exp_data)
        assert id(stereo_exp_data.tl) != id(self.stereo_exp_data.tl)
        assert id(stereo_exp_data.plt) != id(self.stereo_exp_data.plt)
        assert id(stereo_exp_data.exp_matrix) != id(self.stereo_exp_data.exp_matrix)

    @staticmethod
    def read_135_tissue_gef():
        '''
        :return: stereo.core.stereo_exp_data.StereoExpData
        '''
        from ..utils._download import _download
        from ..io.reader import read_gef
        # without dir_path, will download to default path `./stereopy_data/`
        file_path = _download(Log1pFake.DEMO_DATA_URL)
        return read_gef(file_path)


if __name__ == "__main__":
    from stereo.io.reader import read_gef

    data = read_gef("/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    log1p_fake_obj = Log1pFake(stereo_exp_data=data)
    log1p_fake_obj.main(log_fast=True, inplace=True)
    log1p_fake_obj.test_copy_safety()
    log1p_fake_obj.memory_profile(log_fast=True, inplace=True)

    data.tl.log1p_fake(log_fast=True, inplace=True)

    data.plt.log1p_plot_1()
