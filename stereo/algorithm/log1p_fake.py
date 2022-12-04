# python core module
import time
from copy import deepcopy

# third part module
import numpy as np

# module in self project
from stereo.log_manager import logger
from stereo.algorithm.algorithm_base import AlgorithmBase, ErrorCode


class Log1pFake(AlgorithmBase):
    log_base: int

    def main(self, log_fast=True, inplace=True, verbose=False):
        """
            This is a fake log1p method.

            :param log_fast:
            :param inplace:
            :param verbose: TODO: verbose not finished
            :return:
        """

        not_used_variable = None
        ircorrect_spell_word = 'should be `incorrect`'
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


if __name__ == "__main__":
    from stereo.io.reader import read_gef

    data = read_gef("/mnt/d/projects/stereopy_dev/demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    log1p_fake_obj = Log1pFake(stereo_exp_data=data)
    log1p_fake_obj.main(log_fast=True, inplace=True)
    log1p_fake_obj.test_copy_safety()
    log1p_fake_obj.test_memory_profile(log_fast=True, inplace=True)

    data.tl.log1p_fake(log_fast=True, inplace=True)
