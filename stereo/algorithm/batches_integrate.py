from stereo.algorithm.algorithm_err_code import ErrorCode
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase


class BatchesIntegrate(MSDataAlgorithmBase):

    def main(self, pca_res_key='pca', res_key='pca_integrated', **kwargs):
        assert self.ms_data.merged_data is not None, f'self.ms_data.merged_data is None'
        self.ms_data.merged_data.tl.batches_integrate(
            pca_res_key=pca_res_key,
            res_key=res_key,
            **kwargs
        )
        return ErrorCode.Success
