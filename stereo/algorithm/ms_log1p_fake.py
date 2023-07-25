from .algorithm_err_code import ErrorCode
from .ms_algorithm_base import MSDataAlgorithmBase


class MSLog1pFake(MSDataAlgorithmBase):

    def main(self, **kwargs):
        print(self.ms_data)
        print(self.pipeline_res)
        # self.pipeline_res.set_result("log1p_fake", {"log1p": 123}, type_key="log1p")
        return ErrorCode.Success
