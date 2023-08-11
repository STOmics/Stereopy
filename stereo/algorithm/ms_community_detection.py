from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from .community_detection import _CommunityDetection
from .ms_algorithm_base import MSDataAlgorithmBase


class MSCommunityDetection(MSDataAlgorithmBase):

    def main(self, **kwargs):
        for data in self.ms_data.data_list:
            assert type(data) is AnnBasedStereoExpData, \
                "this MSData method can only run with AnnBasedStereoExpData temporarily"
        cd = _CommunityDetection()
        cd._main(self.ms_data.data_list, **kwargs)
        return cd
