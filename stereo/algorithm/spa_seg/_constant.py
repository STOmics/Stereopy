from enum import Enum

class SpotSize(Enum):
    MERFISH_SPOT_SIZE = 20
    SEQFISH_SPOT_SIZE = 0.03
    SLIDESEQV2_SPOT_SIZE = 15
    VISIUM_SPOT_SIZE = 100
    STEREO_SPOT_SIZE = 1


class CellbinSize(Enum):
    # The average size of mammalian cell is approximately equal to Stereo-seq bin14*bin14
    CELLBIN_SIZE = 14