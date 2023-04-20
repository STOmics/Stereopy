import enum


class StainType(enum.Enum):
    ssDNA = 'ssdna'
    DAPI = 'dapi'
    HE = 'HE'
    mIF = 'mIF'


class CellBinElement(object):
    def __init__(self):
        self.schedule = None
        self.task_name = ''
        self.sub_task_name = ''
