from enum import Enum


class AeroCubeSignal():

    class ImageEventSignal(Enum):
        IDENTIFY_AEROCUBES = 0x00010001
        GET_AEROCUBE_POSE  = 0x00010002

    class ResultEventSignal(Enum):
        IMP_OPERATION_OK   = 0x00020001

    def is_valid_signal():
        pass
