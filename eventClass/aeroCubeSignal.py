from enum import Enum


class AeroCubeSignal():

    class ImageEventSignal(Enum):
        IDENTIFY_AEROCUBES = 0x00010001
        GET_AEROCUBE_POSE  = 0x00010002

    class ResultEventSignal(Enum):
        IMP_OPERATION_OK   = 0x00020001

    class SystemEventSignal(Enum):
        POWERING_OFF       = 0x00030001
