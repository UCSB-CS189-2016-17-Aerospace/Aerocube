from enum import Enum


class Signal():

    class ImageEventSignal(Enum):
        IDENTIFY_AEROCUBES = 0x0001 0001
        GET_AEROCUBE_POSE  = 0x0001 0002

    class ResultEventSignal(Enum):
        IMP_OPERATION_OK   = 0x0002 0001
