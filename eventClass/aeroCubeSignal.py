from enum import Enum
from abc import ABCMeta, abstractmethod


class AeroCubeSignal():
    __metaclass__ = ABCMeta


class ImageEventSignal(Enum):
    IDENTIFY_AEROCUBES     = 0x00010001
    GET_AEROCUBE_POSE      = 0x00010002


class ResultEventSignal(Enum):
    # ImP Operations
    IMP_OPERATION_OK       = 0x00020001
    IMP_OP_FAILED          = 0x00020002
    # Internal Storage Operations
    INTERN_STORE_OP_OK     = 0x00020003
    INTERN_STORE_OP_FAILED = 0x00020004
    # External Communication Operations
    EXT_COMM_OP_OK         = 0x00020005
    EXT_COMM_OP_FAILED     = 0x00020006
    # Job completed
    IDENT_AEROCUBES_FIN     = 0x00020007


class SystemEventSignal(Enum):
    POWERING_OFF           = 0x00030001
