from enum import Enum


class AeroCubeSignal():
    pass


class ImageEventSignal(Enum):
    IDENTIFY_AEROCUBES     = 0x00010001
    GET_AEROCUBE_POSE      = 0x00010002

    def __str__(self):
        return str(self.value)

class StorageEventSignal(Enum):
    STORE_INTERNALLY       = 0.00030001
    STORE_EXTERNALLY       = 0.00030002

    def __str__(self):
        return str(self.value)

class ResultEventSignal(Enum):
    OK                     = 0x00020000
    WARN                   = 0x00020001
    ERROR                  = 0x00020002
    # ImP Operations
    IMP_OPERATION_OK       = 0x00020003
    IMP_OP_FAILED          = 0x00020004
    # Internal Storage Operations
    INTERN_STORE_OP_OK     = 0x00020005
    INTERN_STORE_OP_FAILED = 0x00020006
    # External Communication Operations
    EXT_COMM_OP_OK         = 0x00020007
    EXT_COMM_OP_FAILED     = 0x00020008
    # Job completed
    IDENT_AEROCUBES_FIN    = 0x00020009

    def __str__(self):
        return str(self.value)


class SystemEventSignal(Enum):
    POWERING_OFF           = 0x00030001

    def __str__(self):
        return str(self.value)