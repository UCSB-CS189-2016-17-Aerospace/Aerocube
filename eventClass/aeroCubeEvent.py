from abc import ABCMeta
import time


class AeroCubeEvent(metaclass=ABCMeta):
    _payload = None
    _signal = None
    _created_at = None

    def __init__(self):
        """
        set created_at timestamp to time.time(), e.g., the time
        since the "Epoch" (see https://en.wikipedia.org/wiki/Unix_time)
        """
        self._created_at = time.time()

    @property
    def created_at(self):
        return self._created_at


class ImageEvent(AeroCubeEvent):
    """
    Payload includes:
    * path to image
    """
    def __init__(self, image_signal):
        super.__init__(self)
        self._signal = image_signal

"""
Payload examples for ResultEvent or variants:
* Error message
* Camera calibration suggestion
"""


class ResultEvent(AeroCubeEvent):
    def __init__(self, result_signal):
        super.__init__(self)
        self._signal = result_signal


class SystemEvent(AeroCubeEvent):
    def __init__(self, system_signal):
        super.__init__(self)
        self._signal = system_signal
