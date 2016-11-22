from abc import ABCMeta, abstractmethod
import time
from aeroCubeSignal import AeroCubeSignal


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

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, other_signal):
        """
        Call class-implemented is_valid_signal method to determine if valid
        """
        if self.is_valid_signal(other_signal):
            self._signal = other_signal
        else:
            raise AttributeError("Invalid signal for event")

    @abstractmethod
    def is_valid_signal(self, signal):
        """
        Abstract method that should be implemented for classes inheriting from
        AeroCubeEvent, which defines what signals are acceptable for the event
        """
        raise NotImplementedError


class ImageEvent(AeroCubeEvent):
    """
    Payload includes:
    * path to image
    """
    def __init__(self, image_signal):
        super().__init__()
        self.signal = image_signal

    def is_valid_signal(self, signal):
        return signal in AeroCubeSignal.ImageEventSignal
"""
Payload examples for ResultEvent or variants:
* Error message
* Camera calibration suggestion
"""


class ResultEvent(AeroCubeEvent):
    def __init__(self, result_signal):
        super().__init__()
        self.signal = result_signal

    def is_valid_signal(self, signal):
        return signal in AeroCubeSignal.ResultEventSignal


class SystemEvent(AeroCubeEvent):
    def __init__(self, system_signal):
        super().__init__()
        self.signal = system_signal

    def is_valid_signal(self, signal):
        return signal in AeroCubeSignal.SystemEventSignal
