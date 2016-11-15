from abc import ABCMeta
import time


class AeroCubeEvent(metaclass=ABCMeta):
    _payload = None
    _signal = None
    _created_at = None

    def __iter__(self):
        """
        set created_at timestamp to time.time(), e.g., the time
        since the "Epoch" (see https://en.wikipedia.org/wiki/Unix_time)
        """
        self._created_at = time.time()

    @property
    def created_at(self):
        return self._created_at


class ImageEvent(AeroCubeEvent):

    def __iter__(self, image_signal):
        super.__init__(self)
        self._signal = image_signal


class ResultEvent(AeroCubeEvent):

    def __iter__(self, result_signal):
        super.__init__(self)
        self._signal = result_signal
