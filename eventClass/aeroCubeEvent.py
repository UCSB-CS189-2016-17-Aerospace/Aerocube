from abc import ABCMeta, abstractmethod
import time
from .aeroCubeSignal import *
from .bundle import Bundle
import json


class AeroCubeEvent(metaclass=ABCMeta):
    _payload = None
    _signal = None
    _created_at = None

    _INVALID_SIGNAL_FOR_EVENT = 'Invalid signal for event'
    _INVALID_PAYLOAD_NOT_BUNDLE = 'Invalid payload, must be instance of Bundle'

    _ERROR_MESSAGES = (
        _INVALID_SIGNAL_FOR_EVENT,
        _INVALID_PAYLOAD_NOT_BUNDLE
    )

    def __init__(self, bundle, created_at=None):
        """
        set created_at timestamp to time.time(), e.g., the time
        since the "Epoch" (see https://en.wikipedia.org/wiki/Unix_time)
        """
        self._created_at = created_at if created_at is not None else time.time()
        self._payload = bundle if bundle is not None else Bundle()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self._payload == other.payload and \
               self._signal == other.signal and \
               self._created_at == other.created_at

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        dict = {
            'signal': str(self._signal),
            'created_at': self._created_at,
            'payload': str(self._payload),
            'class': str(self.__class__.__name__)
        }
        return json.dumps(dict)

    @staticmethod
    def construct_from_json(event_json_str):
        print('Constructing from json: {}'.format(event_json_str))
        loaded = event_json_str
        signal_int = int(loaded['signal'])
        created_at = float(loaded['created_at'])
        payload = loaded['payload']
        bundle = Bundle.construct_from_json(payload)
        class_name = loaded['class']
        event = None
        if class_name == ImageEvent.__name__:
            signal = ImageEventSignal(signal_int)
            event = ImageEvent(image_signal=signal, bundle=bundle, created_at=created_at)
        elif class_name == ResultEvent.__name__:
            signal = ResultEventSignal(signal_int)
            event = ResultEvent(result_signal=signal, bundle=bundle, created_at=created_at)
        elif class_name == SystemEvent.__name__:
            signal = SystemEventSignal(signal_int)
            event = SystemEvent(system_signal=signal, bundle=bundle, created_at=created_at)
        else:
            pass
            # TODO: Throw err
        return event

    @property
    def created_at(self):
        return self._created_at

    @property
    def signal(self):
        return self._signal

    @property
    def payload(self):
        return self._payload

    @signal.setter
    def signal(self, other_signal):
        """
        Call class-implemented is_valid_signal method to determine if valid
        """
        if self.is_valid_signal(other_signal):
            self._signal = other_signal
        else:
            raise AttributeError(self._INVALID_SIGNAL_FOR_EVENT)

    @payload.setter
    def payload(self, other_payload):
        """
        Replaces this event's payload
        :param other_payload: the payload to replace this event's payload
        :return: raises AttributeError if other_payload is not an instance of Bundle
        """
        if isinstance(other_payload, Bundle):
            self._payload = other_payload
        else:
            raise AttributeError(self._INVALID_PAYLOAD_NOT_BUNDLE)

    def merge_payload(self, other_payload):
        """
        Merges another payload, replacing duplicate key-value pairs with values from
        :param other_payload: the payload to merge into this event's payload
        :return: raises AttributeError if other_payload is not an instance of Bundle
        """
        if isinstance(other_payload, Bundle):
            self._payload.merge_from_bundle(other_payload)
        else:
            raise AttributeError(self._INVALID_PAYLOAD_NOT_BUNDLE)

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
    def __init__(self, image_signal, bundle=Bundle(), created_at=time.time()):
        super().__init__(bundle, created_at)
        self.signal = image_signal

    def is_valid_signal(self, signal):
        return signal in ImageEventSignal
"""
Payload examples for ResultEvent or variants:
* Error message
* Camera calibration suggestion
"""


class ResultEvent(AeroCubeEvent):
    def __init__(self, result_signal, bundle=Bundle(), created_at=time.time()):
        super().__init__(bundle, created_at)
        self.signal = result_signal

    def is_valid_signal(self, signal):
        return signal in ResultEventSignal


class SystemEvent(AeroCubeEvent):
    def __init__(self, system_signal, bundle=Bundle(), created_at=time.time()):
        super().__init__(bundle, created_at)
        self.signal = system_signal

    def is_valid_signal(self, signal):
        return signal in SystemEventSignal

if __name__ == '__main__':
    print("I'm in aeroCubeEvent main!")
