import json
import time
import uuid
from abc import ABCMeta, abstractmethod

from jobs.settings import job_id_bundle_key
from logger import Logger
from .aeroCubeSignal import *
from .bundle import Bundle

logger = Logger('aeroCubeEvent.py', active=True, external=True)


class AeroCubeEvent(metaclass=ABCMeta):
    """
    :ivar _payload:
    :ivar _signal:
    :ivar _created_at:
    :ivar _uuid:
    """

    _INVALID_SIGNAL_FOR_EVENT = 'Invalid signal for event'
    _INVALID_PAYLOAD_NOT_BUNDLE = 'Invalid payload, must be instance of Bundle'

    _ERROR_MESSAGES = (
        _INVALID_SIGNAL_FOR_EVENT,
        _INVALID_PAYLOAD_NOT_BUNDLE
    )

    def __init__(self, bundle, signal, created_at=None, id=None):
        """
        set created_at timestamp to time.time(), e.g., the time
        since the "Epoch" (see https://en.wikipedia.org/wiki/Unix_time)
        """
        self._created_at = created_at if created_at is not None else time.time()
        self._payload = bundle if bundle is not None else Bundle()
        # Check if is_valid_signal
        self.signal = signal
        # Hex string of deterministic uuid
        self._uuid = id if id is not None else \
            uuid.uuid5(uuid.NAMESPACE_OID, "{}-{}-{}".format(self.__class__.__name__, self._signal, self._created_at)).hex

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._uuid == other.uuid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        structure = {
            'signal': str(self._signal),
            'created_at': self._created_at,
            'payload': str(self._payload),
            'class': str(self.__class__.__name__),
            'uuid': self._uuid
        }
        return str(structure)

    def to_json(self):
        json_dict = {
            'signal': str(self._signal),
            'created_at': self._created_at,
            'payload': self._payload.to_json(),
            'class': str(self.__class__.__name__),
            'uuid': self._uuid
        }
        return json.dumps(json_dict)

    @staticmethod
    def construct_from_json(event_json_str):
        logger.debug(
            AeroCubeEvent.__name__,
            'construct_from_json',
            msg='Constructing from json: \r\n{}\r\n'.format(event_json_str),
            id=None)
        loaded = json.loads(event_json_str)
        signal_int = int(loaded['signal'])
        created_at = float(loaded['created_at'])
        payload = loaded['payload']
        bundle = Bundle.construct_from_json(payload)
        class_name = loaded['class']
        uuid = loaded['uuid']
        logger.debug(
            class_name=AeroCubeEvent.__name__,
            func_name='construct_from_json',
            msg='Constructing from json: \r\n{}\r\n'.format(event_json_str),
            id=bundle.strings(job_id_bundle_key))
        event = None
        if class_name == ImageEvent.__name__:
            signal = ImageEventSignal(signal_int)
            event = ImageEvent(image_signal=signal, bundle=bundle, created_at=created_at, id=uuid)
        elif class_name == ResultEvent.__name__:
            signal = ResultEventSignal(signal_int)
            event = ResultEvent(result_signal=signal, calling_event_uuid=bundle.strings(ResultEvent.CALLING_EVENT_UUID), bundle=bundle, created_at=created_at, id=uuid)
        elif class_name == SystemEvent.__name__:
            signal = SystemEventSignal(signal_int)
            event = SystemEvent(system_signal=signal, bundle=bundle, created_at=created_at, id=uuid)
        elif class_name == StorageEvent.__name__:
            signal = StorageEventSignal(signal_int)
            event = StorageEvent(storage_signal=signal, bundle=bundle, created_at=created_at, id=uuid)
        else:
            raise TypeError('AeroCubeEvent.construct_from_json: ERROR: {} is not a valid subclass of AeroCubeEvent'.format(class_name))
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

    @property
    def uuid(self):
        return self._uuid

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
    Payload members:
    * _FILE_PATH (string)
    :cvar _FILE_PATH: payload key; path to file for image event
    """
    FILE_PATH = 'FILE_PATH'
    SCAN_ID = 'SCAN_ID'
    SCAN_MARKERS = 'SCAN_MARKERS'
    SCAN_CORNERS = 'SCAN_CORNERS'
    SCAN_MARKER_IDS = 'SCAN_MARKER_IDS'
    SCAN_POSES = 'SCAN_POSES'

    def __init__(self, image_signal, bundle=None, created_at=None, id=None):
        if bundle is None:
            bundle = Bundle()
        if created_at is None:
            created_at = time.time()
        super().__init__(bundle, image_signal, created_at, id)

    def is_valid_signal(self, signal):
        return signal in ImageEventSignal


class StorageEvent(AeroCubeEvent):
    INT_STORAGE_REL_PATH = 'INT_STORAGE_REL_PATH'
    INT_STORE_PAYLOAD_KEYS = 'INT_STORE_PAYLOAD_KEYS'
    EXT_STORAGE_TARGET = 'EXT_STORAGE_TARGET'
    EXT_STORE_PAYLOAD_KEYS = 'EXT_STORE_PAYLOAD_KEYS'

    def __init__(self, storage_signal, bundle=None, created_at=None, id=None):
        if bundle is None:
            bundle = Bundle()
        if created_at is None:
            created_at = time.time()
        if storage_signal is StorageEventSignal.STORE_EXTERNALLY and bundle.strings(self.EXT_STORAGE_TARGET) is None:
            raise AttributeError("Store external event must have external storage target!")
        super().__init__(bundle, storage_signal, created_at, id)

    def is_valid_signal(self, signal):
        return signal in StorageEventSignal

    def parse_storage_keys(self):
        type_key_pairs = None
        data = dict()
        if self.signal is StorageEventSignal.STORE_INTERNALLY:
            type_key_pairs = self.payload.iterables(self.INT_STORE_PAYLOAD_KEYS)
        elif self.signal is StorageEventSignal.STORE_EXTERNALLY:
            type_key_pairs = self.payload.iterables(self.EXT_STORE_PAYLOAD_KEYS)
        # TODO: get each bundle key's proper value by calling the correct getter
        for pair in type_key_pairs:
            bundle_type, key = pair.split(':')
            data[key] = getattr(self.payload, bundle_type)(key)
        return data

"""
Payload examples for ResultEvent or variants:
* Error message
* Camera calibration suggestion
"""


class ResultEvent(AeroCubeEvent):
    CALLING_EVENT_UUID = 'CALLING_EVENT'

    def __init__(self, result_signal, calling_event_uuid, bundle=None, created_at=None, id=None):
        if bundle is None:
            bundle = Bundle()
        if created_at is None:
            created_at = time.time()
        # print('ResultEvent.init: \r\n{}\r\n'.format(bundle))
        super().__init__(bundle, result_signal, created_at, id)
        self.payload.insert_string(ResultEvent.CALLING_EVENT_UUID, calling_event_uuid)

    def is_valid_signal(self, signal):
        return signal in ResultEventSignal


class SystemEvent(AeroCubeEvent):
    def __init__(self, system_signal, bundle=None, created_at=None, id=None):
        if bundle is None:
            bundle = Bundle()
        if created_at is None:
            created_at = time.time()
        super().__init__(bundle, system_signal, created_at, id)

    def is_valid_signal(self, signal):
        return signal in SystemEventSignal

if __name__ == '__main__':
    pass
