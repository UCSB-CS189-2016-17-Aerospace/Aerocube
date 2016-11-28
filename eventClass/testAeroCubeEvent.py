from .aeroCubeEvent import AeroCubeEvent, ImageEvent, ResultEvent, SystemEvent
from .aeroCubeSignal import AeroCubeSignal
from .bundle import Bundle, BundleKeyError
import unittest


class TestAeroCubeEvent(unittest.TestCase):
    def test_image_event_init(self):
        event = ImageEvent(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_image_event_init_failed(self):
        self.assertRaises(AttributeError,
                          ImageEvent,
                          AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)

    def test_result_event_init(self):
        event = ResultEvent(AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_result_event_init_failed(self):
        self.assertRaises(AttributeError,
                          ResultEvent,
                          AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

    def test_system_event_init(self):
        event = SystemEvent(AeroCubeSignal.SystemEventSignal.POWERING_OFF)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_system_event_init_failed(self):
        self.assertRaises(AttributeError,
                          SystemEvent,
                          AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)


class TestAeroCubeSignal(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

    def test_all_enum_values_unique(self):
        s1 = set([e.value for e in AeroCubeSignal.ImageEventSignal])
        s2 = set([e.value for e in AeroCubeSignal.ResultEventSignal])
        s3 = set([e.value for e in AeroCubeSignal.SystemEventSignal])
        self.assertSetEqual(s1.intersection(s2).intersection(s3), set())


class TestAeroCubePayload(unittest.TestCase):

    def setUp(self):
        self._event = ImageEvent(AeroCubeSignal.ImageEventSignal.GET_AEROCUBE_POSE)
        self._VALID_KEY = 'VALID_KEY'
        self._VALID_NUM = 42
        self._VALID_STRING = 'a string'

    def tearDown(self):
        self._event = None

    def test_init_payload(self):
        self.assertEqual(self._event._payload, Bundle())

    def test_retrieve_from_empty_payload(self):
        self.assertRaises(BundleKeyError, self._event._payload.strings, self._VALID_KEY)

    def test_add_to_payload(self):
        self._event._payload.insert_number(self._VALID_KEY, self._VALID_NUM)
        self.assertEqual(self._event._payload.numbers(self._VALID_KEY), self._VALID_NUM)

if __name__ == '__main__':
    unittest.main()
