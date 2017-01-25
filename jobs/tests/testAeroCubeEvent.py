import unittest

from jobs.aeroCubeEvent import ImageEvent, ResultEvent, SystemEvent
from jobs.aeroCubeSignal import *
from jobs.bundle import Bundle, BundleKeyError


class TestAeroCubeEventInit(unittest.TestCase):
    def test_image_event_init(self):
        event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_image_event_init_failed(self):
        self.assertRaises(AttributeError,
                          ImageEvent,
                          ResultEventSignal.IMP_OPERATION_OK)

    def test_result_event_init(self):
        calling_event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        event = ResultEvent(ResultEventSignal.IMP_OPERATION_OK,
                            calling_event.uuid)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)
        self.assertIsNotNone(event.payload.strings(ResultEvent.CALLING_EVENT_UUID))

    def test_result_event_init_failed_invalid_signal(self):
        calling_event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self.assertRaises(AttributeError,
                          ResultEvent,
                          ImageEventSignal.IDENTIFY_AEROCUBES,
                          calling_event.uuid)

    def test_result_event_init_failed_no_calling_event(self):
        self.assertRaises(AttributeError,
                          ResultEvent,
                          ResultEventSignal.IMP_OPERATION_OK,
                          None)

    def test_system_event_init(self):
        event = SystemEvent(SystemEventSignal.POWERING_OFF)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_system_event_init_failed(self):
        self.assertRaises(AttributeError,
                          SystemEvent,
                          ImageEventSignal.IDENTIFY_AEROCUBES)


class TestAeroCubeEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.expectedFailure
    def test_eq(self):
        self.fail()

    @unittest.expectedFailure
    def test_ne(self):
        self.fail()

    @unittest.expectedFailure
    def test_to_json(self):
        self.fail()

    @unittest.expectedFailure
    def test_construct_from_json(self):
        self.fail()

    @unittest.expectedFailure
    def test_construct_from_json_invalid(self):
        self.fail()

    @unittest.expectedFailure
    def test_created_at(self):
        self.fail()

    @unittest.expectedFailure
    def test_signal(self):
        self.fail()

    @unittest.expectedFailure
    def test_payload(self):
        self.fail()

    @unittest.expectedFailure
    def test_uuid(self):
        self.fail()

    @unittest.expectedFailure
    def test_merge_payload(self):
        self.fail()

    @unittest.expectedFailure
    def test_merge_payload_invalid_arg(self):
        self.fail()


class TestAeroCubeSignal(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(ImageEventSignal.IDENTIFY_AEROCUBES)

    def test_all_enum_values_unique(self):
        s1 = set([e.value for e in ImageEventSignal])
        s2 = set([e.value for e in ResultEventSignal])
        s3 = set([e.value for e in SystemEventSignal])
        self.assertSetEqual(s1.intersection(s2).intersection(s3), set())


class TestAeroCubePayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._VALID_KEY = 'VALID_KEY'
        cls._VALID_NUM = 42
        cls._VALID_STRING = 'a string'

    def setUp(self):
        self._event = ImageEvent(ImageEventSignal.GET_AEROCUBE_POSE, Bundle())

    def tearDown(self):
        del self._event._payload
        del self._event

    def test_init_payload(self):
        self.assertEqual(self._event.payload, Bundle())

    def test_retrieve_from_empty_payload(self):
        self.assertRaises(BundleKeyError, self._event._payload.strings, self._VALID_KEY)

    def test_add_to_payload(self):
        self._event._payload.insert_number(self._VALID_KEY, self._VALID_NUM)
        self.assertEqual(self._event._payload.numbers(self._VALID_KEY), self._VALID_NUM)

if __name__ == '__main__':
    unittest.main()
