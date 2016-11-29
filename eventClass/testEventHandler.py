import unittest
from collections import deque
from eventHandler import EventHandler
from aeroCubeEvent import ImageEvent, ResultEvent
from aeroCubeSignal import *


class TestEventHandler(unittest.TestCase):

    # Set Up and Tear Down functions

    def setUp(self):
        self._handler = EventHandler()

    def tearDown(self):
        self._handler = None

    @classmethod
    def setUpClass(cls):
        cls._VALID_IMAGE_EVENT = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        cls._VALID_RESULT_EVENT = ResultEvent(ResultEventSignal.IMP_OPERATION_OK)

    @classmethod
    def tearDownClass(cls):
        pass

    # Init

    def test_init(self):
        self.assertIsNotNone(self._handler._event_deque)

    # enqueue_event

    def test_enqueue_event(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT]))

    def test_enqueue_event_multiple_events(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self._handler.enqueue_event(self._VALID_RESULT_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT, self._VALID_RESULT_EVENT])
                         )

    def test_enqueue_event_invalid_arg(self):
        self.assertRaises(TypeError, self._handler.enqueue_event, 'non_event')

    # dequeu_event

    def test_dequeue_event(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT]))
        self._handler._dequeue_event()
        self.assertEqual(self._handler._event_deque, deque())

    def test_dequeue_event_exception(self):
        self.assertRaises(IndexError, self._handler._dequeue_event)

    def test_peek_current_event(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_current_event(),
                         self._VALID_IMAGE_EVENT)

    # peek_current_event

    def test_peek_current_event_when_empty(self):
        self.assertIsNone(self._handler._peek_current_event())

    def test_peek_current_event_multiple(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self._handler.enqueue_event(self._VALID_RESULT_EVENT)
        self.assertEqual(self._handler._peek_current_event(),
                         self._VALID_IMAGE_EVENT)

    # peek_last_added_event

    def test_peek_last_added_event_when_empty(self):
        self.assertIsNone(self._handler._peek_last_added_event())

    def test_peek_last_added_event_not_empty(self):
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_last_added_event(),
                         self._VALID_IMAGE_EVENT)

    def test_peek_last_added_event_multiple(self):
        self._handler.enqueue_event(self._VALID_RESULT_EVENT)
        self._handler.enqueue_event(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_last_added_event(),
                         self._VALID_IMAGE_EVENT)

    # any_events

    def test_any_events_not_empty(self):
        self._handler._event_deque.append(self._VALID_IMAGE_EVENT)
        self.assertTrue(self._handler.any_events())

    def test_any_events_empty(self):
        self.assertFalse(self._handler.any_events())

    # is_valid_element

    def test_positive_is_valid_element(self):
        self.assertTrue(EventHandler.is_valid_element(self._VALID_IMAGE_EVENT))

    def test_negative_is_valid_element(self):
        self.assertFalse(EventHandler.is_valid_element(1))
        self.assertFalse(EventHandler.is_valid_element('my_string'))


if __name__ == '__main__':
    unittest.main()
