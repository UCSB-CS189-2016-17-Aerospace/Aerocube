import unittest
from collections import deque
from eventHandler import EventHandler
from aeroCubeEvent import ImageEvent, ResultEvent
from aeroCubeSignal import AeroCubeSignal


class TestEventHandler(unittest.TestCase):
    VALID_IMAGE_EVENT = ImageEvent(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)
    VALID_RESULT_EVENT = ResultEvent(AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)

    def test_init(self):
        handler = EventHandler()
        self.assertIsNotNone(handler._event_deque)

    def test_any_events_not_empty(self):
        handler = EventHandler()
        handler._event_deque.append(self.VALID_IMAGE_EVENT)
        self.assertTrue(handler.any_events())

    def test_enqueue_event(self):
        handler = EventHandler()
        handler.enqueue_event(self.VALID_IMAGE_EVENT)
        self.assertEqual(handler._event_deque, deque([self.VALID_IMAGE_EVENT]))

    def test_enqueue_event_multiple_events(self):
        handler = EventHandler()
        handler.enqueue_event(self.VALID_IMAGE_EVENT)
        handler.enqueue_event(self.VALID_RESULT_EVENT)
        self.assertEqual(handler._event_deque,
                         deque([self.VALID_IMAGE_EVENT, self.VALID_RESULT_EVENT])
                         )

    def test_enqueue_event_invalid_arg(self):
        handler = EventHandler()
        self.assertRaises(TypeError, handler.enqueue_event, 'non_event')

    def test_dequeue_event(self):
        handler = EventHandler()
        handler.enqueue_event(self.VALID_IMAGE_EVENT)
        self.assertEqual(handler._event_deque, deque([self.VALID_IMAGE_EVENT]))
        handler.dequeue_event()
        self.assertEqual(handler._event_deque, deque())

    def test_dequeue_event_exception(self):
        handler = EventHandler()
        self.assertRaises(IndexError, handler.dequeue_event)

    def test_peek_current_event(self):
        handler = EventHandler()
        handler.enqueue_event(self.VALID_IMAGE_EVENT)
        self.assertEqual(handler.peek_current_event(), self.VALID_IMAGE_EVENT)

    def test_peek_current_event_multiple(self):
        handler = EventHandler()
        handler.enqueue_event(self.VALID_IMAGE_EVENT)
        handler.enqueue_event(self.VALID_RESULT_EVENT)
        self.assertEqual(handler.peek_current_event(), self.VALID_IMAGE_EVENT)

    def test_any_events_empty(self):
        handler = EventHandler()
        self.assertFalse(handler.any_events())

    def test_positive_is_valid_element(self):
        self.assertTrue(EventHandler.is_valid_element(self.VALID_IMAGE_EVENT))

    def test_negative_is_valid_element(self):
        self.assertFalse(EventHandler.is_valid_element(1))
        self.assertFalse(EventHandler.is_valid_element('my_string'))


if __name__ == '__main__':
    unittest.main()
