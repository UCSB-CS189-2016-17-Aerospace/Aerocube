import unittest
from collections import deque

from jobs.aeroCubeEvent import ImageEvent, ResultEvent
from jobs.aeroCubeSignal import *
from jobs.jobHandler import JobHandler


class TestEventHandler(unittest.TestCase):

    # Set Up and Tear Down functions

    @classmethod
    def setUpClass(cls):
        cls._VALID_IMAGE_EVENT = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        cls._VALID_RESULT_EVENT = ResultEvent(ResultEventSignal.IMP_OPERATION_OK,
                                              cls._VALID_IMAGE_EVENT.uuid)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._handler = JobHandler()
        self._handler.set_start_event_observer(self.event_observer)

    def event_observer(self, event):
        pass

    def tearDown(self):
        self._handler = None

    # Init

    def test_init(self):
        self.assertIsNotNone(self._handler._event_deque)

    # enqueue_event

    def test_enqueue_event(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT]))

    def test_enqueue_event_multiple_events(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self._handler.enqueue_job(self._VALID_RESULT_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT, self._VALID_RESULT_EVENT])
                         )

    def test_enqueue_event_invalid_arg(self):
        self.assertRaises(TypeError, self._handler.enqueue_job, 'non_event')

    # dequeue_event

    def test_dequeue_event(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._event_deque,
                         deque([self._VALID_IMAGE_EVENT]))
        self._handler._dequeue_job()
        self.assertEqual(self._handler._event_deque, deque())

    def test_dequeue_event_exception(self):
        self.assertRaises(IndexError, self._handler._dequeue_job)

    def test_peek_current_event(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_current_event(),
                         self._VALID_IMAGE_EVENT)

    # peek_current_event

    def test_peek_current_event_when_empty(self):
        self.assertIsNone(self._handler._peek_current_event())

    def test_peek_current_event_multiple(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self._handler.enqueue_job(self._VALID_RESULT_EVENT)
        self.assertEqual(self._handler._peek_current_event(),
                         self._VALID_IMAGE_EVENT)

    # peek_last_added_event

    def test_peek_last_added_event_when_empty(self):
        self.assertIsNone(self._handler._peek_last_added_event())

    def test_peek_last_added_event_not_empty(self):
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_last_added_event(),
                         self._VALID_IMAGE_EVENT)

    def test_peek_last_added_event_multiple(self):
        self._handler.enqueue_job(self._VALID_RESULT_EVENT)
        self._handler.enqueue_job(self._VALID_IMAGE_EVENT)
        self.assertEqual(self._handler._peek_last_added_event(),
                         self._VALID_IMAGE_EVENT)

    # any_events

    def test_any_events_not_empty(self):
        self._handler._event_deque.append(self._VALID_IMAGE_EVENT)
        self.assertTrue(self._handler.any_jobs())

    def test_any_events_empty(self):
        self.assertFalse(self._handler.any_jobs())

    # is_valid_element

    def test_positive_is_valid_element(self):
        self.assertTrue(JobHandler.is_valid_element(self._VALID_IMAGE_EVENT))

    def test_negative_is_valid_element(self):
        self.assertFalse(JobHandler.is_valid_element(1))
        self.assertFalse(JobHandler.is_valid_element('my_string'))

    # state-change

    def test_state_change_functions(self):
        self.fail() # TODO: need to implement tests


if __name__ == '__main__':
    unittest.main()
