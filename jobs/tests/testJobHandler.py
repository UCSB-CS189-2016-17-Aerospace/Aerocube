import unittest
from collections import deque
from unittest.mock import Mock

from ..aeroCubeJob import *
from ..jobHandler import JobHandler


class TestJobHandler(unittest.TestCase):

    # Set Up and Tear Down functions

    @classmethod
    def setUpClass(cls):
        cls.image_event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        cls.other_image_event = ImageEvent(ImageEventSignal.GET_AEROCUBE_POSE)
        cls.result_event = ResultEvent(ResultEventSignal.IMP_OPERATION_OK,
                                       cls.image_event.uuid)
        cls.job_event_leaf_node = AeroCubeJobEventNode(cls.image_event)
        cls.job_event_root_node = AeroCubeJobEventNode(cls.other_image_event, cls.job_event_leaf_node)
        cls.valid_job = AeroCubeJob(cls.job_event_root_node)
        cls.other_valid_job = AeroCubeJob(cls.job_event_leaf_node)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._on_event_mock = Mock()
        self._on_enqueue_mock = Mock()
        self._on_dequeue_mock = Mock()
        self._handler = JobHandler(self._on_event_mock, self._on_enqueue_mock, self._on_dequeue_mock)

    def tearDown(self):
        self._handler._job_deque.clear()
        self._handler._job_deque = None
        self._handler = None

    # Init

    def test_init(self):
        on_event_mock = Mock()
        on_enqueue_mock = Mock()
        on_dequeue_mock = Mock()

        job_handler = JobHandler(on_event_mock, on_enqueue_mock, on_dequeue_mock)
        # Deque is set and empty
        self.assertIsNotNone(self._handler._job_deque)
        self.assertEqual(0, len(self._handler._job_deque))
        job_handler._on_start_event(self.image_event)
        job_handler._on_job_enqueue(self.valid_job)
        job_handler._on_job_dequeue(self.other_valid_job)
        # Each handler set appropriately
        on_event_mock.assert_called_once_with(self.image_event)
        on_enqueue_mock.assert_called_once_with(self.valid_job)
        on_dequeue_mock.assert_called_once_with(self.other_valid_job)
        # State is STARTED
        self.assertEqual(JobHandler.State.STARTED, job_handler._state)

    # set_observers

    def test_set_start_event_observer(self):
        on_event_mock = Mock()
        self._handler.set_start_event_observer(on_event_mock)
        self._handler._on_start_event(self.image_event)
        on_event_mock.assert_called_once_with(self.image_event)

    def test_set_start_dequeue_observer(self):
        on_dequeue_mock = Mock()
        self._handler.set_job_dequeue_observer(on_dequeue_mock)
        self._handler._on_job_dequeue(self.valid_job)
        on_dequeue_mock.assert_called_once_with(self.valid_job)

    def test_set_start_enqueue_observer(self):
        on_enqueue_mock = Mock()
        self._handler.set_job_enqueue_observer(on_enqueue_mock)
        self._handler._on_job_enqueue(self.valid_job)
        on_enqueue_mock.assert_called_once_with(self.valid_job)

    # enqueue_event

    def test_enqueue_job(self):
        self._handler.enqueue_job(self.valid_job)
        self.assertEqual(self._handler._job_deque, deque([self.valid_job]))

    def test_enqueue_multiple_jobs(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.enqueue_job(self.other_valid_job)
        self.assertEqual(self._handler._job_deque,
                         deque([self.valid_job, self.other_valid_job]))

    def test_enqueue_job_runs_start_sending_events(self):
        self._handler._start_sending_events = Mock()
        self._handler.enqueue_job(self.valid_job)
        # Simulate state change that is not run due to mock function
        self._handler._state = JobHandler.State.PENDING
        self._handler.enqueue_job(self.other_valid_job)
        self._handler._start_sending_events.assert_called_once_with()

    def test_enqueue_job_raises_on_non_job(self):
        self.assertRaises(TypeError, self._handler.enqueue_job)
        self.assertRaises(TypeError, self._handler.enqueue_job, self.image_event)

    # dequeue

    def test_dequeue_job(self):
        self._handler.enqueue_job(self.valid_job)
        self.assertEqual(self._handler._job_deque,
                         deque([self.valid_job]))
        self._handler._dequeue_job()
        self.assertEqual(self._handler._job_deque, deque())

    def test_dequeue_event_exception(self):
        self.assertRaises(IndexError, self._handler._dequeue_job)

    # peek

    def test_peek_current_job_when_empty(self):
        self.assertIsNone(self._handler._peek_current_event())

    def test_peek_current_job(self):
        self._handler.enqueue_job(self.valid_job)
        self.assertEqual(self._handler._peek_current_event(),
                         self.valid_job.current_event)

    def test_peek_current_job_multiple(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.enqueue_job(self.other_valid_job)
        self.assertEqual(self._handler._peek_current_event(),
                         self.valid_job.current_event)
        self.assertNotEqual(self._handler._peek_current_event(),
                            self.other_valid_job.current_event)

    # peek_last

    def test_peek_last_added_job_when_empty(self):
        self.assertIsNone(self._handler._peek_last_added_job())

    def test_peek_last_added_job_not_empty(self):
        self._handler.enqueue_job(self.valid_job)
        self.assertEqual(self._handler._peek_last_added_job(),
                         self.valid_job)

    def test_peek_last_added_event_multiple(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.enqueue_job(self.other_valid_job)
        self.assertEqual(self._handler._peek_last_added_job(),
                         self.other_valid_job)

    # any_jobs

    def test_any_jobs_not_empty(self):
        self._handler._job_deque.append(self.valid_job)
        self.assertTrue(self._handler.has_jobs)
        self._handler._job_deque.append(self.other_valid_job)
        self.assertTrue(self._handler.has_jobs)

    def test_any_events_empty(self):
        self.assertFalse(self._handler.has_jobs)

    # is_valid_element

    def test_positive_is_valid_element(self):
        self.assertTrue(JobHandler.is_valid_element(self.valid_job))

    def test_negative_is_valid_element(self):
        self.assertFalse(JobHandler.is_valid_element(1))
        self.assertFalse(JobHandler.is_valid_element('my_string'))
        self.assertFalse(JobHandler.is_valid_element(None))
        self.assertFalse(JobHandler.is_valid_element(self.image_event))

    # properties

    def test_property_state(self):
        self.assertEqual(self._handler.state, self._handler._state)

    def test_has_jobs(self):
        self.assertFalse(self._handler.has_jobs)
        self._handler._job_deque.append(self.valid_job)
        self.assertTrue(self._handler.has_jobs)

    # state changes

    def test_state_init(self):
        self.assertEqual(self._handler._state, JobHandler.State.STARTED)

    def test_state_force_stop(self):
        self._handler.force_stop()
        self.assertEqual(self._handler._state, JobHandler.State.STOPPED)

    def test_state_enqueue(self):
        self._handler.enqueue_job(self.valid_job)
        self.assertEqual(self._handler._state, JobHandler.State.PENDING)

    def test_state_stop(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.stop()
        self.assertEqual(self._handler._state, JobHandler.State.PENDING_STOP_ON_RESOLVE)

    def test_state_restart_empty(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.force_stop()
        self._handler._job_deque.popleft()
        self._handler.restart()
        self.assertEqual(self._handler._state, JobHandler.State.STARTED)

    def test_state_restart_has_jobs(self):
        self._handler.enqueue_job(self.valid_job)
        self._handler.force_stop()
        self._handler.restart()
        self.assertEqual(self._handler._state, JobHandler.State.PENDING)

    # can_state_resolve

    def test_can_state_resolve(self):
        self._handler._state = JobHandler.State.PENDING_STOP_ON_RESOLVE
        self.assertTrue(self._handler._can_state_resolve())
        self._handler._state = JobHandler.State.PENDING
        self.assertTrue(self._handler._can_state_resolve())
        self._handler._state = JobHandler.State.STARTED
        self.assertRaises(JobHandler.NotAllowedInStateException, self._handler._can_state_resolve)
        self._handler._state = JobHandler.State.STOPPED
        self.assertRaises(JobHandler.NotAllowedInStateException, self._handler._can_state_resolve)

if __name__ == '__main__':
    unittest.main()
