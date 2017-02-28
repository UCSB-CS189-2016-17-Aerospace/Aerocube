import unittest
import numpy as np

from jobs.aeroCubeJob import *


class TestAeroCubeJobEventNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._IMAGE_EVENT = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        cls._OK_RESULT_EVENT = ResultEvent(ResultEventSignal.OK, cls._IMAGE_EVENT.uuid)
        cls._WARN_RESULT_EVENT = ResultEvent(ResultEventSignal.WARN, cls._IMAGE_EVENT.uuid)
        cls._ERR_RESULT_EVENT = ResultEvent(ResultEventSignal.ERROR, cls._IMAGE_EVENT.uuid)
        cls._LEAF_EVENT_NODE = AeroCubeJobEventNode(event=cls._IMAGE_EVENT)
        from logger import Logger
        Logger.prevent_external()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        self.assertIsNotNone(AeroCubeJobEventNode(event=self._IMAGE_EVENT))

    def test_init_raises_on_non_result_event(self):
        self.assertRaises(AttributeError, AeroCubeJobEventNode, None)
        self.assertRaises(AttributeError, AeroCubeJobEventNode, self._OK_RESULT_EVENT)

    def test_init_event(self):
        job_event_node = AeroCubeJobEventNode(self._IMAGE_EVENT)
        self.assertEqual(job_event_node._event, self._IMAGE_EVENT)

    def test_init_ok_event_mapping(self):
        job_event_node = AeroCubeJobEventNode(self._IMAGE_EVENT, ok_event_node=self._LEAF_EVENT_NODE)
        self.assertIsNotNone(job_event_node._event_signal_map[ResultEventSignal.OK])

    def test_init_warn_event_mapping(self):
        job_event_node = AeroCubeJobEventNode(self._IMAGE_EVENT, warn_event_node=self._LEAF_EVENT_NODE)
        self.assertIsNotNone(job_event_node._event_signal_map[ResultEventSignal.WARN])

    def test_init_err_event_mapping(self):
        job_event_node = AeroCubeJobEventNode(self._IMAGE_EVENT, err_event_node=self._LEAF_EVENT_NODE)
        self.assertIsNotNone(job_event_node._event_signal_map[ResultEventSignal.ERROR])

    def test_uuid_property(self):
        self.assertEqual(self._LEAF_EVENT_NODE.event_uuid, self._IMAGE_EVENT.uuid)

    def test_event_property(self):
        self.assertEqual(self._LEAF_EVENT_NODE.event, self._IMAGE_EVENT)

    def test_next_event_raises_without_event(self):
        self.assertRaises(TypeError, self._LEAF_EVENT_NODE.next_event_node, None)

    def test_next_event_raises_with_wrong_event_type(self):
        self.assertRaises(TypeError, self._LEAF_EVENT_NODE.next_event_node, self._IMAGE_EVENT)

    def test_next_event_raises_on_key_not_defined(self):
        self.assertRaises(LookupError,
                          self._LEAF_EVENT_NODE.next_event_node,
                          ResultEvent(ResultEventSignal.EXT_COMM_OP_FAILED, self._IMAGE_EVENT.uuid))

    def test_next_event_returns_proper_event_node(self):
        job_event_node = AeroCubeJobEventNode(self._IMAGE_EVENT, ok_event_node=self._LEAF_EVENT_NODE)
        self.assertEqual(job_event_node._event_signal_map[ResultEventSignal.OK], self._LEAF_EVENT_NODE)


class TestAeroCubeJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    def setUp(self):
        self._IMAGE_EVENT = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self._IMAGE_EVENT_1 = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self._IMAGE_EVENT_2 = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self._IMAGE_EVENT_3 = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        self._IMAGE_EVENT_LEAF_NODE = AeroCubeJobEventNode(self._IMAGE_EVENT)
        self._IMAGE_EVENT_OK_NODE = AeroCubeJobEventNode(self._IMAGE_EVENT_1, ok_event_node=self._IMAGE_EVENT_LEAF_NODE)
        # IMAGE_EVENT_WARN_NODE's tree has height 2 (3 layers of nodes)
        self._IMAGE_EVENT_WARN_NODE = AeroCubeJobEventNode(self._IMAGE_EVENT_2, warn_event_node=self._IMAGE_EVENT_OK_NODE)
        self._IMAGE_EVENT_ERR_NODE = AeroCubeJobEventNode(self._IMAGE_EVENT_3, err_event_node=self._IMAGE_EVENT_LEAF_NODE)
        self._JOB = AeroCubeJob(self._IMAGE_EVENT_OK_NODE)

    def tearDown(self):
        del self._JOB._root_event_node
        del self._JOB._current_node
        del self._JOB

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init_raises_with_none_event_node(self):
        self.assertRaises(AttributeError, AeroCubeJob, None)

    def test_init_raises_with_improper_event_node_arg(self):
        self.assertRaises(AttributeError, AeroCubeJob, self._IMAGE_EVENT)

    def test_init_passes_created_at_arg(self):
        created_at = time.time() - 1
        self.assertEqual(AeroCubeJob(self._IMAGE_EVENT_LEAF_NODE, created_at=created_at).created_at, created_at)

    def test_init_defaults_to_current_time(self):
        before_created_at = time.time() - 1
        self.assertLess(before_created_at, AeroCubeJob(self._IMAGE_EVENT_LEAF_NODE).created_at)

    def test_init(self):
        self.assertEqual(self._JOB._root_event_node, self._IMAGE_EVENT_OK_NODE)
        self.assertEqual(self._JOB._current_node, self._IMAGE_EVENT_OK_NODE)
        self.assertEqual(self._JOB._root_event_node, self._JOB._current_node)
        self.assertIsNotNone(self._JOB._uuid)

    def test_properties(self):
        self.assertEqual(self._JOB.root_event, self._JOB._root_event_node.event)
        self.assertEqual(self._JOB.current_event, self._JOB._current_node.event)
        self.assertEqual(self._JOB.created_at, self._JOB._created_at)
        self.assertEqual(self._JOB.uuid, self._JOB._uuid)
        self.assertEqual(self._JOB.is_finished, self._JOB._current_node is None)

    def test_update_node_raises_on_non_result_event(self):
        self.assertRaises(AttributeError, self._JOB.update_current_node, None)
        self.assertRaises(AttributeError, self._JOB.update_current_node, self._IMAGE_EVENT)

    def test_update_node_raises_on_non_matching_result_event(self):
        other_event = ImageEvent(ImageEventSignal.GET_AEROCUBE_POSE)
        result_event = ResultEvent(ResultEventSignal.OK, other_event.uuid)
        self.assertRaises(AttributeError, self._JOB.update_current_node, result_event)

    def test_update_node_updates_current_node(self):
        # TODO: all self._IMAGE_EVENT* have the same UUID; is this expected behavior? I feel like it's certainly
        # unwanted. This test should be failing, as self._IMAGE_EVENT is not the proper event to respond to.
        result_event = ResultEvent(ResultEventSignal.WARN, self._IMAGE_EVENT.uuid)
        job = AeroCubeJob(self._IMAGE_EVENT_WARN_NODE)
        job.update_current_node(result_event)
        self.assertEqual(job.current_event, self._IMAGE_EVENT_OK_NODE.event)
        second_result_event = ResultEvent(ResultEventSignal.OK, self._IMAGE_EVENT.uuid)
        job.update_current_node(second_result_event)
        self.assertEqual(job.current_event, self._IMAGE_EVENT)

    def test_update_node_alters_next_payload_if_flag_set(self):
        result_event = ResultEvent(ResultEventSignal.OK, self._JOB.current_event.uuid)
        self._JOB.current_event.payload.insert_string(ImageEvent.FILE_PATH, 'file_path')
        self._JOB.update_current_node(result_event, merge_payload=True)
        self.assertIsNotNone(self._JOB.current_event.payload.strings(ImageEvent.FILE_PATH))

    def test_update_node_does_not_alter_next_payload_if_flag_not_set(self):
        # TODO: test is failing
        result_event = ResultEvent(ResultEventSignal.OK, self._JOB.current_event.uuid)
        self._JOB.current_event.payload.insert_string(ImageEvent.FILE_PATH, 'file_path')
        self._JOB.update_current_node(result_event)
        self.assertRaises(Exception, self._JOB.current_event.payload.strings, ImageEvent.FILE_PATH)


class TestAeroCubeJobConstructors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    def test_create_image_upload_job(self):
        img_path = "my_phony_path"
        job = AeroCubeJob.create_image_upload_job(img_path,
                                                  int_storage=True,
                                                  ext_store_target='FIREBASE')
        self.assertIsNotNone(job)
        self.assertIsInstance(job.root_event, ImageEvent)
        self.assertEqual(job.root_event.payload.strings(ImageEvent.FILE_PATH), img_path)

        int_store_node = job._current_node.event_signal_map[ResultEventSignal.OK]
        self.assertIsInstance(int_store_node.event, StorageEvent)
        self.assertEqual(int_store_node.event.signal, StorageEventSignal.STORE_INTERNALLY)
        np.testing.assert_equal(int_store_node.event.payload.raws(StorageEvent.INT_STORE_PAYLOAD_KEYS),
                                [ImageEvent.SCAN_ID, ImageEvent.SCAN_CORNERS, ImageEvent.SCAN_MARKER_IDS])

        ext_store_node = int_store_node.event_signal_map[ResultEventSignal.OK]
        self.assertIsInstance(ext_store_node.event, StorageEvent)
        self.assertEqual(ext_store_node.event.signal, StorageEventSignal.STORE_EXTERNALLY)
        self.assertEqual(ext_store_node.event.payload.strings(StorageEvent.EXT_STORAGE_TARGET), 'FIREBASE')