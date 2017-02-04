from .aeroCubeEvent import *
from .aeroCubeSignal import ResultEventSignal


class AeroCubeJobEventNode:
    """
    :ivar _event
    :ivar _event_signal_map
    """

    def __init__(self, event, ok_event_node=None, warn_event_node=None, err_event_node=None):
        if not isinstance(event, AeroCubeEvent):
            raise AttributeError('Invalid event parameter, must be instance of AeroCubeEvent')
        if isinstance(event, ResultEvent):
            raise AttributeError('Event nodes may not be created with a ResultEvent')
        self._event = event
        self._event_signal_map = {
            ResultEventSignal.OK: ok_event_node,
            ResultEventSignal.WARN: warn_event_node,
            ResultEventSignal.ERROR: err_event_node
        }

    def __eq__(self, other):
        return isinstance(other, AeroCubeJobEventNode) and \
               other.event == self._event and \
               self._event_signal_map == other.event_signal_map

    def __ne__(self, other):
        return not self == other

    @property
    def event_uuid(self):
        return self._event.uuid

    @property
    def event(self):
        return self._event

    @property
    def event_signal_map(self):
        return self._event_signal_map

    def next_event_node(self, result_event):
        """
        next_event_node attempts to use a result_event to determine the next event_node
        :param result_event:
        :return None
        :raises LookupError if init is not updated to match potential result event signals
        """
        if not isinstance(result_event, ResultEvent):
            raise TypeError('Can only traverse jobs with a result_event')
        try:
            return self._event_signal_map[result_event.signal]
        except KeyError as err:
            raise LookupError('Cannot process result event with signal {} for current event with signal {}'
                              .format(result_event.signal, self._event.signal))


class AeroCubeJob:
    """
    :ivar _root_event_node:
    :ivar _current_node:
    :ivar _created_at:
    :ivar _uuid:
    """

    def __init__(self, root_event_node, created_at=time.time(), id=None):
        if not isinstance(root_event_node, AeroCubeJobEventNode):
            raise AttributeError('Invalid event node parameter, must be instance of AeroCubeJobEventNode')
        self._root_event_node = root_event_node
        self._current_node = root_event_node
        self._created_at = created_at
        self._uuid = id if id is not None else \
            uuid.uuid5(uuid.NAMESPACE_OID, "{}-{}".format(self.__class__.__name__, self._created_at)).hex

    def __str__(self):
        return "Job UUID: {}; Root Event UUID: {}; Current Event UUID: {}".format(self._uuid,
                                                                                  self._root_event_node.event_uuid,
                                                                                  self._current_node.event_uuid)

    def __eq__(self, other):
        return self._uuid == other.uuid

    def __ne__(self, other):
        return not self == other

    @property
    def root_event(self):
        return self._root_event_node.event

    @property
    def current_event(self):
        return self._current_node.event

    @property
    def created_at(self):
        return self._created_at

    @property
    def uuid(self):
        return self._uuid

    @property
    def is_finished(self):
        return self._current_node is None

    def update_current_node(self, result_event, merge_payload=False):
        """
        Updates the current node property and returns the updated node's event.
        If the current node is a leaf node, returns None
        :param result_event: the result_event corresponding to the current node
        :param merge_payload: if set to True, merges the bundle of the current bundle with the
            next node's event's bundle
        """
        # Check if event is a proper event (ResultEvent)
        if not isinstance(result_event, ResultEvent):
            raise AttributeError('AeroCubeJob.update_and_retrieve_next_event: ERROR: resolve_event requires a ResultEvent')
        # Check if ResultEvent is for the current calling event
        if self.current_event.uuid != result_event.payload.strings(ResultEvent.CALLING_EVENT_UUID):
            raise AttributeError('AeroCubeJob.update_and_retrieve_next_event: ERROR: result event with CALLING_EVENT_UUID:{} received not for current calling event:{}'.format(result_event.payload.strings(ResultEvent.CALLING_EVENT_UUID), self.current_event.uuid))
        # Merge bundle if param is set to True before moving to next node
        replaced_node = self._current_node
        self._current_node = self._current_node.next_event_node(result_event)
        if merge_payload is True:
            self._current_node.event.merge_payload(replaced_node.event.payload)

    # Constructors -- use to construct specific type of AeroCubeJobs

    @staticmethod
    def create_image_upload_job(starting_bundle):
        # TODO: add error event nodes
        """
        Sequence of events
        1. ImageEvent - identify AeroCubes
        2. StorageEvent - store internally
        3. StorageEvent - store externally
        :param starting_bundle:
        :return:
        """
        # Create events
        ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES)
        StorageEvent(StorageEventSignal.STORE_INTERNALLY)
        StorageEvent(StorageEventSignal.STORE_EXTERNALLY)
        # Wrap events around event nodes
        # Create and return Job object
        pass
