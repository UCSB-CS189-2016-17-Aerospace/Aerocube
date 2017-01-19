from .aeroCubeEvent import *


class AeroCubeJobEventNode:
    """
    :ivar _event
    :ivar _event_signal_map
    """

    def __init__(self, event, event_signal_map=None):
        if not isinstance(event, AeroCubeEvent):
            raise AttributeError('Invalid event parameter, must be instance of AeroCubeEvent')
        self._event = event
        self._event_signal_map = event_signal_map

    @property
    def is_leaf_node(self):
        return self._event_signal_map is None

    def traverse_job(self, result_event):
        if not isinstance(result_event, ResultEvent):
            raise TypeError('Can only traverse jobs with a result_event')
        try:
            return self._event_signal_map[result_event.signal]
        except KeyError as err:
            raise LookupError('Cannot process result event with signal {} for current event with signal {}'
                              .format(result_event.signal, self._event.signal))


class AeroCubeJob:
    """
    :ivar _root_event:
    :ivar _created_at:
    :ivar _uuid:
    """

    def __init__(self, event_node, created_at=time.time()):
        if not isinstance(event_node, AeroCubeJobEventNode):
            raise AttributeError('Invalid event node parameter, must be instance of AeroCubeJobEventNode')
        self._root_event = event_node
        self._current_node = event_node
        self._created_at = created_at
        self._uuid = id if id is not None else \
            uuid.uuid5(uuid.NAMESPACE_OID, "{}-{}".format(self.__class__.__name__, self._created_at)).hex

    def __str__(self):
        pass

    @property
    def created_at(self):
        return self._created_at

    @property
    def root_event(self):
        return self._root_event

    @property
    def uuid(self):
        return self._uuid

    def traverse_jobs(self, result_event):
        if self._current_node.is_leaf_node:
            return None
        else:
            self._current_node = self._current_node.traverse_job(result_event)

