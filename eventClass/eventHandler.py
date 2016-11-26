from collections import deque
from aeroCubeEvent import AeroCubeEvent


class EventHandler(object):
    """
    Handles incoming events in a queue, providing methods to peek at events and
    properly resolve/dequeue them.
    Raises a TypeError if an object that is not an AeroCubeEvent is Attempted
    to be enqueued.
    For all other operations, EventHandler will return None if an operation is
    improperly used (e.g., if dequeue_event is called on an EventHandler with
    no events).
    """
    _event_deque = None

    def __init__(self):
        self._event_deque = deque()

    def enqueue_event(self, event):
        """
        Adds a new event
        :param event: the new eevnt to be added
        :return:
        """
        if EventHandler.is_valid_element(event):
            self._event_deque.append(event)
        else:
            raise TypeError("Attempted to queue invalid object to EventHandler")

    def dequeue_event(self):
        """
        Retrieves the next event to be handled
        :return:
        """
        return self._event_deque.popleft()

    def peek_current_event(self):
        """
        Peeks at the current event
        :return: the current event
        """
        if self.any_events():
            return self._event_deque[0]
        else:
            return None

    def peek_last_added_event(self):
        """
        Peeks at the most recently added event
        :return: the most recently added event
        """
        if self.any_events():
            return self._event_deque[-1]
        else:
            return None

    def any_events(self):
        """
        Check if there are any events
        :return: true if there are events
        """
        return len(self._event_deque) > 0

    def resolve_event(self):
        """
        Logic to resolve "finished" events/jobs in the EventHandler deque could
        either be handled within this class, or lie in the calling class.
        Needs to be determined.
        """
        pass

    @staticmethod
    def is_valid_element(obj):
        """
        Check if the obj is an instance of AeroCubeEvent
        :param obj:
        :return:
        """
        return isinstance(obj, AeroCubeEvent)
