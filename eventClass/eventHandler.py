from collections import deque
from aeroCubeEvent import AeroCubeEvent


class EventHandler(object):
    """
    Handles incoming events in a queue, providing methods to peek at events and
    properly resolve/dequeue them.
    """
    _event_deque = None

    def __init__(self):
        self._event_deque = deque()

    def enqueue_event(self, event):
        if EventHandler.is_valid_element(event):
            self._event_deque.append(event)
        else:
            raise TypeError("Attempted to queue invalid object to EventHandler")

    def dequeue_event(self):
        return self._event_deque.popleft()

    def peek_current_event(self):
        if self.any_events():
            return self._event_deque[0]
        else:
            return None

    def peek_last_added_event(self):
        if self.any_events():
            return self._event_deque[-1]
        else:
            return None

    def any_events(self):
        return len(self._event_deque) > 0

    def resolve_event(self):
        """
        Logic to resolve "finished" events/jobs in the EventHandler deque could
        either be handled within this class, or lie in the calling class.
        Needs to be determined.
        """
        pass

    @staticmethod
    def is_valid_element(object):
        return isinstance(object, AeroCubeEvent)
