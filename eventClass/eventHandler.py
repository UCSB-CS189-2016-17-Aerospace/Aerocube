from collections import deque
from .aeroCubeEvent import AeroCubeEvent, ResultEvent
from enum import Enum


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

    # TODO: Structure for Event Priorities

    class NotAllowedInStateException(Exception):
        """
        NotAllowedInStateException is thrown when a function is called that is not permitted in the current EventHandler
        state
        """
        def __init__(self, message):
            super(EventHandler.NotAllowedInStateException, self).__init__(message)

    class EventHandlerState(Enum):
        # Ready to receive
        STARTED                     = 0x0000aaaa
        # Waiting for results
        PENDING                     = 0x0000bbbb
        # Paused
        STOPPED                     = 0x0000cccc
        # Will Stop After Current Event Resolved
        PENDING_STOP_ON_RESOLVE     = 0x0000dddd

    def __init__(self, start_event_observer=None, enqueue_observer=None, dequeue_observer=None):
        self._event_deque = deque()
        self._on_start_event = start_event_observer
        self._on_enqueue = enqueue_observer
        self._on_dequeue = dequeue_observer
        self._state = self.EventHandlerState.STARTED

    def set_start_event_observer(self, observer):
        """
        Set a function to be called when start_event is run
        :param observer: a function with parameter event, the event that is started
        """
        self._on_start_event = observer

    def set_enqueue_observer(self, observer):
        """
        Set a function to be called when an event is enqueued
        :param observer: a function with parameter event, the event that is enqueued
        """
        self._on_enqueue = observer

    def set_dequeue_observer(self, observer):
        """
        Set a function to be called when dequeue_event is run
        :param observer: a function with parameter event, the event that is dequeued
        """
        self._on_dequeue = observer

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

    def start(self):
        """
        Attempts to put the EventHandler in a STARTED state
        :return: True if successful, False if not
        """
        if self._state == self.EventHandlerState.STOPPED:
            self._state = self.EventHandlerState.STARTED
            return True
        return False

    def restart(self):
        """
        Attempts to restart an EventHandler from a STOPPED state and send the first event in the queue
        Precondition: State changed from STOPPED to STARTED
        """
        if self.start() and self.any_events():
            self._start_event()

    def stop(self):
        """
        Attempts to put the EventHandler in a STOPPED state, waiting until the currently pending event is resolved
        :return: True if successfully directs the EventHandler to switch to a STOPPED state, either then or
        after the current event is resolved. False otherwise.
        """
        if self._state == self.EventHandlerState.PENDING:
            self._state = self.EventHandlerState.PENDING_STOP_ON_RESOLVE
        elif self._state == self.EventHandlerState.STARTED:
            self._state = self.EventHandlerState.STOPPED
        else:
            return False
        return True

    def force_stop(self):
        """
        Forces the EventHandler into a STOPPED state. Note, calling resolve_event will trigger an error while the
        EventHandler is in a STOPPED state. If an attempt to resolve the event is dropped, upon re-starting
        :return:
        """
        self._state = self.EventHandlerState.STOPPED

    def get_state(self):
        """
        Get the state of the eventHandler
        :return: The state of the eventHandler
        """
        return self._state

    def any_events(self):
        """
        Check if there are any events
        :return: true if there are events
        """
        return len(self._event_deque) > 0

    def _dequeue_event(self):
        """
        Retrieves the next event to be handled
        :return:
        """
        return self._event_deque.popleft()

    def _peek_current_event(self):
        """
        Peeks at the current event
        :return: the current event
        """
        if self.any_events():
            return self._event_deque[0]
        else:
            return None

    def _peek_last_added_event(self):
        """
        Peeks at the most recently added event
        :return: the most recently added event
        """
        if self.any_events():
            return self._event_deque[-1]
        else:
            return None

    def resolve_event(self, event):
        """
        Logic to resolve "finished" events/jobs in the EventHandler deque could
        either be handled within this class, or lie in the calling class.
        Needs to be determined.
        """
        if not isinstance(event, ResultEvent):
            raise AttributeError('ERROR: resolve_event requires a ResultEvent')

        # TODO: Check if the ResultEvent corresponds to an Event in the Queue

        # Check State
        if self._state in (self.EventHandlerState.PENDING, self.EventHandlerState.PENDING_STOP_ON_RESOLVE):
            resolved_event = self._dequeue_event()
            # TODO: Further resolution handling
        elif self._state == self.EventHandlerState.STARTED:
            # Raise Error
            raise EventHandler.NotAllowedInStateException('ERROR: Attempted to resolve event while not pending a result')
        elif self._state == self.EventHandlerState.STOPPED:
            # Raise Error
            raise EventHandler.NotAllowedInStateException('ERROR: Attempted to resolve event while stopped')

        # Upon Resolution Update State
        if self._state == self.EventHandlerState.PENDING_STOP_ON_RESOLVE:
            self._state = self.EventHandlerState.STOPPED
        elif self._state == self.EventHandlerState.PENDING:
            self._state = self.EventHandlerState.STARTED

        # Attempt to send next event
        if self._state == self.EventHandlerState.STARTED:
            self._start_event()

    def _start_event(self):
        """
        Start an event by calling the on_start_event function and passing it the first event in the queue.
        This action puts the EventHandler in a PENDING state.
        :raises NotImplementedError if on_start_event is not defined
        """
        if self._on_start_event is not None:
            self._on_start_event(self._peek_current_event())
            self._state = self.EventHandlerState.PENDING
        else:
            raise NotImplementedError('ERROR: Must call set_start_event_observer before an event can be sent')

    @staticmethod
    def is_valid_element(obj):
        """
        Check if the obj is an instance of AeroCubeEvent
        :param obj:
        :return:
        """
        return isinstance(obj, AeroCubeEvent)
