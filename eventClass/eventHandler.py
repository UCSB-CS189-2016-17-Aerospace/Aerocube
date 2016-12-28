from collections import deque
from enum import Enum
from .aeroCubeEvent import AeroCubeEvent, ResultEvent
from .aeroCubeSignal import *


class EventHandler(object):
    """
    Handles incoming events in a queue, providing methods to peek at events and
    properly resolve/dequeue them.
    Raises a TypeError if an object that is not an AeroCubeEvent is Attempted
    to be enqueued.
    For all other operations, EventHandler will return None if an operation is
    improperly used (e.g., if dequeue_event is called on an EventHandler with
    no events).
    :ivar _event_deque: queue-style structure that stores incoming events
    :ivar _on_start_event: function handler for "on_start_event"
    :ivar _on_enqueue: function handler for "on_enqueue"
    :ivar _on_dequeue: function handler for "on_dequeue"
    :ivar _state: state chosen from inner class State that controls how incoming events are dealt with
    """

    # TODO: Structure for Event Priorities

    class NotAllowedInStateException(Exception):
        """
        NotAllowedInStateException is thrown when a function is called that is not permitted in the current EventHandler
        state
        """
        def __init__(self, message):
            super(EventHandler.NotAllowedInStateException, self).__init__(message)

    class State(Enum):
        # Ready to receive & start events
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
        self._state = EventHandler.State.STARTED

    # setters for function handlers

    def set_start_event_observer(self, observer):
        """
        Set a function to be called when start_event is run
        :param observer: a function with parameter event, the event that is started
        """
        self._on_start_event = observer
        print('EventHandler: Set on_start_event')

    def set_enqueue_observer(self, observer):
        """
        Set a function to be called when an event is enqueued
        :param observer: a function with parameter event, the event that is enqueued
        """
        self._on_enqueue = observer
        print('EventHandler: Set on_enqueue')

    def set_dequeue_observer(self, observer):
        """
        Set a function to be called when dequeue_event is run
        :param observer: a function with parameter event, the event that is dequeued
        """
        self._on_dequeue = observer
        print('EventHandler: Set on_dequeue')

    # getter functions or functions to allow observation of the internal event deque

    def enqueue_event(self, event):
        """
        Adds a new event
        :param event: the new event to be added
        :return:
        """
        print('EventHandler: Enqueued event: \r\n{}\r\n'.format(event))
        if EventHandler.is_valid_element(event):
            self._event_deque.append(event)
        else:
            raise TypeError("Attempted to queue invalid object to EventHandler")
        # Try to restart the sending process on enqueue
        try:
            self._start_sending_events()
        except EventHandler.NotAllowedInStateException as e:
            print(e)

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

    # state-change functions

    def restart(self):
        """
        Attempts to put the EventHandler in a STARTED state from STOPPED
        Precondition: State is STOPPED
        :return: True if successful, False if not
        """
        if self._state == EventHandler.State.STOPPED:
            self._state = EventHandler.State.STARTED
            self._start_sending_events()
            return True
        return False

    def _start_sending_events(self):
        """
        Attempts to send the first event in the queue
        Precondition: State is STARTED
        """
        if self._state != EventHandler.State.STARTED:
            raise EventHandler.NotAllowedInStateException('ERROR: EventHandler must be in STARTED state to send events')
        if self.any_events():
            self._start_event()

    def _continue_sending_events(self):
        """
        Attempts
        Precondition: State is PENDING
        """
        if self._state != EventHandler.State.PENDING:
            raise EventHandler.NotAllowedInStateException('ERROR: EventHandler must be in PENDING state to continue sending events')
        self._state = EventHandler.State.STARTED
        print('EventHandler._continue_sending_events: State changed to {}'.format(self._state))
        self._start_sending_events()

    def _resolve_state(self):
        """
        Precondition: State is PENDING or PENDING_STOP_ON_RESOLVE
        """
        if self._state == EventHandler.State.PENDING:
            self._continue_sending_events()
        elif self._state == EventHandler.State.PENDING_STOP_ON_RESOLVE:
            self._state = EventHandler.State.STOPPED
            print('EventHandler._resolve_state: State changed to {}'.format(self._state))
        else:
            raise EventHandler.NotAllowedInStateException('ERROR: EventHandler must be in PENDING or PENDING_STOP_ON_RESOLVE to resolve current event')

    def stop(self):
        """
        Attempts to put the EventHandler in a STOPPED state, waiting until the currently pending event is resolved
        :return: True if successfully directs the EventHandler to switch to a STOPPED state, either then or
        after the current event is resolved. False otherwise.
        """
        if self._state == EventHandler.State.PENDING:
            self._state = EventHandler.State.PENDING_STOP_ON_RESOLVE
            print('EventHandler.stop: State changed to {}'.format(self._state))
        elif self._state == EventHandler.State.STARTED:
            self._state = EventHandler.State.STOPPED
            print('EventHandler.stop: State changed to {}'.format(self._state))
        else:
            return False
        return True

    def force_stop(self):
        """
        Forces the EventHandler into a STOPPED state. Note, calling resolve_event will trigger an error while the
        EventHandler is in a STOPPED state. If an attempt to resolve the event is dropped, upon re-starting
        :return:
        """
        print('EventHandler: Force Stop Triggered')
        self._state = EventHandler.State.STOPPED
        print('EventHandler.force_stop: State changed to {}'.format(self._state))

    def resolve_event(self, event):
        """
        Logic to resolve "finished" events/jobs in the EventHandler deque could
        either be handled within this class, or lie in the calling class.
        Needs to be determined.
        :return: if event is resolved/finished, return true; else, return false
        """
        if self._should_event_resolve(event):
            # Modify EventHandler queue (assuming state is valid)
            resolved_event = self._dequeue_event()
            print('EventHandler.resolve_event: Resolved Event: \r\n{}\r\n'.format(resolved_event))
            # Update state
            self._resolve_state()
            print('EventHandler.resolve_event: State changed to {}'.format(self._state))
            return True
        else:
            # Do not modify EventHandler state or queue, but log ResultEvent
            print('EventHandler.resolve_event: ResultEvent (not resolved): \r\n{}\r\n'.format(event))
            # Return False to indicate calling_event not finished
            return False

    def _should_event_resolve(self, event):
        # Check if event is a proper event (ResultEvent)
        if not isinstance(event, ResultEvent):
            raise AttributeError('EventHandler._should_event_resolve: ERROR: resolve_event requires a ResultEvent')
        # Check if state is valid
        if self._state == EventHandler.State.STARTED:
            # Raise Error
            raise EventHandler.NotAllowedInStateException('ERROR: Attempted to resolve event while not pending for a result')
        elif self._state == EventHandler.State.STOPPED:
            # Raise Error
            raise EventHandler.NotAllowedInStateException('ERROR: Attempted to resolve event while stopped')
        # Check if ResultEvent is for the current calling event
        if self._peek_current_event().uuid != event.payload.strings(ResultEvent.CALLING_EVENT_UUID):
            raise AttributeError('EventHandler._should_event_resolve: ERROR: result event with CALLING_EVENT_UUID:{} received not for current calling event:{}',
                                 event.payload.strings(ResultEvent.CALLING_EVENT_UUID), self._peek_current_event().uuid)
        return event.signal == ResultEventSignal.IDENT_AEROCUBES_FIN

    def _start_event(self):
        """
        Start an event by calling the on_start_event function and passing it the first event in the queue.
        This action puts the EventHandler in a PENDING state.
        Preconditions: state must be STARTED; on_start_event must be not None
        :raises NotImplementedError if on_start_event is not defined
        """
        if self._state != EventHandler.State.STARTED:
            raise EventHandler.NotAllowedInStateException('ERROR: Attempted to start event while not in STARTED state')
        if self._on_start_event is not None:
            print('EventHandler._start_event: Starting event: \r\n{}\r\n'.format(self._peek_current_event()))
            self._state = EventHandler.State.PENDING
            self._on_start_event(self._peek_current_event())
            print('EventHandler._start_event: State changed to {}'.format(self._state))
        else:
            raise NotImplementedError('ERROR: Must call set_start_event_observer before an event can be sent')

    # static helper function(s)

    @staticmethod
    def is_valid_element(obj):
        """
        Check if the obj is an instance of AeroCubeEvent
        :param obj:
        :return:
        """
        return isinstance(obj, AeroCubeEvent)
