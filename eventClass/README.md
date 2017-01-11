# Events Class Module
## Public Classes
### AeroCubeEvent
Abstract class which has derived classes to represent different kinds of events.
These events include:
* **ImageEvent** - event requesting an operation from the Image Processing Module
* **ResultEvent** - event to be returned after a requested operation, which can serve as an acknowledgement of a successful operation or a negative acknowledgement with an error
* **SystemEvent** - represents relevant events from the system, such as power-related events

Every event should be constructed with an appropriate AeroCubeSignal Enum object to represent the type of message sent.

### AeroCubeSignal
Collection of signals to be used when constructing AeroCubeEvents. AeroCubeSignal has inner classes which represent the different type of signals available (e.g., those related to ImageEvents, ResultEvents, or otherwise). The hex values that the different signals are set to should never be called directly, as signal validation is done against Enum instances.

### EventHandler
Class providing a queue in which events can be enqeueued, dequeued, or inspected. Controls the order of incoming events.

### Bundle
Provides functionality to store different data of the following types:
* Numbers
* Strings
* Raw (e.g., bytestream for an image)
* Iterables

Used with AeroCubeEvent to help structure the payload of an event.

### Event Handler States 
// include portion explaining that currently the EH only works w/one event at a time. As well as the purpose of states
* **STARTED** - This state reperesents that the evetn handler is ready to both receive and starrt events.
* **PENDING** - The state means that the event handler is waiting for results on a task that the controller is currently processing.
* **STOPPED** - Simply means that the event handler is paused, therefore cannot accept any new events.
* **PENDING_STOP_ON_RESOLVE** - This state essentially 

