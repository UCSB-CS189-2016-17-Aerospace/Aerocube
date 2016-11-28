# Internal Storage Module
## Usage
```
store(location, pickleable, use_relative_location=True):
    Public abstraction of storage
    :param location: string path
    :param pickleable: data that can be pickled
    :param use_relative_location: bool
    :return: An event containing a signal that describes success or failure
```

```
retrieve(location):
    Public abstraction of retrieval
    :param location: relative location
    :return:
```
