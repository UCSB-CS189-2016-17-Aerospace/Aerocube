# AeroCube
## Brief
This repository more-or-less contains all functionality intended to be run on the Jetson TX1 board, and forms the backbone of our application.
## Guide
A breakdown of the folders found at the root-level directory of the repository follows:
* **ImP** - image processing module responsible for scanning an image and returning information about detected AeroCubes; also contains the data structures that represent an AeroCube and its members; contains a module for camera calibration
* **controller** - stateless Controller that, upon receiving events through a TCP connection, calls the appropriate function dependent on the event's signal, returning the results of the function call to the TCP client
* **dataStorage** - handles internal storage on the Jetson (e.g., saving scan information, scanned images)
* **externalComm** - interface to handle external communication, particularly to the Firebase database
* **flaskServer** - more appropriately named the Job Handler; organizes incoming Jobs that may result from requests to the Flask Server or through a listener on Firebase; processes Jobs by sending events sequentially to the Controller to resolve each event
* **ipcProto** - early prototype to test inter-process communication (between Controller and Flask Server)
* **jobs** - module defining a Job, or a sequence of events that represent a task to be completed (e.g., an ImageUpload job, which consists of scanning an image with ImP and storing it internally and externally); also defines the AeroCubeEvent class, the instance passed between the Controller and Flask Server, and the AeroCubeSignals that dictate what action the controller takes for a given event
* **shellScripts** - collection of convenient scripts (e.g., for starting up different parts of the application)
* **systemTests** - collection of system tests, such as the main use case
* **tcpService** - module collecting TCP logic and implementation into one location, allowing Controller and Flask Server to call it concisely


## Reference
For additional information, please see our Wiki page at: <https://github.com/UCSB-CS189-2016-17-Aerospace/Aerocube/wiki>
