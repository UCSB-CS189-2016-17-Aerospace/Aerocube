#Controller Documentation 

##Controller.py 
####For anyone working on the controller.py file here are some tips:
To run this file, you first run controller.py and then run flaskserver/mockTcpEventHandler.py. The mockTcpEventHandler.py is essentially a client that will mock the event handler signals. 

At the moment this setup just sends a message from client to server and the server prints it and sends it back to the client to print it. After that the controller.py file stays on, and the client closes. The actual event handler client won't close (just for demonstration at the moment).

The reason that it checks if the data != False is because in the server.receive_data() function, I set an empty message to False (Just a simple way to check if we actually have a message). 



##TCP Server Usage 
In order to run the tests in testTCPserver.py you have to first run that file and then run mockClient.py simultaneously.
Test feedback should come out in the terminal where you run the testTCPserver.py file. 


