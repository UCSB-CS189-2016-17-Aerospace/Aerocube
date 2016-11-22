import socket 
# in order to run tests, run the testTCPserver.py file and then run this file 

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1',5005))
data = client.recv(1024)
client.send(data)
client.close()
