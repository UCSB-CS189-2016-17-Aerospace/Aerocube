import socket 
import pickle 


TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect_to_controller():
	s.connect((TCP_IP, TCP_PORT))

def close_connection():
	s.close()

def send_to_controller(data):
	x = pickle.dumps(data)
	s.send(x)

def receive_from_controller():
	incoming_data = s.recv(BUFFER_SIZE)
	#this last line currently holds the assumption that it will receive a string back, not tested on Json object being returned 
	return incoming_data.decode()
