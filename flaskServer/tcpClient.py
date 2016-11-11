import socket 
import pickle 

class TcpClient:
	def __init__(self, ip, port, bufferSize):
		self.TCP_IP = ip 
		self.TCP_PORT = port
		self.BUFFER_SIZE = bufferSize
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def connect_to_controller(self):
		self.s.connect((self.TCP_IP, self.TCP_PORT))

	def send_to_controller(self,data):
		x = pickle.dumps(data)
		self.s.send(x)

	# only tested on receiving strings, not json
	def receive_from_controller(self):
		incoming_data = self.s.recv(self.BUFFER_SIZE)
		return incoming_data.decode()

	def close(self):
		self.s.close()