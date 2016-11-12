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
		message = pickle.dumps(data)
		self.s.send(message)

	def receive_from_controller(self):
		incoming_data = self.s.recv(self.BUFFER_SIZE)
		return pickle.loads(incoming_data)

	def close(self):
		self.s.close()


