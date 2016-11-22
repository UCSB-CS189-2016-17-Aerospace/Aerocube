import socket
import pickle


class TcpServer:
	def __init__(self,ip,port,bufferSize):
		self.TCP_IP = ip
		self.TCP_PORT = port
		self.BUFFER_SIZE = bufferSize
		self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.s.bind((self.TCP_IP,self.TCP_PORT))
		

	def accept_connection(self):
		self.s.listen(1)
		global conn, addr
		conn, addr = self.s.accept()

	def send_response(self, response):
		encoded_response = pickle.dumps(response)
		try:
			conn.send(encoded_response)	
		except socket.error as e:
			print('Cant send response to client: %s' % e)
		

	def receive_data(self):
		data = conn.recv(self.BUFFER_SIZE)
		if not data:
			message = False
		else:
			message = pickle.loads(data)
		return message

	def close_connection(self):
		conn.close()


