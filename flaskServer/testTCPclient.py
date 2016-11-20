import unittest 
from tcpClient import *
import socket
from testClass import TestClass
import random 

class TcpClientTestCase(unittest.TestCase):
	def setUp(self):
		self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.portNumber = random.randrange(5005,6000,1)
		self.serv.bind(('127.0.0.1',self.portNumber))
		self.serv.listen(1)
		self.client = TcpClient('127.0.0.1',self.portNumber,1024)
		self.client.connect_to_controller()
		self.conn = self.serv.accept()[0]

	def tearDown(self):
		self.conn.close()
		self.client.close()
		


	#sometimes gives weird resource warning of an unclosed socket, but tearDown states otherwise. (Does work though.)
	def test_sending_message(self):
		RAW_MESSAGE = TestClass('John','pizza man')
		self.client.send_to_controller(RAW_MESSAGE)
		data = pickle.loads(self.conn.recv(1024))
		self.assertEqual(data.name, RAW_MESSAGE.name)
		self.client.close()
		self.conn.close()
		

	#sometimes gives weird resource warning of an unclosed socket, but tearDown states otherwise. (Does work though.)
	def test_receiving_message(self):
		RAW_MESSAGE = TestClass('John', 'Pizza man')
		encoded_message = pickle.dumps(RAW_MESSAGE)
		self.conn.send(encoded_message)
		received_message = self.client.receive_from_controller()
		self.client.close()
		self.conn.close()
		self.assertEqual(received_message.name, RAW_MESSAGE.name)
	

if __name__ == '__main__':
	unittest.main()