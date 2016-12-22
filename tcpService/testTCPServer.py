import unittest
from .tcpServer import TcpServer
import socket
from .testClassTcpServer import TestClassTcpServer

# in order to run tests, run this file and then run mockClient.py


class TcpServerTestCase(unittest.TestCase):
	def setUp(self):
		self.server = TcpServer('127.0.0.1', 5005, 1024)
		self.server.accept_connection()

	def tearDown(self):
		self.server.close_connection()

	# sometimes gives resourcewarning: unclosed socket, but tearDown func does call close_connection()
	# tests do work however
	def test_send_and_receive_message(self):
		RAW_MESSAGE = TestClassTcpServer('Tom', 'pizza man')
		correct_response = (RAW_MESSAGE.name, RAW_MESSAGE.job)
		self.server.send_response(RAW_MESSAGE)
		received = self.server.receive_data()
		received_response = (received.name, received.job)
		self.assertEqual(received_response, correct_response)


if __name__ == '__main__':
	unittest.main()
