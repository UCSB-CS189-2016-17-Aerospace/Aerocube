import socket

from logger import Logger
from .tcpUtils import TcpUtil

logger = Logger('tcpClient.py', active=True, external=False)


class TcpClient:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_controller(self):
        try:
            self.s.connect((self.TCP_IP, self.TCP_PORT))
            logger.success(
                self.__class__.__name__,
                func_name='connect_to_controller',
                msg='Connected',
                id=None)
        except socket.error as e:
            logger.err(
                self.__class__.__name__,
                func_name='connect_to_controller',
                msg='Failed to connect to TCP server: {}'.format(e),
                id=None)

    def send_to_controller(self, data):
        encoded_message = TcpUtil.encode_string(data)
        try:
            bytes_sent = self.s.send(encoded_message)
            logger.success(
                self.__class__.__name__,
                func_name='send_to_controller',
                msg='{} bytes sent data to controller'.format(bytes_sent),
                id=None)
        except socket.error as e:
            logger.err(
                self.__class__.__name__,
                func_name='send_to_controller',
                msg='Failed to send message to TCP Server: {}'.format(e),
                id=None)

    def receive_from_controller(self):
        encoded_message = self.s.recv(self.BUFFER_SIZE)
        decoded_message = TcpUtil.decode_string(encoded_message)
        logger.success(
            self.__class__.__name__,
            func_name='receive_from_controller',
            msg='Received: {}'.format(decoded_message),
            id=None)
        return decoded_message

    def close(self):
        self.s.close()
