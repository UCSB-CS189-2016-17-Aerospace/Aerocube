import socket

from logger import Logger
from .tcpUtils import TcpUtil

logger = Logger('tcpServer.py', active=True, external=False)


class TcpServer:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.conn = None
        self.addr = None
        self.s.bind((self.TCP_IP, self.TCP_PORT))

    def accept_connection(self):
        self.s.listen(1)
        self.conn, self.addr = self.s.accept()
        logger.debug(
            self.__class__.__name__,
            'accept_connection',
            msg='Connection accepted',
            id=None)

    def send_response(self, response_string):
        encoded_response = TcpUtil.encode_string(response_string)
        # print('TcpServer.send_response: Sending message: \r\n{}\r\n'.format(encoded_response))
        logger.debug(
            self.__class__.__name__,
            'send_response',
            msg='Sending message: \r\n{}\r\n'.format(encoded_response),
            id=None)
        try:
            self.conn.send(encoded_response)
            logger.debug(
                self.__class__.__name__,
                'send_response',
                msg='Response sent',
                id=None)
        except socket.error as e:
            logger.err(
                self.__class__.__name__,
                'send_response',
                msg='Can\'t send response to client {}'.format(e),
                id=None)

    def receive_data(self):
        encoded_message = self.conn.recv(self.BUFFER_SIZE)
        decoded_string = TcpUtil.decode_string(encoded_message)
        # print('TcpServer.receive_data: Received message: \r\n{}\r\n'.format(decoded_string))
        return decoded_string

    def close_connection(self):
        self.conn.close()
