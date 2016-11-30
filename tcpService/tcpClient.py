import socket
import json
from eventClass.aeroCubeEvent import AeroCubeEvent
from .tcpUtils import TcpUtil


class TcpClient:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_controller(self):
        try:
            self.s.connect((self.TCP_IP, self.TCP_PORT))
            print('TcpClient.connect_to_controller: Connected')
        except socket.error as e:
            print('TcpClient.connect_to_controller: Cant connect to TCP server: %s' % e)

    def send_to_controller(self, data):
        encoded_message = TcpUtil.encode_string(data)
        try:
            bytes_sent = self.s.send(encoded_message)
            print('TcpClient.send_to_controller: {} bytes sent data to controller'.format(bytes_sent))
        except socket.error as e:
            print('TcpClient.send_to_controller: Cant send message to TCP server: %s' % e)

    def receive_from_controller(self):
        encoded_message = self.s.recv(self.BUFFER_SIZE)
        decoded_message = TcpUtil.decode_string(encoded_message)
        print('TcpClient.receive_from_controller: Received response: \r\n{}\r\n'.format(decoded_message))
        return decoded_message

    def close(self):
        self.s.close()
