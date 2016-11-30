import socket
import dill
from eventClass.aeroCubeEvent import AeroCubeEvent
import json


class TcpServer:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None
        self.s.bind((self.TCP_IP, self.TCP_PORT))

    def accept_connection(self):
        self.s.listen(1)
        self.conn, self.addr = self.s.accept()
        print('TcpServer: Connection accepted')

    def send_response(self, response):
        encoded_response = json.dumps(response).encode()
        try:
            self.conn.send(encoded_response)
            print('TcpServer: Response sent')
        except socket.error as e:
            print('Cant send response to client: %s' % e)

    def receive_data(self):
        data = self.conn.recv(self.BUFFER_SIZE)
        message = AeroCubeEvent.construct_from_json(json.loads(data.decode()))
        if isinstance(message, AeroCubeEvent):
            print('TcpServer: Data Received')
            return message
        else:
            raise AttributeError('ERROR: Data must be an Event')

    def close_connection(self):
        self.conn.close()
