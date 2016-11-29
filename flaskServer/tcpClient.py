import socket
import dill
from eventClass.aeroCubeEvent import AeroCubeEvent


class TcpClient:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_controller(self):
        try:
            self.s.connect((self.TCP_IP, self.TCP_PORT))
            print('TcpClient: Connected')
        except socket.error as e:
            print('TcpClient: Cant connect to TCP server: %s' % e)

    def send_to_controller(self, data):
        message = dill.dumps(data)
        test_de_dill = dill.loads(message)
        print('Test re/de dill on tcpClient:\r\n')
        print(test_de_dill)
        re_dill = dill.dumps(test_de_dill)
        try:
            bytes_sent = self.s.send(re_dill)
            print('TcpClient: {} bytes sent data to controller'.format(bytes_sent))
        except socket.error as e:
            print('TcpClient: Cant send message to TCP server: %s' % e)

    def receive_from_controller(self):
        incoming_data = self.s.recv(self.BUFFER_SIZE)
        message = dill.loads(incoming_data)
        if isinstance(message, AeroCubeEvent):
            print('TcpClient: Received message: {}'.format(message))
        else:
            print('TcpClient: Warning: Received message that is not instance of ResultEvent')
        return message

    def close(self):
        self.s.close()
