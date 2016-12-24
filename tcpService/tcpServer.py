import socket
from eventClass.aeroCubeEvent import AeroCubeEvent
from .tcpUtils import TcpUtil


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
        print('TcpServer.accept_connection: Connection accepted')

    def send_response(self, response_string):
        encoded_response = TcpUtil.encode_string(response_string)
        # print('TcpServer.send_response: Sending message: \r\n{}\r\n'.format(encoded_response))
        try:
            self.conn.send(encoded_response)
            print('TcpServer.send_response: Response sent')
        except socket.error as e:
            print('TcpServer.send_response: Cant send response to client: %s' % e)

    def receive_data(self):
        encoded_message = self.conn.recv(self.BUFFER_SIZE)
        decoded_string = TcpUtil.decode_string(encoded_message)
        # print('TcpServer.receive_data: Received message: \r\n{}\r\n'.format(decoded_string))
        return decoded_string

    def close_connection(self):
        self.conn.close()
