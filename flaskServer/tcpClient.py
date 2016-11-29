import socket
import pickle 


class TcpClient:
    def __init__(self, ip, port, bufferSize):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = bufferSize
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_controller(self):
        try:
            self.s.connect((self.TCP_IP, self.TCP_PORT))
        except socket.error as e:
            print('Cant connect to TCP server: %s' % e)

    def send_to_controller(self,data):
        message = pickle.dumps(data)
        try:
            self.s.send(message)
        except socket.error as e:
            print('Cant send message to TCP server: %s' % e)

    def receive_from_controller(self):
        incoming_data = self.s.recv(self.BUFFER_SIZE)
        message = pickle.loads(incoming_data)
        return message

    def close(self):
        self.s.close()
