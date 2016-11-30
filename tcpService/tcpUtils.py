class TcpUtil(object):
    @staticmethod
    def encode_string(string):
        return string.encode()

    @staticmethod
    def decode_string(string):
        return string.decode()
