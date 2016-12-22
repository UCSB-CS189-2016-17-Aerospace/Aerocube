class ControllerSettings():
    _ip_addr = '127.0.0.1'
    _port = 5005

    @staticmethod
    def IP_ADDR():
        return ControllerSettings._ip_addr

    @staticmethod
    def PORT():
        return ControllerSettings._port
