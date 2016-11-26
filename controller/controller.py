from controller.tcpServer import TcpServer
from eventClass.aeroCubeSignal import AeroCubeSignal
from externalComm.externalComm import process
from dataStorage.dataStorage import store
from eventClass.aeroCubeEvent import *
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor


class Controller:
    def __init__(self):
        self.server = TcpServer('127.0.0.1', 5005, 1024)

    def return_status(self, status):
        """
        returns status back to event handler
        :param status: status signal
        :return: void
        """
        result_event_status = ResultEvent(result_signal=status)
        self.server.send_response(result_event_status)

    def scan_image(self, file_path):
        imp = ImageProcessor(file_path)
        return imp._find_fiducial_markers()  # assuming this method returns the vectors and corners

    def store_locally(self, path, data):
        self.return_status(store(location=path, pickleable=data))

    def store_data_externally(self, database, scan_id, data, img_path):
        self.return_status(process(func='-w', database=database, scanID=scan_id, data=data))
        self.return_status(process(func='-iw', database=database, scanID=scan_id, data=img_path))

    def initiate_scan(self, scan_id, payload):
        results = self.scan_image(payload.string(0))
        # payload.string(0) should be the path to the image
        self.store_locally(path=scan_id, data=results)
        self.store_data_externally(database=payload.string(1), ID=scan_id, data=results, img_path=payload.string(0))
        # payload.string(1) should be the database
        self.return_status()

    def run(self):
        self.server.accept_connection()
        while 1:
            data = self.server.receive_data()
            if data != False:
                if data.signal() == AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES:
                    self.initiate_scan(scan_ID=data.created_at, payload=data.payload())
                else:
                    pass
                    # IM CONFUSED AF


if __name__ == '__main__':
    controller = Controller()
    controller.run()
