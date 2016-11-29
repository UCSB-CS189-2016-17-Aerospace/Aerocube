from controller.tcpServer import TcpServer
# import packages from Aerocube directory
from eventClass.aeroCubeSignal import AeroCubeSignal
from eventClass.aeroCubeEvent import *
from externalComm.externalComm import process
from dataStorage.dataStorage import store
# import packages from Aerocube-ImP directory
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
        try:
            imp = ImageProcessor(file_path)
            (corners, marker_ids, _) = imp._find_fiducial_markers()
            self.return_status(AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)
            return corners, marker_ids
        except:
            self.return_status(AeroCubeSignal.ResultEventSignal.IMP_OP_FAILED)

    def store_locally(self, path, data):
        self.return_status(store(location=path, pickleable=data))

    def store_data_externally(self, database, scan_id, data, img_path):
        try:
            process(func='-w', database=database, scanID=scan_id, data=data)
            process(func='-iw', database=database, scanID=scan_id, data=img_path)
            self.return_status(AeroCubeSignal.ResultEventSignal.EXT_COMM_OP_OK)
        except ValueError:
            self.return_status(AeroCubeSignal.ResultEventSignal.EXT_COMM_OP_FAILED)

    def initiate_scan(self, scan_id, payload):
        logging.info("scan "+scan_id+ " initiated")
        results = self.scan_image(payload.string(0))
        logging.info("scan "+scan_id+" image complete")
        # payload.string(0) should be the path to the image
        self.store_locally(path=scan_id, data=results)
        logging.info(str(scan_id)+" stored locally")
        self.store_data_externally(database=payload.string(1), ID=scan_id, data=results, img_path=payload.string(0))
        # payload.string(1) should be the database
        logging.info(str(scan_id) + " stored on firebase")
        self.return_status(AeroCubeSignal.ResultEventSignal.IDENT_AEROCUBES_FIN)
        logging.info(str(scan_id)+ "complete")
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
    print("ysysysysysys")
    controller.run()
