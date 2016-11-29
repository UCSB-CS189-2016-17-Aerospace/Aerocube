from controller.tcpServer import TcpServer
# import packages from Aerocube directory
from eventClass.aeroCubeSignal import *
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
        print('Controller: Sending ResultEvent: {}'.format(result_event_status))
        self.server.send_response(result_event_status)

    def scan_image(self, file_path):
        try:
            print('Controller: Instantiating ImP')
            imp = ImageProcessor(file_path)
            print('Controller: Finding fiducial markers')
            (corners, marker_ids) = imp._find_fiducial_markers()
            print('Controller: Results Received, sending ResultEvent')
            self.return_status(ResultEventSignal.IMP_OPERATION_OK)
            return corners, marker_ids
        except:
            print('Controller: ImP Failed')
            self.return_status(ResultEventSignal.IMP_OP_FAILED)

    def store_locally(self, path, data):
        print('Controller: Storing data locally')
        self.return_status(store(location=path, pickleable=data))

    def store_data_externally(self, database, scan_id, data, img_path):
        try:
            print('Controller: Storing data externally')
            process(func='-w', database=database, scanID=scan_id, data=data)
            print('Controller: Storing image externally')
            process(func='-iw', database=database, scanID=scan_id, data=img_path)
            print('Controller: Successfully stored externally, sending ResultEvent')
            self.return_status(ResultEventSignal.EXT_COMM_OP_OK)
        except ValueError:
            print('Controller: External storage failed')
            self.return_status(ResultEventSignal.EXT_COMM_OP_FAILED)

    def initiate_scan(self, scan_id, payload):
        print('Controller: Initiate Scan')
        file_path = payload.strings('FILE_PATH')
        print('Controller: Payload FILE_PATH is {}'.format(file_path))
        # logging.info("scan {} initiated".format(scan_id))
        results = self.scan_image(file_path=file_path)
        print('Controller: Scanning results received')
        print(results)
        # logging.info("scan {} image complete".format(scan_id))
        # payload.strings('FILE_PATH') should be the path to the image
        self.store_locally(path=scan_id, data=results)
        # logging.info("{} stored locally".format(scan_id))

        self.store_data_externally(database=payload.strings('EXT_STORAGE_TARGET'),
                                   scan_id=scan_id,
                                   data=results,
                                   img_path=file_path)
        # payload.strings('EXT_STORAGE_TARGET') should be the database
        # logging.info("{} stored on firebase".format(scan_id))
        self.return_status(ResultEventSignal.IDENT_AEROCUBES_FIN)
        # logging.info(str(scan_id)+ "complete")

    def run(self):
        self.server.accept_connection()
        print('Controller: Connection accepted')
        while 1:
            event = self.server.receive_data()
            if event.signal == ImageEventSignal.IDENTIFY_AEROCUBES:
                self.initiate_scan(scan_id=event.created_at, payload=event.payload)
            else:
                pass


if __name__ == '__main__':
    controller = Controller()
    print("ysysysysysys")
    controller.run()
