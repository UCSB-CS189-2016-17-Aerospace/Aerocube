import dill
from controller.tcpServer import TcpServer
# import packages from Aerocube directory
from eventClass.aeroCubeSignal import *
from eventClass.aeroCubeEvent import *
from externalComm.externalComm import *
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
        return
        # result_event_status = ResultEvent(result_signal=status)
        # print('Controller: Sending ResultEvent: {}'.format(result_event_status))
        # self.server.send_response(result_event_status)

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
            external_write(database=database, scanID=scan_id, data=data)
            print('Controller: Storing image externally')
            external_store_img( database=database, scanID=scan_id, data=img_path)
            print('Controller: Successfully stored externally, sending ResultEvent')
            self.return_status(ResultEventSignal.EXT_COMM_OP_OK)
        except ValueError:
            print('Controller: External storage failed')
            self.return_status(ResultEventSignal.EXT_COMM_OP_FAILED)

    def initiate_scan(self, scan_id, payload):
        print('Controller: Initiate Scan')
        print(payload)
        file_path = payload.strings('FILE_PATH')
        print('Controller: Payload FILE_PATH is {}'.format(file_path))
        # logging.info("scan {} initiated".format(scan_id))
        results = self.scan_image(file_path=file_path)
        print('Controller: Scanning results received')
        print(results)
        # logging.info("scan {} image complete".format(scan_id))
        # payload.strings('FILE_PATH') should be the path to the image
        self.store_locally(path=str(scan_id), data=results)
        # logging.info("{} stored locally".format(scan_id))
        serializable_results = (list(map((lambda c: c.tolist()), results[0])),
                                results[1].tolist())
        self.store_data_externally(database=payload.strings('EXT_STORAGE_TARGET'),
                                   scan_id=scan_id,
                                   data=serializable_results,
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
            print('Received Event: \r\n')
            print(event)
            print(event.payload)
            if event.signal == ImageEventSignal.IDENTIFY_AEROCUBES:
                self.initiate_scan(scan_id=event.created_at, payload=event.payload)
            else:
                pass


if __name__ == '__main__':
    controller = Controller()
    print("ysysysysysys")
    # controller.run()
    # Create event to mock event coming in
    bundle = Bundle()
    filepath = "/home/ubuntu/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg"
    bundle.insert_string('FILE_PATH', filepath)
    bundle.insert_string('EXT_STORAGE_TARGET', 'FIREBASE')
    event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES, bundle)
    controller.initiate_scan(scan_id=event.created_at, payload=event.payload)
