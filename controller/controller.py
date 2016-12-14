from eventClass.bundle import Bundle
from tcpService.tcpServer import TcpServer
# import packages from Aerocube directory
from eventClass.aeroCubeSignal import ImageEventSignal, ResultEventSignal, SystemEventSignal
from eventClass.aeroCubeEvent import AeroCubeEvent, ImageEvent, ResultEvent
from externalComm.externalComm import process
from dataStorage.dataStorage import store, store_image
# import packages from Aerocube-ImP directory
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor


class Controller:
    def __init__(self):
        self.server = TcpServer('127.0.0.1', 5005, 1024)
        self.calling_event = None

    def return_status(self, status):
        """
        returns status back to event handler
        :param status: status signal
        :return: void
        """
        # return
        print('Controller.return_status: Status is {}'.format(status))
        result_event_status = ResultEvent(result_signal=status, calling_event=self.calling_event.uuid)
        print('Controller.return_status: Sending ResultEvent: \r\n{}\r\n'.format(result_event_status))
        self.server.send_response(str(result_event_status))

    def scan_image(self, file_path):
        try:
            print('Controller.scan_image: Instantiating ImP at {}'.format(file_path))
            imp = ImageProcessor(file_path)
            print('Controller.scan_image: Finding fiducial markers')
            (corners, marker_ids) = imp._find_fiducial_markers()
            # store_image('test_output.png', imp.draw_fiducial_markers(corners, marker_ids))
            return corners, marker_ids
        except:
            print('Controller.scan_image: ImP Failed')
            self.return_status(ResultEventSignal.IMP_OP_FAILED)

    def store_locally(self, path, data):
        print('Controller.store_locally: Storing data locally')
        store(location=path, pickleable=data)
        # self.return_status(store(location=path, pickleable=data))

    def store_data_externally(self, database, scan_id, data, img_path):
        try:
            print('Controller.store_data_externally: Storing data externally')
            process(func='-w', database=database, location='scans', scanID=scan_id, data=data, testing=True)
            print('Controller.store_data_externally: Storing image externally')
            process(func='-iw', database=database, location='scans', scanID=scan_id, data=img_path, testing=True)
            print('Controller.store_data_externally: Successfully stored externally, sending ResultEvent')
            self.return_status(ResultEventSignal.EXT_COMM_OP_OK)
        except ValueError:
            print('Controller.store_data_externally: External storage failed')
            self.return_status(ResultEventSignal.EXT_COMM_OP_FAILED)

    def initiate_scan(self, scan_id, payload):
        print('Controller.initiate_scan: Initiate Scan')
        # print(payload)
        file_path = payload.strings('FILE_PATH')
        print('Controller.initiate_scan: Payload FILE_PATH is {}'.format(file_path))
        results = self.scan_image(file_path=file_path)
        print('Controller.initiate_scan: Results Received, sending ResultEvent')
        self.return_status(ResultEventSignal.IMP_OPERATION_OK)
        print('Controller.initiate_scan: Scanning results received')
        print('Controller.initiate_scan: \r\n{}\r\n'.format(results))
        # payload.strings('FILE_PATH') should be the path to the image
        self.store_locally(path=str(scan_id), data=results)
        serializable_results = (list(map((lambda c: c.tolist()), results[0])),
                                results[1].tolist())
        output_path = "/home/ubuntu/GitHub/Aerocube/ImP/output_files/scan_with_markers.jpg"
        self.draw_and_store_image_with_detected_markers(database=payload.strings('EXT_STORAGE_TARGET'),
                                                        scan_id=str(scan_id).split('.')[0],
                                                        img_path=file_path,
                                                        output_path=output_path,
                                                        scan_results=results)
        self.store_data_externally(database=payload.strings('EXT_STORAGE_TARGET'),
                                   scan_id=str(scan_id).split('.')[0],
                                   data=serializable_results,
                                   img_path=file_path)
        # payload.strings('EXT_STORAGE_TARGET') should be the database
        self.return_status(ResultEventSignal.IDENT_AEROCUBES_FIN)

    def run(self):
        self.server.accept_connection()
        print('Controller.run: Connection accepted')
        while 1:
            json_string = self.server.receive_data()
            event = AeroCubeEvent.construct_from_json(json_string)
            self.calling_event = event
            print('Controller.run: Received Event: \r\n{}\r\n'.format(event))
            if event.signal == ImageEventSignal.IDENTIFY_AEROCUBES:
                self.initiate_scan(scan_id=event.created_at, payload=event.payload)
            else:
                pass


if __name__ == '__main__':
    controller = Controller()
    print("Controller.main: Controller is instantiated")
    testing = False
    # Create event to mock event coming in
    if testing:
        bundle = Bundle()
        filepath = "/home/ubuntu/GitHub/Aerocube/ImP/imageProcessing/test_files/jetson_test1.jpg"
        bundle.insert_string('FILE_PATH', filepath)
        bundle.insert_string('EXT_STORAGE_TARGET', 'FIREBASE')
        event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES, bundle)
        controller.initiate_scan(scan_id=event.created_at, payload=event.payload)
    else:
        controller.run()
