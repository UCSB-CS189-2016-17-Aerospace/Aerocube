# import settings
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from dataStorage.dataStorage import store
from externalComm.externalComm import process
from jobs.aeroCubeEvent import AeroCubeEvent, ImageEvent, StorageEvent, ResultEvent
from jobs.aeroCubeSignal import ImageEventSignal, StorageEventSignal, ResultEventSignal
from jobs.bundle import Bundle
from tcpService.settings import TcpSettings
from tcpService.tcpServer import TcpServer
from .settings import ControllerSettings


class Controller:
    """
    :ivar _server: TcpServer instance created upon init
    :ivar _dispatcher: dictionary that maps AeroCubeSignals to functions
    :ivar calling_event:
    """
    def __init__(self):
        self._server = TcpServer(ControllerSettings.IP_ADDR(),
                                 ControllerSettings.PORT(),
                                 TcpSettings.BUFFER_SIZE())
        # All functions must return tuple of type (ResultEventSignal, bundle)
        self._dispatcher = {
            ImageEventSignal.IDENTIFY_AEROCUBES: self.initiate_scan,
            StorageEventSignal.STORE_INTERNALLY: self.store_internally,
            StorageEventSignal.STORE_EXTERNALLY: self.store_data_externally
        }
        self.calling_event = None

    @property
    def server(self):
        return self._server

    @property
    def dispatcher(self):
        return self._dispatcher

    def return_status(self, status, bundle):
        """
        Returns status back to event handler
        :param status: status signal
        :param bundle: needs to be set before response is sent
        :return: void
        """
        # return
        print('Controller.return_status: Status is {}'.format(status))
        result_event = ResultEvent(result_signal=status, calling_event_uuid=self.calling_event.uuid)
        print('Controller.return_status: Sending ResultEvent: \r\n{}\r\n'.format(result_event))
        self.server.send_response(result_event.to_json())

    def scan_image(self, img_event):
        """
        Handler for ImageEventSignal.IDENTIFY_AEROCUBES.
        :param img_event:
        :return:
        """
        # Initialize variables
        file_path = img_event.payload.strings(ImageEvent.FILE_PATH)
        try:
            print('Controller.scan_image: Instantiating ImP at {}'.format(file_path))
            imp = ImageProcessor(file_path)
            print('Controller.scan_image: Finding fiducial markers')
            # TODO: replace with imp.scan_image(signal)
            corners, marker_ids = imp._find_fiducial_markers()
            # Set result signal to OK
            result_signal = ResultEventSignal.OK
            # Prepare bundle from original
            results_bundle = img_event.payload
            results_bundle.insert_string(ImageEvent.SCAN_ID, img_event.created_at)
            results_bundle.insert_raw(ImageEvent.SCAN_CORNERS, corners)
            results_bundle.insert_raw(ImageEvent.SCAN_MARKER_IDS, marker_ids)
        except Exception:
            print('Controller.scan_image: ImP Failed')
            results_signal = ResultEventSignal.ERROR
            results_bundle = img_event.payload
        return results_signal, results_bundle

    def store_internally(self, store_event):
        """
        Handler for StorageEvent.STORE_INTERNALLY.
        :param store_event:
        :return:
        """
        print('Controller.store_locally: Storing data locally')
        # TODO: data is hardcoded, need to find way for StorageEvent to indicate what should be stored
        data = (store_event.payload.raw(ImageEvent.SCAN_ID),
                store_event.payload.raw(ImageEvent.SCAN_CORNERS),
                store_event.payload.raw(ImageEvent.SCAN_MARKER_IDS))
        store(store.event.payload.strings(StorageEvent.INT_STORAGE_REL_PATH),
              data)
        return ResultEventSignal.OK, store_event.payload

    def store_data_externally(self, database, scan_id, data, img_path):
        try:
            print('Controller: Storing data externally')
            external_write(database=database, scanID=scan_id, data=data)
            print('Controller: Storing image externally')
            external_store_img(database=database, scanID=scan_id, data=img_path)
            print('Controller: Successfully stored externally, sending ResultEvent')
            self.return_status(ResultEventSignal.EXT_COMM_OP_OK)
        except ValueError:
            print('Controller.store_data_externally: External storage failed')
            self.return_status(ResultEventSignal.EXT_COMM_OP_FAILED)

    def initiate_scan(self, scan_id, payload):
        print('Controller.initiate_scan: Initiate Scan')
        # print(payload)
        file_path = payload.strings(ImageEvent.FILE_PATH)
        print('Controller.initiate_scan: Payload FILE_PATH is {}'.format(file_path))
        results = self.scan_image(file_path=file_path)
        print('Controller.initiate_scan: Results Received, sending ResultEvent')
        self.return_status(ResultEventSignal.IMP_OPERATION_OK)
        print('Controller.initiate_scan: Scanning results received')
        print('Controller.initiate_scan: \r\n{}\r\n'.format(results))
        # payload.strings('FILE_PATH') should be the path to the image
        self.store_internally(path=str(scan_id), data=results)
        serializable_results = (list(map((lambda c: c.tolist()), results[0])),
                                results[1].tolist())
        self.store_data_externally(database=payload.strings(StorageEvent.EXT_STORAGE_TARGET),
                                   scan_id=str(scan_id).split('.')[0],
                                   data=serializable_results,
                                   img_path=file_path)
        # payload.strings('EXT_STORAGE_TARGET') should be the database
        self.return_status(ResultEventSignal.IDENT_AEROCUBES_FIN)

    def run(self):
        self.server.accept_connection()
        print('Controller.run: Connection accepted')
        while 1:
            # Parse received data and rehydrate AeroCubeEvent object
            json_string = self.server.receive_data()
            event = AeroCubeEvent.construct_from_json(json_string)
            # Set as calling event
            self.calling_event = event
            print('Controller.run: Received Event: \r\n{}\r\n'.format(event))
            try:
                status, bundle = self.dispatcher[event.signal](event)
                self.return_status(status, bundle)
            except KeyError:
                print('Controller.run: No function found to handle event of type \r\n{}\r\n'.format(event.signal))
            except Exception as ex:
                # all other exceptions
                # TODO: how to handle?
                template = "An exception of type {0} occured. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)


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
