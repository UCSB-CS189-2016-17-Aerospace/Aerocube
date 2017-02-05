import numpy as np
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from dataStorage.dataStorage import store
from externalComm.externalComm import external_write, external_store_img
from jobs.aeroCubeEvent import AeroCubeEvent, ImageEvent, StorageEvent, ResultEvent
from jobs.aeroCubeSignal import ImageEventSignal, StorageEventSignal, ResultEventSignal
from jobs.bundle import Bundle
from tcpService.settings import TcpSettings
from tcpService.tcpServer import TcpServer
from controller.settings import ControllerSettings


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
            ImageEventSignal.IDENTIFY_AEROCUBES: self.scan_image,
            StorageEventSignal.STORE_INTERNALLY: self.store_internally,
            StorageEventSignal.STORE_EXTERNALLY: self.store_externally
        }
        self.calling_event = None

    @property
    def server(self):
        return self._server

    @property
    def dispatcher(self):
        return self._dispatcher

    def return_status(self, status, result_bundle, event):
        """
        Returns status back to event handler
        :param status: status signal
        :param result_bundle: needs to be set before response is sent
        :return: void
        """
        # return
        print('Controller.return_status: Status is {}'.format(status))
        result_event = ResultEvent(result_signal=status,
                                   calling_event_uuid=self.calling_event.uuid,
                                   bundle=result_bundle)
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
            # Ensure data is JSONifiable
            corners, marker_ids = np.array(corners).tolist(), np.array(marker_ids).tolist()
            print('Controller.scan_image: Done with scan!')
            # Set result signal to OK
            result_signal = ResultEventSignal.OK
            # Prepare bundle from original
            result_bundle = img_event.payload
            result_bundle.insert_string(ImageEvent.SCAN_ID, str(img_event.created_at).split('.')[0])
            result_bundle.insert_iterable(ImageEvent.SCAN_CORNERS, corners)
            result_bundle.insert_iterable(ImageEvent.SCAN_MARKER_IDS, marker_ids)
            print('Controller.scan_image: Done with setting bundle : {}'.format(str(result_bundle)))
        except Exception as ex:
            print('Controller.scan_image: ImP Failed')
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            result_signal = ResultEventSignal.ERROR
            result_bundle = img_event.payload
        return result_signal, result_bundle

    def store_internally(self, store_event):
        """
        Handler for StorageEvent.STORE_INTERNALLY.
        :param store_event:
        :return:
        """
        print('Controller.store_internally: Storing data locally')
        # TODO: data is hardcoded, need to find way for StorageEvent to indicate what should be stored
        data = store_event.parse_storage_keys()
        print(data)
        store(store_event.payload.strings(ImageEvent.SCAN_ID),
              data)
        return ResultEventSignal.OK, store_event.payload

    def store_externally(self, store_event):
        database = store_event.payload.strings(StorageEvent.EXT_STORAGE_TARGET)
        scan_id = store_event.payload.strings(ImageEvent.SCAN_ID)
        data = store_event.parse_storage_keys()
        img_data = store_event.payload.strings(ImageEvent.FILE_PATH)
        try:
            print('Controller: Storing data externally')
            external_write(database=database, scanID=scan_id,
                           data=data, testing=True)
            print('Controller: Storing image externally')
            external_store_img(database=database, scanID=scan_id,
                               srcImage=img_data, testing=True)
            print('Controller: Successfully stored externally, sending ResultEvent')
            return ResultEventSignal.OK, store_event.payload
        except ValueError as ex:
            print('Controller.store_data_externally: External storage failed with args {}'.format(ex.args))
            return ResultEventSignal.ERROR, store_event.payload

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
            rcved_event = AeroCubeEvent.construct_from_json(json_string)
            # Set as calling rcved_event
            self.calling_event = rcved_event
            print('Controller.run: Received Event: \r\n{}\r\n'.format(rcved_event))
            try:
                status, bundle = self.dispatcher[rcved_event.signal](rcved_event)
                self.return_status(status, bundle, rcved_event)
            except KeyError:
                print('Controller.run: No function found to handle rcved_event of type \r\n{}\r\n'.format(rcved_event.signal))
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
