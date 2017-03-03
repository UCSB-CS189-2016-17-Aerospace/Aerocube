import numpy as np

from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.aerocubeMarker import AeroCube
from controller.settings import ControllerSettings
from dataStorage.dataStorage import store
from externalComm.externalComm import external_write, external_store_img
from jobs.aeroCubeEvent import AeroCubeEvent, ImageEvent, StorageEvent, ResultEvent
from jobs.aeroCubeSignal import ImageEventSignal, StorageEventSignal, ResultEventSignal
from jobs.bundle import Bundle
from jobs.settings import job_id_bundle_key
from logger import Logger
from tcpService.settings import TcpSettings
from tcpService.tcpServer import TcpServer

logger = Logger('controller.py', active=True, external=True)


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
        :param event:
        :return: void
        """
        # return
        logger.debug(
            self.__class__.__name__,
            func_name='return_status',
            msg='Status is {}'.format(status),
            id=event.payload.strings(job_id_bundle_key))
        result_event = ResultEvent(result_signal=status,
                                   calling_event_uuid=self.calling_event.uuid,
                                   bundle=result_bundle)
        logger.debug(
            self.__class__.__name__,
            func_name='return_status',
            msg='Sending ResultEvent: \r\n{}\r\n'.format(result_event),
            id=event.payload.strings(job_id_bundle_key))
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
            logger.debug(
                self.__class__.__name__,
                func_name='scan_image',
                msg='Instantiating ImP at {}'.format(file_path),
                id=img_event.payload.strings(job_id_bundle_key))
            imp = ImageProcessor(file_path)
            logger.debug(
                self.__class__.__name__,
                func_name='scan_image',
                msg='Finding fiducial markers',
                id=img_event.payload.strings(job_id_bundle_key))
            aerocubes_as_json, markers_as_json = imp.identify_markers_for_storage()
            logger.success(
                self.__class__.__name__,
                func_name='scan_image',
                msg='Done with Scan!',
                id=img_event.payload.strings(job_id_bundle_key))
            # Set result signal to OK
            result_signal = ResultEventSignal.OK
            # Prepare bundle from original
            result_bundle = img_event.payload
            result_bundle.insert_string(ImageEvent.SCAN_ID, str(img_event.created_at).split('.')[0])
            result_bundle.insert_raw(AeroCube.STR_KEY_CUBE_IDS, aerocubes_as_json[AeroCube.STR_KEY_CUBE_IDS])
            result_bundle.insert_raw(AeroCube.STR_KEY_QUATERNIONS, aerocubes_as_json[AeroCube.STR_KEY_QUATERNIONS])
            result_bundle.insert_raw(AeroCube.STR_KEY_DISTANCES, aerocubes_as_json[AeroCube.STR_KEY_DISTANCES])
            result_bundle.insert_raw(AeroCube.STR_KEY_MARKERS_DETECTED, aerocubes_as_json[AeroCube.STR_KEY_MARKERS_DETECTED])
            result_bundle.insert_raw(ImageEvent.SCAN_MARKERS, markers_as_json)
            logger.success(
                self.__class__.__name__,
                func_name='scan_image',
                msg='Controller.scan_image: Done with setting bundle : {}'.format(str(result_bundle)),
                id=img_event.payload.strings(job_id_bundle_key))
            
        except Exception as ex:
            logger.err(
                self.__class__.__name__,
                func_name='scan_image',
                msg='ImP Failed',
                id=img_event.payload.strings(job_id_bundle_key))
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            logger.err(
                self.__class__.__name__,
                func_name='scan_image',
                msg=message,
                id=img_event.payload.strings(job_id_bundle_key))
            result_signal = ResultEventSignal.ERROR
            result_bundle = img_event.payload
        return result_signal, result_bundle

    def store_internally(self, store_event):
        """
        Handler for StorageEvent.STORE_INTERNALLY.
        :param store_event:
        :return:
        """
        logger.debug(
            self.__class__.__name__,
            func_name='store_internally',
            msg='Storing data locally',
            id=store_event.payload.strings(job_id_bundle_key))
        data = store_event.parse_storage_keys()
        logger.debug(
            self.__class__.__name__,
            func_name='store_internally',
            msg=data,
            id=store_event.payload.strings(job_id_bundle_key))
        store(store_event.payload.strings(ImageEvent.SCAN_ID),
              data)
        return ResultEventSignal.OK, store_event.payload

    def store_externally(self, store_event):
        database = store_event.payload.strings(StorageEvent.EXT_STORAGE_TARGET)
        scan_id = store_event.payload.strings(ImageEvent.SCAN_ID)
        data = store_event.parse_storage_keys()
        print(store_event.payload)
        print(data)
        img_data = store_event.payload.strings(ImageEvent.FILE_PATH)
        try:
            logger.debug(
                self.__class__.__name__,
                func_name='store_externally',
                msg='Storing data externally',
                id=store_event.payload.strings(job_id_bundle_key))
            external_write(database=database, scanID=scan_id, location='scans',
                           data=data, testing=True)
            logger.debug(
                self.__class__.__name__,
                func_name='store_externally',
                msg='Storing image externally',
                id=store_event.payload.strings(job_id_bundle_key))
            external_store_img(database=database, scanID=scan_id,
                               srcImage=img_data, testing=True)
            logger.success(
                self.__class__.__name__,
                func_name='store_externally',
                msg='Successfully stored externally, sending ResultEvent',
                id=store_event.payload.strings(job_id_bundle_key))
            return ResultEventSignal.OK, store_event.payload
        except ValueError as ex:
            logger.err(
                self.__class__.__name__,
                func_name='store_externally',
                msg='External storage failed with args {}'.format(ex.args),
                id=store_event.payload.strings(job_id_bundle_key))
            return ResultEventSignal.ERROR, store_event.payload

    def run(self):
        self.server.accept_connection()
        logger.debug(
            class_name=None,
            func_name='run',
            msg='Controller connection accepted',
            id=None)
        while 1:
            # Parse received data and rehydrate AeroCubeEvent object
            json_string = self.server.receive_data()
            rcved_event = AeroCubeEvent.construct_from_json(json_string)
            # Set as calling rcved_event
            self.calling_event = rcved_event
            logger.success(
                self.__class__.__name__,
                func_name='run',
                msg='Received Event: \r\n{}\r\n'.format(rcved_event),
                id=rcved_event.payload.strings(job_id_bundle_key))
            try:
                status, bundle = self.dispatcher[rcved_event.signal](rcved_event)
                self.return_status(status, bundle, rcved_event)
            except KeyError:
                logger.err(
                    self.__class__.__name__,
                    func_name='run',
                    msg='No function found to handle rcved_event of type \r\n{}\r\n'.format(rcved_event.signal),
                    id=rcved_event.payload.strings(job_id_bundle_key))
            except Exception as ex:
                # all other exceptions
                # TODO: how to handle?
                template = "An exception of type {0} occured. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                logger.err(
                    self.__class__.__name__,
                    func_name='run',
                    msg=message,
                    id=rcved_event.payload.strings(job_id_bundle_key))


if __name__ == '__main__':
    controller = Controller()
    logger.debug(
        class_name=None,
        func_name='main',
        msg='Controller is instantiated',
        id=None)
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
