from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from werkzeug import secure_filename

from controller.settings import ControllerSettings
from jobs.aeroCubeEvent import ResultEvent, AeroCubeEvent
from jobs.aeroCubeJob import AeroCubeJob
from jobs.jobHandler import JobHandler
from jobs.settings import job_id_bundle_key
from logger import Logger
from tcpService.settings import TcpSettings
from tcpService.tcpClient import TcpClient
from .settings import FlaskServerSettings

# Module-level references to "singleton" objects

_handler = None
_client = None

logger = Logger('restEndpoint.py', active=True, firebase=True)


def initialize_endpoint():
    """
    Initializes module members (handler, client, Flask app).
    Should only be called once.
    :return: (handler, client, app, api)
    """
    # Instantiate and set JobHandler functions
    job_handler = get_job_handler()
    job_handler.set_start_event_observer(on_send_event)
    job_handler.set_job_enqueue_observer(on_enqueue_job)
    job_handler.set_job_dequeue_observer(on_dequeue_job)
    # Instantiate TcpClient
    client = get_tcp_client()
    # Instantiate Flask app
    app, api = create_flask_app()
    return job_handler, client, app, api


def create_flask_app():
    """
    Creates a Flask app and api. Does NOT exhibit singleton behavior -- if called after a Flask app has
    already been instantiated, it will simply create a new Flask app.
    :return: (app, api)
    """
    app = Flask(__name__)
    api = Api(app)
    CORS(app)
    app.config[PhotoUpload.UPLOAD_FOLDER] = FlaskServerSettings.get_static_img_dir()
    api.add_resource(PhotoUpload, '/api/uploadImage')
    return app, api


# Functions to initialize singleton variables


def get_job_handler():
    """
    If module reference to handler already points to instantiated handler, simply return it.
    Otherwise, create a new one.
    Useful for REST-related classes that do not persist state between requests (and, therefore, require a
    reference to the handler each time).
    :return: Job Handler object; creates new one if does not already exist
    """
    # Get _handler from outside function scope
    global _handler
    if _handler is None:
        _handler = JobHandler()
    return _handler


def get_tcp_client():
    """
    If module reference to client already points to instantiated TcpClient, simply return it.
    Otherwise, create a new one.
    :return: TcpClient object; creates new one if does not already exist
    """
    # Get _client from outside function scope
    global _client
    if _client is None:
        _client = TcpClient(ControllerSettings.IP_ADDR(),
                            ControllerSettings.PORT(),
                            TcpSettings.BUFFER_SIZE())
        _client.connect_to_controller()
    return _client

# Handlers for EventHandler instance


def on_send_event(job_handler, event):
    """
    :param event:
    """
    logger.debug(
        class_name=None,
        func_name='on_send_event',
        msg='Started event about to send: \r\n{}\r\n'.format(event),
        id=event.payload.strings(job_id_bundle_key))
    # Send through TCP Client
    client = get_tcp_client()
    client.send_to_controller(event.to_json())
    # Check State
    # Receive Events until EventHandler.resolve_event() returns true
    decoded_response = client.receive_from_controller()
    result_event = AeroCubeEvent.construct_from_json(decoded_response)
    if not isinstance(result_event, ResultEvent):
        logger.warn(
            class_name=None,
            func_name='on_send_event',
            msg='Received message that is not instance of ResultEvent',
            id=result_event.payload.strings(job_id_bundle_key))
    else:
        try:
            event_resolved = job_handler.resolve_event(result_event)
            logger.debug(
                class_name=None,
                func_name='on_send_event',
                msg='Resolving event: event_resolved={} for event={}'.format(event_resolved, event),
                id=result_event.payload.strings(job_id_bundle_key))
            if event_resolved is True:
                logger.success(
                    class_name=None,
                    func_name='on_send_event',
                    msg='Event resolved!',
                    id=result_event.payload.strings(job_id_bundle_key))
            elif not event_resolved:
                logger.warn(
                    class_name=None,
                    func_name='on_send_event',
                    msg='Unexpected failure to resolve event!',
                    id=result_event.payload.strings(job_id_bundle_key))
        except JobHandler.NotAllowedInStateException as e:
            logger.err(
                class_name=None,
                func_name='on_send_event',
                msg=e,
                id=result_event.payload.strings(job_id_bundle_key))


def on_enqueue_job(job):
    # Do we need anything here?
    pass


def on_dequeue_job(job):
    # Do we need anything here?
    logger.debug(
        class_name=None,
        func_name='on_dequeue_job',
        msg='Job has dequeued: {}'.format(str(job)),
        id=None)


class PhotoUpload(Resource):
    """
    Handles GET and POST requests for the Flask Server.
    Constructed each time a request is handled. Therefore, it cannot persist any state across requests.
    """
    PHOTO = 'photo'
    UPLOAD_FOLDER = 'UPLOAD_FOLDER'

    def __init__(self):
        logger.debug(
            class_name=self.__class__.__name__,
            func_name='init',
            msg='Endpoint instantiated',
            id=None)

    def get(self):
        """
        Returns a success message on GET requests to verify a working connection.
        :return:
        """
        return {'server status': 'server is up and running'}

    def post(self):
        """
        Parses a request to:
            * Save the image locally
            * Send an event to the EventHandler to initiate an ImP operation
        :return:
        """
        file = request.files[self.PHOTO]
        filename = secure_filename(file.filename)
        filepath = app.config[self.UPLOAD_FOLDER]
        full_file_path = filepath + filename
        file.save(full_file_path)

        # Create Job
        new_job = AeroCubeJob.create_image_upload_job(full_file_path,
                                                      int_storage=True,
                                                      ext_store_target='firebase')
        # Enqueue Job
        get_job_handler().enqueue_job(new_job)
        return {'upload status': 'file upload successful'}

if __name__ == "__main__":
    job_handler, client, app, api = initialize_endpoint()
    # Run Flask app
    # NOTE: cannot run with debug=True, as it will cause the module to re-run
    # and mess up imported files
    app.run(debug=False, port=FlaskServerSettings.PORT(), ssl_context='adhoc')
