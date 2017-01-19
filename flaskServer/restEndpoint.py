"""
TODO: handler (EventHandler instance) is referenced nowhere but w/in PhotoUpload;
    should it be moved inside of PhotoUpload, or be passed as a parameter?
TODO: client (TcpClient instance) is referenced nowhere but in on_send_event, and
    seems it should be passed as a parameter instead of referenced as a global
"""
import os

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from werkzeug import secure_filename

from controller.settings import ControllerSettings
from jobs.aeroCubeEvent import ImageEvent, ResultEvent, AeroCubeEvent
from jobs.aeroCubeSignal import *
from jobs.bundle import Bundle
from jobs.jobHandler import JobHandler
from tcpService.settings import TcpSettings
from tcpService.tcpClient import TcpClient
from .settings import FlaskServerSettings

app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['UPLOAD_FOLDER'] = FlaskServerSettings.get_static_img_dir()

handler = JobHandler()
client = TcpClient(ControllerSettings.IP_ADDR(),
                   ControllerSettings.PORT(),
                   TcpSettings.BUFFER_SIZE())
client.connect_to_controller()

# define handlers for EventHandler instance


def on_send_event(event):
    """
    :param event:
    """
    print('RestEndpoint.on_send_event: Started event about to send: \r\n{}\r\n'.format(event))
    # Send through TCP Client
    client.send_to_controller(event.to_json())
    # Check State
    # Receive Events until EventHandler.resolve_event() returns true
    while True:
        decoded_response = client.receive_from_controller()
        result_event = AeroCubeEvent.construct_from_json(decoded_response)
        if not isinstance(result_event, ResultEvent):
            print('RestEndpoint.on_send_event: Warning: Received message that is not instance of ResultEvent')
        else:
            try:
                event_resolved = handler.resolve_event(result_event)
                if event_resolved:
                    break
                else:
                    continue
            except JobHandler.NotAllowedInStateException as e:
                print(e)


def on_enqueue_event():
    # Do we need anything here?
    pass


def on_dequeue_event():
    # Do we need anything here?
    pass


handler.set_start_event_observer(on_send_event)
handler.set_enqueue_observer(on_enqueue_event)
handler.set_dequeue_observer(on_dequeue_event)


class PhotoUpload(Resource):
    """
    Handles GET and POST requests for the Flask Server.
    """
    def get(self):
        """
        Returns a success message on GET requests to verify a working connnection.
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
        file = request.files['photo']
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
        file.save(filepath + filename)
        # Create Event
        bundle = Bundle()
        bundle.insert_string('FILE_PATH', filepath + filename)
        bundle.insert_string('EXT_STORAGE_TARGET', 'FIREBASE')
        new_event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES, bundle)
        # Enqueue Event
        handler.enqueue_event(new_event)
        return {'upload status': 'file upload sucessful'}

api.add_resource(PhotoUpload, '/api/uploadImage')

if __name__ == "__main__":
    # NOTE: cannot run with debug=True, as it will cause the module to re-run
    # and mess up imported files
    app.run(debug=False, port=FlaskServerSettings.PORT(), ssl_context='adhoc')
