from flask import Flask, render_template, url_for, request, redirect
from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename
from flask_cors import CORS, cross_origin
import os
from eventClass.eventHandler import EventHandler
from eventClass.aeroCubeEvent import ImageEvent, SystemEvent
from eventClass.aeroCubeSignal import *
from eventClass.bundle import Bundle
from .tcpClient import TcpClient
app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'flaskServer/static/img/'

handler = EventHandler()
client = TcpClient('127.0.0.1',5005,1024)
client.connect_to_controller()


def on_send_event(event):
    # Send through TCP Client
    client.send_to_controller(event)
    # Receive Event
    result_event = client.receive_from_controller()
    # Check State
    try:
        handler.resolve_event(result_event)
    except EventHandler.NotAllowedInStateException as e:
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
    def get(self):
        return {'server status': 'server is up and running'}

    def post(self):
        file = request.files['photo']
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
        file.save(filepath + filename)
        # Create Event
        bundle = Bundle()
        bundle.insert_string('FILE_PATH', filepath)
        bundle.insert_string('EXT_STORAGE_TARGET', 'FIREBASE')
        new_event = ImageEvent(ImageEventSignal.IDENTIFY_AEROCUBES, bundle)
        # Enqueue Event
        handler.enqueue_event(new_event)
        return {'upload status' : 'file upload sucessful'}


api.add_resource(PhotoUpload, '/api/uploadImage')

if __name__ == "__main__":
    # NOTE: cannot run with debug=True, as it will cause the module to re-run
    # and mess up imported files
    app.run(debug=False, port=5000, ssl_context='adhoc')

