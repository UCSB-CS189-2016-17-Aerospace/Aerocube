from flask import Flask, render_template, url_for, request, redirect
from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename
from flask_cors import CORS, cross_origin
import os
from eventClass.eventHandler import EventHandler

app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'static/img/'

handler = EventHandler()


class PhotoUpload(Resource):
	def get(self):
		return {'server status': 'server is up and running'}


	def post(self):
		file = request.files['photo']
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return {'upload status' : 'file upload sucessful'}


api.add_resource(PhotoUpload, '/api/uploadImage')

if __name__ == "__main__":
	app.run(debug=False, port=5000, ssl_context='adhoc')
