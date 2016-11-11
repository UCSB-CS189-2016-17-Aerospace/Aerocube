from flask import Flask, render_template, url_for,request, redirect 
from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename
import os

app = Flask(__name__)
api = Api(app)

app.config['UPLOAD_FOLDER'] = 'static/img/'

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
	app.run(debug=True)