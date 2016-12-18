from flask import Flask
import unittest
import requests
import subprocess
import os
import flaskServer.restEndpoint as restEndpoint
from .settings import FlaskServerSettings

# Make sure to have an image named sampleImage.png on the flaskServer directory in order to run these tests


class RestTestCase(unittest.TestCase):
    _sample_image_path = os.path.join(FlaskServerSettings.get_flask_server_dir(), 'static/img/sampleImage.png')

    def setUp(self):
        restEndpoint.app.config['TESTING'] = True
        self.app = restEndpoint.app.test_client()

    def tearDown(self):
        subprocess.call(['rm', 'static/img/sampleImage.png'])

    def test_successful_upload(self):
        files = {'photo': open('sampleImage.png', 'rb')}
        response = requests.post('http://127.0.0.1:5000/api/uploadImage', files=files)
        self.assertEqual(response.status_code, 200)

    def test_file_existence(self):
        files = {'photo': open('sampleImage.png', 'rb')}
        ls_command = subprocess.check_output(['ls', 'static/img'])
        self.assertFalse('sampleImage.png' in ls_command,
                         msg='Precondition: {0} image should not exist at the API image upload endpoint.'.format('sampleImage.png'))
        existence = False
        requests.post('http://127.0.0.1:5000/api/uploadImage', files=files)
        ls_command = subprocess.check_output(['ls', 'static/img'])
        if 'sampleImage.png' in ls_command:
            existence = True
        self.assertTrue(existence,
                        msg='Postcondition: {0} was not sucessfully created at the API image upload endpoint.'.format('sampleImage.png'))


if __name__ == '__main__':
    unittest.main()
