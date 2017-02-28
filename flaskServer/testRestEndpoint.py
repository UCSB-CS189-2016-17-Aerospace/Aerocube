import unittest
import requests
import subprocess
import os
import flaskServer.restEndpoint as restEndpoint
from flaskServer.settings import FlaskServerSettings


class TestRestEndpointInstantiationMethods(unittest.TestCase):

    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    def tearDown(self):
        restEndpoint._handler = None
        restEndpoint._client = None

    def test_handler_and_client_none_on_load(self):
        self.assertIsNone(restEndpoint._handler)
        self.assertIsNone(restEndpoint._client)

    def test_handler_not_none_after_get_and_matching_id(self):
        self.assertIsNone(restEndpoint._handler)
        handler = restEndpoint.get_job_handler()
        self.assertIsNotNone(restEndpoint._handler)
        self.assertEqual(id(handler), id(restEndpoint._handler))

    def test_client_not_none_after_get_and_matching_id(self):
        self.assertIsNone(restEndpoint._client)
        client = restEndpoint.get_tcp_client()
        self.assertIsNotNone(restEndpoint._client)
        self.assertEqual(id(client), id(restEndpoint._client))

    def test_create_flask(self):
        app, api = restEndpoint.create_flask_app()
        self.assertIsNotNone(app)
        self.assertIsNotNone(api)
        self.assertEqual(app.config[restEndpoint.PhotoUpload.UPLOAD_FOLDER], FlaskServerSettings.get_static_img_dir())

    def test_initialize_endpoint(self):
        self.assertIsNone(restEndpoint._handler)
        self.assertIsNone(restEndpoint._client)
        handler, client, app, api = restEndpoint.initialize_endpoint()
        self.assertIsNotNone(restEndpoint._handler)
        self.assertIsNotNone(restEndpoint._client)
        self.assertIsNotNone(app)
        self.assertIsNotNone(api)


class TestRestEndpoint(unittest.TestCase):
    """
    TODO: fails unless the controller is also running
    Make sure _test_img has the name (as a str) of an existing file
    in the designated test_file dir.
    """
    _test_img = 'ucsb_logo.jpg'
    _static_img_dir = FlaskServerSettings.get_static_img_dir()
    _test_img_path = os.path.join(FlaskServerSettings.get_test_files_dir(), _test_img)
    _static_img_path = os.path.join(FlaskServerSettings.get_static_img_dir(), _test_img)

    @classmethod
    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    def setUp(self):
        self.handler, self.client, self.app, self.api = restEndpoint.initialize_endpoint()
        self.app.config['TESTING'] = True
        self.app = restEndpoint.app.test_client()

    def tearDown(self):
        # Remove references to JobHandler
        restEndpoint._handler = None
        del self.handler
        # Close TcpClient connection
        restEndpoint._client.close()
        restEndpoint._client = None
        del self.client
        # Remove test image
        subprocess.call(['rm', self._static_img_path])

    def test_successful_upload(self):
        files = {'photo': open(self._test_img_path, 'rb')}
        response = requests.post('http://127.0.0.1:5005/api/uploadImage', files=files)
        self.assertEqual(response.status_code, 200)

    def test_file_existence(self):
        files = {'photo': open(self._test_img_path, 'rb')}
        ls_command = subprocess.getoutput('ls ' + self._static_img_dir)
        self.assertFalse(self._test_img in ls_command.split('\n'),
                         msg='Precondition: {0} image should not exist at the API image upload endpoint.'.format(self._test_img))
        requests.post('http://127.0.0.1:5005/api/uploadImage', files=files)
        ls_command = subprocess.getoutput('ls ' + self._static_img_dir)
        self.assertTrue(self._test_img in ls_command.split('\n'),
                        msg='Postcondition: {0} was not successfully created at the API image upload endpoint.'.format(self._test_img))


if __name__ == '__main__':
    unittest.main()
