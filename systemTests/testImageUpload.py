import unittest
import subprocess
import os
import signal
import time
from shellScripts.scriptWrapper import ScriptWrapper


class TestImageUpload(unittest.TestCase):
    _UPLOAD_STATUS_SUCCESS = "{\"upload status\": \"file upload sucessful\"}"

    def setUp(self):
        """
        Instantiates the controller and then the Flask Server (in that order).
        """
        # run the controller
        print("~~~ Running the controller ~~~")
        print([ScriptWrapper.runControllerPath()])
        self.controller = subprocess.Popen(ScriptWrapper.runControllerPath())
        # wait for controller to instantiate
        time.sleep(2)

        # run Flask Server
        print("~~~ Running the Flask Server ~~~")
        self.flask_server = subprocess.Popen(ScriptWrapper.runFlaskServerPath())
        # wait for Flask Server and controller to connect
        time.sleep(5)

    def test_run_controller_and_handler_and_upload_image(self):
        """
        Calls curlCommand script command, checking output against known successful output.
        """

        # initiate image upload via curl command
        print("~~~ Calling curl command ~~~")
        try:
            output = subprocess.check_output(ScriptWrapper.curlCommandPath(), timeout=25)
        except subprocess.TimeoutExpired:
            self.fail("Image upload timed out!")

        self.assertEqual(output.splitlines()[-1].decode(), self._UPLOAD_STATUS_SUCCESS)

    def tearDown(self):
        """
        Kills the processes of the Controller and Flask Server.
        Checks to see if they have been terminated correctly, and prints an error otherwise.
        """
        os.killpg(os.getpgid(self.controller.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.flask_server.pid), signal.SIGTERM)
        if self.controller.poll() is None:
            print("Warning! Controller has not terminated!")
        if self.flask_server.poll() is None:
            print("Warning! Flask Server has not terminated!")


if __name__ == '__main__':
    unittest.main()
