import unittest
import subprocess
import time
from shellScripts.scriptWrapper import ScriptWrapper


class TestImageUpload(unittest.TestCase):
    _UPLOAD_STATUS_SUCCESS = "{\"upload status\": \"file upload sucessful\"}"

    def test_run_controller_and_handler_and_upload_image(self):
        """
        Calls scripts from shellScripts as done in normal execution.
        """
        # run the controller
        print("~~~ Running the controller ~~~")
        controller = subprocess.Popen(ScriptWrapper.runControllerPath())
        # wait for controller to instantiate
        time.sleep(2)

        # run Flask Server
        print("~~~ Running the Flask Server ~~~")
        flask_server = subprocess.Popen(ScriptWrapper.runFlaskServerPath())
        # wait for Flask Server and controller to connect
        time.sleep(5)

        # initiate image upload via curl command
        print("~~~ Calling curl command ~~~")
        try:
            output = subprocess.check_output(ScriptWrapper.curlCommandPath(), timeout=25)
        except TimeoutExpired:
            self.fail("Image upload timed out!")

        # kill sub processes before checking final output
        controller.kill()
        flask_server.kill()
        self.assertEqual(output.splitlines()[-1].decode(), self._UPLOAD_STATUS_SUCCESS)
        if controller.poll() is None:
            print("Warning! Controller has not terminated!")
        if flask_server.poll() is None:
            print("Warning! Flask Server has not terminated!")

if __name__ == '__main__':
    unittest.main()
