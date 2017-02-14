import unittest
import subprocess
import time
import psutil
from shellScripts.scriptWrapper import ScriptWrapper


class TestImageUpload(unittest.TestCase):
    _UPLOAD_STATUS_SUCCESS = "{\"upload status\": \"file upload successful\"}"

    def setUp(self):
        """
        Instantiates the controller and then the Flask Server (in that order).
        """
        # run the controller
        print("~~~ Running the controller ~~~")
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
            output = subprocess.check_output(ScriptWrapper.curlCommandPath(), timeout=30)
        except subprocess.TimeoutExpired:
            self.fail("Image upload timed out!")

        self.assertEqual(self._UPLOAD_STATUS_SUCCESS, output.splitlines()[-1].decode())

    def tearDown(self):
        """
        Kills the processes of the Controller and Flask Server.
        Checks to see if they have been terminated correctly, and prints an error otherwise.
        """
        procs = psutil.Process(self.controller.pid).children(recursive=True) + \
            psutil.Process(self.flask_server.pid).children(recursive=True)
        for proc in procs:
            proc.terminate()
        _, still_alive = psutil.wait_procs(
                            procs,
                            timeout=3,
                            callback=lambda p: print("process {} terminated with exit code {}".format(p, p.returncode))
                         )
        for proc in still_alive:
            proc.kill()
        if self.controller.poll() is None:
            print("Warning! Controller has not been killed!")
        if self.flask_server.poll() is None:
            print("Warning! Flask Server has not killed!")


if __name__ == '__main__':
    unittest.main()
