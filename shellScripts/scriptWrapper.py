import os


class ScriptWrapper():
    """
    Wraps calls to shell scripts in the current directory
    """
    _SCRIPT_DIR = os.path.dirname(__file__)
    _RUN_CONTROLLER = 'runController.sh'
    _RUN_FLASK_SERVER = 'runFlaskServer.sh'
    _CURL_COMMAND = 'curlCommand.sh'

    @classmethod
    def curlCommandPath(cls):
        return os.path.join(cls._SCRIPT_DIR, cls._CURL_COMMAND)

    @classmethod
    def runControllerPath(cls):
        return os.path.join(cls._SCRIPT_DIR, cls._RUN_CONTROLLER)

    @classmethod
    def runFlaskServerPath(cls):
        return os.path.join(cls._SCRIPT_DIR, cls._RUN_FLASK_SERVER)
