import os


class FlaskServerSettings():
    @staticmethod
    def get_flask_server_dir():
        return os.path.dirname(__file__)
