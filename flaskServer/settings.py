import os


class FlaskServerSettings():
    _static_img_rel_path = 'static/img/'
    _test_files_rel_path = 'test_files'

    @staticmethod
    def get_flask_server_dir():
        return os.path.dirname(__file__)

    @staticmethod
    def get_static_img_dir():
        return os.path.join(FlaskServerSettings.get_flask_server_dir(),
                            FlaskServerSettings._static_img_rel_path)

    @staticmethod
    def get_test_files_dir():
        return os.path.join(FlaskServerSettings.get_flask_server_dir(),
                            FlaskServerSettings._test_files_rel_path)
