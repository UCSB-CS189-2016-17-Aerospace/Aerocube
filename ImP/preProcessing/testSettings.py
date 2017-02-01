import os


class testSettings():
    _test_files_dir_name = 'test_pictures'

    @classmethod
    def get_test_files_path(cls):
        return os.path.join(os.path.dirname(__file__), cls._test_files_dir_name)
