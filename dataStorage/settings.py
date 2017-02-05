import os
storageDirectory = 'scanInformation/'


def get_storage_directory():
    return os.path.join(os.path.dirname(__file__), storageDirectory)
