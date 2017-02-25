import pickle

import cv2

from jobs.aeroCubeSignal import *
from .settings import *


def store(rel_path, data):
    """
    Public abstraction of storage
    :param rel_path: string path, relative to storage directory (defined by dataStorage module)
    :param data: data that can be pickled
    :return: Boolean indicating whether storage was successful or not
    """
    return _store_json(get_storage_directory() + rel_path, data)


# TODO: need test
def store_image(location, img):
    """
    """
    cv2.imwrite(location, img)


def retrieve(location):
    """
    Public abstraction of retrieval
    :param location: relative location
    :return:
    """
    return _retrieve_json(get_storage_directory() + location)


# below is random stuff
def _store_json(location, json):
    """
    'Private' method to store json data
    :param location: the location of the file
    :param json: data to store
    :return: boolean indicating whether store was successful or not
    """
    try:
        pickle.dump(json, open(location, "wb"), pickle.HIGHEST_PROTOCOL)
        return True
    except OSError as err:
        return False


def _retrieve_json(location):
    """
    'Private' method to retrieve json data
    :param location:
    :return:
    """
    try:
        return pickle.load(open(location,"rb"))
    except OSError as err:
        pass
        # TODO: Handle err
