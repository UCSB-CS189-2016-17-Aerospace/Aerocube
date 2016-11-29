import pickle
import cv2
from .settings import *
from eventClass.aeroCubeSignal import *


def store(location, pickleable, use_relative_location=True):
    """
    Public abstraction of storage
    :param location: string path
    :param pickleable: data that can be pickled
    :param use_relative_location: bool
    :return: An event containing a signal that describes success or failure
    """
    return _store_json(get_storage_directory() + location, pickleable)

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
    :param json:
    :return:
    """
    try:
        pickle.dump(json, open(location, "wb"), pickle.HIGHEST_PROTOCOL)
        # return ResultEventSignal.INTERN_STORE_OP_OK
    except OSError as err:
        return ResultEventSignal.INTERN_STORE_OP_FAILED


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
