import json
from .commClass import FirebaseComm
# TODO: data_json finalization


def _select_database(database, testing):
    if database == 'firebase':
        return FirebaseComm(testing)
    else:
        raise ValueError('database not found!')


def external_write(database, scanID, location=None, data=None, testing=False):
    comm = _select_database(database=database, testing=testing)
    return comm.write(location=location, id=scanID, data=data)


def external_store_img(database, scanID, srcImage=None, testing=False):
    comm = _select_database(database=database, testing=testing)
    return comm.imageStore(id=scanID, srcImage=srcImage)


def external_read(database, scanID, location=None, testing=False):
    comm = _select_database(database=database, testing=testing)
    return comm.read(location=location, id=scanID)


def external_delete(database, scanID, location=None, testing=False):
    comm = _select_database(database=database, testing=testing)
    return comm.delete(location=location, id=scanID)


def external_download_image(database, scanID, testing=False):
    comm = _select_database(database=database, testing=testing)
    return comm.imageDownload(scanID)
