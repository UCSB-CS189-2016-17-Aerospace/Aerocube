import json
from commClass import FirebaseComm
#TODO: data_json finalization
def process(func, database, location, data):
    """
    :param func: '-w'|'-r'
    :param database: 'firebase'
    :param location: path in database
    :param data_json: just ID or ID and data depending or read or write
    :return:
    """
    comm=None
    if(database=='firebase'):
        comm=FirebaseComm()
    else:
        raise ValueError('database not found!')
    if func == '-w':
        comm.write(location=location,data=data)
    elif func == '-r':
        comm.read(location=location,data=data)
    else:
        raise SyntaxError('func not accepted, -w for write, -r for read.')


def store(location, method, data_json):
    """
    :param location:
    :param method:
    :param data_json:
    :return:
    """
    # TODO: move error message for Database not found to process().
    if method is None:
        raise ValueError('Database does not exist.')
    method(location,data_json)

def retrieve(location, method):
    """

    :param location:
    :param method:
    :return:
    """
    # TODO: move error message for Database not found to process()
    if method is None:
        raise ValueError('Database does not exist.')
    method(location)
