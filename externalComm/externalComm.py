import json

#TODO: data_json finalization
def process(func, database, location, data=None):
    """
    :param func: '-w'|'-r'
    :param database: 'firebase'
    :param location: path in database
    :param data_json: optional
    :return:
    """

    if func == '-w':
        if data is None:
            raise ValueError('Data cannot be none with func flag -w set.')
        method = firebase_write if database == 'firebase' else None
        store(location=location, method=method, data_json=data_json)
    elif func == '-r':
        method = firebase_read if database == 'firebase' else None
        retrieve(location=location, method=method)
    else:
        raise SyntaxError('Usage: -w for write, -r for read.')


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
