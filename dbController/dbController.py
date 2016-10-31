import json
import sys
from dbControllerConfig import parse_config
from firebase import write as firebase_write, read as firebase_read


def process(func, config_json, data_json=None):
    """

    :param func: function command line argument
    :param config_json: config
    :param data_json: *optional
    :return:
    """
    (location, database) = parse_config(config_json=config_json)
    # TODO: Enumerate string values

    if func == '-w':
        if data_json is None:
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

'''
Command line args:
1. Function (store or retrieve)
2. Config (where, how, etc. as JSON)
3. Data (optional, as JSON)
'''
if __name__ == '__main__':
    func = sys.argv[1]
    config = sys.argv[2]
    data = sys.argv[3] if len(sys.argv) > 2 else None

