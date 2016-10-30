import json
from enum import Enum

"""
{
    location: string, (table name, path within JSON tree)
    method: enumerated_value (Firebase, etc)
}
"""

# TODO: Handle enumeration of storage/retrieval methods


def parse_config(config_json):
    """

    :param config_json:
    :return: (location, method)
    """
    pass


def generate_config(location, method, *args, **kwargs):
    """

    :param location:
    :param method:
    :param args:
    :param kwargs:
    :return: a valid json object containing a dbController configuration
    """
    pass



"""
{
    image_process: {

    },
    results: {
        some_id: {
            attr: value
        }
    },
}
"""