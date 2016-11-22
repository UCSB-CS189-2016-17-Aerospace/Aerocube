from numbers import Number
import collections


class BundleKeyError(Exception):
    def __init__(self, message):
        super(BundleKeyError, self).__init__(message)


class Bundle(object):
    _strings = {}
    _numbers = {}
    _raws = {}
    _iterables = {}
    
    _IMPROPER_KEY_FORMAT_STRING = "{} is not properly formatted"
    _INCORRECT_TYPE_STRING = 'Not a string'
    _INCORRECT_TYPE_NUMBER = 'Not a number'
    _INCORRECT_TYPE_ITERABLE = 'Not an iterable'
    _INCORRECT_TYPE_RAW_IS_STRING = 'Not raw data, found string'
    _INCORRECT_TYPE_RAW_IS_NUMBER = 'Not raw data, found number'

    _ERROR_MESSAGES = (
        _IMPROPER_KEY_FORMAT_STRING,
        _INCORRECT_TYPE_STRING,
        _INCORRECT_TYPE_NUMBER,
        _INCORRECT_TYPE_ITERABLE,
        _INCORRECT_TYPE_RAW_IS_NUMBER,
        _INCORRECT_TYPE_RAW_IS_STRING
    )
    

    def __getitem__(self, item):
        return self.__dict__[item]

    @staticmethod
    def is_valid_key(key):
        """
        is_valid_key defines valid keys to be uppercase characters and underscores only
        :param key: a potential key
        :return: validity of the key
        """
        for c in key:
            if c != '_' and not c.isalpha():
                return False
        return key.isupper()

    def strings(self, key=None):
        """
        Accessor for the string dictionary or a specific key of the dictionary
        :param key:
        :return:
        """
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            try:
                return self._strings[key]
            except KeyError as err:
                raise BundleKeyError(str(err))
        else:
            return self._strings

    def numbers(self, key=None):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            return self._numbers[key]
        else:
            return self._numbers

    def raws(self, key=None):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            return self._raws[key]
        else:
            return self._raws

    def iterables(self, key):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            return self._iterables[key]
        else:
            return self._iterables

    def insert_string(self, key, s):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(s, str):
            self._strings[key] = s
        else:
            raise AttributeError(self._INCORRECT_TYPE_STRING)

    def insert_number(self, key, num):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(num, Number):
            self._numbers[key] = num
        else:
            raise AttributeError(self._INCORRECT_TYPE_NUMBER)

    def insert_raw(self, key, data):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(data, str):
            raise AttributeError(self._INCORRECT_TYPE_RAW_IS_STRING)
        elif isinstance(data, Number):
            raise AttributeError(self._INCORRECT_TYPE_RAW_IS_NUMBER)
        else:
            self._raws[key] = data

    def insert_iterable(self, key, iter):
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(iter, collections.Iterable):
            self._iterables[key] = iter
        else:
            raise AttributeError(self._INCORRECT_TYPE_ITERABLE)
