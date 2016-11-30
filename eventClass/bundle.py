from numbers import Number
import collections
import json


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

    def __eq__(self, other):
        """
        Enables use of == comparison, comparing instance type and all dicts for equality
        :param other: the other Bundle
        :return: True if equal, False if inequal
        """
        return isinstance(other, self.__class__) and \
               self._strings == other.strings() and \
               self._numbers == other.numbers() and \
               self._raws == other.raws() and \
               self._iterables == other.iterables()

    def __ne__(self, other):
        """
        Enables use of != comparison, comparing instance type and all dicts for equality
        :param other: the other Bundle
        :return: False if equal, True if inequal
        """
        return not self.__eq__(other)

    def __str__(self):
        dict = {
            'strings': self._strings,
            'numbers': self._numbers,
            'raws': self._raws,
            'iterables': self._iterables
        }
        strings = json.dumps(self._strings)
        numbers = json.dumps(self._numbers)
        raws = json.dumps(self._raws)
        iterables = json.dumps(self._iterables)
        return json.dumps(dict)

    @staticmethod
    def construct_from_json(bundle_json_string):
        """
        Take a string JSON representation of a Bundle instance and construct a
        new Bundle
        :param bundle_json_string: string JSON representation
        :return: instance of Bundle()
        """
        bundle = Bundle()
        loaded = json.loads(bundle_json_string)

        new_strings = loaded['strings']
        for key in new_strings.keys():
            bundle.insert_string(key, new_strings[key])

        new_numbers = loaded['numbers']
        for key in new_numbers.keys():
            bundle.insert_number(key, new_numbers[key])

        new_raws = loaded['raws']
        for key in new_raws.keys():
            bundle.insert_raw(key, new_raws[key])

        new_iterables = loaded['iterables']
        for key in new_iterables.keys():
            bundle.insert_iterable(key, new_iterables[key])

        return bundle

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

    def merge_from_bundle(self, other_bundle):
        """
        Merges another bundle into this bundle, replacing duplicate key-value pairs with values from other_bundle
        :param other_bundle: an instance of Bundle
        """
        self._strings.update(other_bundle.strings())
        self._numbers.update(other_bundle.numbers())
        self._raws.update(other_bundle.raws())
        self._iterables.update(other_bundle.iterables())

    def strings(self, key=None):
        """
        Accessor for a specific key or the entire strings dict
        :raises BundleKeyError if the key is not none, is valid, and no key-value pair is found
        :raises AttributeError if the key is invalid
        :param key: optional
        :return: the value in the key-value pair for a given key if not None, or the entire strings dict
        """
        if key is not None and not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            try:
                return self._strings[key]
            except KeyError as err:
                raise BundleKeyError(str(err))
        else:
            return self._strings

    def numbers(self, key=None):
        """
        Accessor for a specific key or the entire numbers dict
        :raises BundleKeyError if the key is not none, is valid, and no key-value pair is found
        :raises AttributeError if the key is invalid
        :param key: optional
        :return: the value in the key-value pair for a given key if not None, or the entire numbers dict
        """
        if key is not None and not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            try:
                return self._numbers[key]
            except KeyError as err:
                raise BundleKeyError(str(err))
        else:
            return self._numbers

    def raws(self, key=None):
        """
        Accessor for a specific key or the entire raws dict
        :raises BundleKeyError if the key is not none, is valid, and no key-value pair is found
        :raises AttributeError if the key is invalid
        :param key: optional
        :return: the value in the key-value pair for a given key if not None, or the entire raws dict
        """
        if key is not None and not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            try:
                return self._raws[key]
            except KeyError as err:
                raise BundleKeyError(str(err))
        else:
            return self._raws

    def iterables(self, key=None):
        """
        Accessor for a specific key or the entire iterables dict
        :raises BundleKeyError if the key is not none, is valid, and no key-value pair is found
        :raises AttributeError if the key is invalid
        :param key: optional
        :return: the value in the key-value pair for a given key if not None, or the entire iterables dict
        """
        if key is not None and not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if key is not None:
            try:
                return self._iterables[key]
            except KeyError as err:
                raise BundleKeyError(str(err))
        else:
            return self._iterables

    def insert_string(self, key, value):
        """
        Insert a string value with a given key
        :raises AttributeError: if the key is invalid
        :raises AttributeError: if the value is not a string
        :param key: a valid key
        :param value: a string
        """
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(value, str):
            self._strings[key] = value
        else:
            raise AttributeError(self._INCORRECT_TYPE_STRING)

    def insert_number(self, key, value):
        """
        Insert a string value with a given key
        :raises AttributeError: if the key is invalid
        :raises AttributeError: if the value is not a number
        :param key: a valid key
        :param value: a number
        """
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(value, Number):
            self._numbers[key] = value
        else:
            raise AttributeError(self._INCORRECT_TYPE_NUMBER)

    def insert_raw(self, key, value):
        """
        Insert a string value with a given key
        :raises AttributeError: if the key is invalid
        :raises AttributeError: if the value is not a raw
        :param key: a valid key
        :param value: a raw
        """
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(value, str):
            raise AttributeError(self._INCORRECT_TYPE_RAW_IS_STRING)
        elif isinstance(value, Number):
            raise AttributeError(self._INCORRECT_TYPE_RAW_IS_NUMBER)
        else:
            self._raws[key] = value

    def insert_iterable(self, key, value):
        """
        Insert a string value with a given key
        :raises AttributeError: if the key is invalid
        :raises AttributeError: if the value is not an iterable
        :param key: a valid key
        :param value: an iterable
        """
        if not Bundle.is_valid_key(key):
            raise AttributeError(self._IMPROPER_KEY_FORMAT_STRING.format(key))

        if isinstance(value, collections.Iterable):
            self._iterables[key] = value
        else:
            raise AttributeError(self._INCORRECT_TYPE_ITERABLE)
