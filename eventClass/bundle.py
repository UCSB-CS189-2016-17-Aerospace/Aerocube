from numbers import Number
import collections


class Bundle(object):
    _strings = {}
    _numbers = {}
    _raws = {}
    _iterables = {}

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
        if key is not None:
            return self._strings[key]
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            return self._strings

    def numbers(self, key=None):
        if key is not None:
            return self._numbers[key]
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            return self._numbers

    def raws(self, key=None):
        if key is not None:
            return self._raws[key]
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            return self._raws

    def iterables(self, key):
        if key is not None:
            return self._iterables[key]
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            return self._iterables

    def insert_string(self, key, s):
        if isinstance(s, str):
            self._strings[key] = s
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            raise AttributeError('Not a string')


    def insert_number(self, key, num):
        if isinstance(num, Number):
            self._numbers[key] = num
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            raise AttributeError('Not a number')

    def insert_raw(self, key, data):
        if isinstance(data, str):
            raise AttributeError('Not raw data, found string')
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        elif isinstance(data, Number):
            raise AttributeError('Not raw data, found number')
        else:
            self._raws[key] = data

    def insert_iterable(self, key, iter):
        if isinstance(iter, collections.Iterable):
            self._iterables[key] = iter
        elif not Bundle.is_valid_key(key):
            raise AttributeError("{} is improperly formatted".format(key))
        else:
            raise AttributeError('Not an iterable')

