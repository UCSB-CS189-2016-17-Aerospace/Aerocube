from abc import ABCMeta, abstractmethod
from firebase import firebase
class Comm():
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self): pass

    def write(self): pass


class FirebaseComm(Comm):
    firebase = firebase.FirebaseApplication()
    def read(self, location, data):
        '''
        :param location: where it is stored
        :param data: ID what you are looking fore
        :return:
        '''
        pass

    def write(self, location, data):
        '''
        :param location: where to store
        :param data: all data
        :return:
        '''
        pass