from abc import ABCMeta, abstractmethod
from firebase import firebase
class Comm():
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self): pass

    def write(self): pass


class FirebaseComm(Comm):
    def __init__(self,testing=False):
        if testing:
            self.firebase = firebase.FirebaseApplication('https://yfn-aerospace-staging.firebaseio.com/', authentication=None)
        else:
            self.firebase = firebase.FirebaseApplication('https://yfn-aerospace.firebaseio.com/', authentication=None)
        self.firebase.authentication=firebase.FirebaseAuthentication(secret='i have no idea',email='alexthielk@gmail.com')
    def read(self, location, id):
        '''
        :param location: where it is stored
        :param id: ID what you are looking for
        :return:
        '''
        result = self.firebase.get(location,id)
        return result

    def write(self, location, data):
        '''
        :param location: where to store
        :param data: all data
        :return:
        '''
        result = self.firebase.post(url=location, data=data)
        print(result)
        return result
