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
            self.firebase.authentication=firebase.FirebaseAuthentication(secret='GHtobDkSPrtoOtVAcPR4OF7dBXzMBEPAH5UALw45',email='yourfirenation@gmail.com')
        else:
            self.firebase = firebase.FirebaseApplication('https://yfn-aerospace.firebaseio.com/', authentication=None)
            self.firebase.authentication=firebase.FirebaseAuthentication(secret='WaPfb7ZK3nFH1RDBUzL71sPIr0LJGp9JSGKE0u1B',email='yourfirenation@gmail.com')
        print(self.firebase.authentication.extra)

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
