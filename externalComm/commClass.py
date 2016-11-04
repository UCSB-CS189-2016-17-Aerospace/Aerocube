from abc import ABCMeta, abstractmethod
class Comm:
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self): pass

    def write(self): pass


class FirebaseComm(Comm):
    def read(self): pass

    def write(self): pass