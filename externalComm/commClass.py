from abc import ABCMeta, abstractmethod
import pyrebase


class Comm():
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, location, id): pass

    @abstractmethod
    def write(self, location, id, data): pass

    @abstractmethod
    def delete(self, location, id): pass

    @abstractmethod
    def imageStore(self, id, srcImage): pass

    @abstractmethod
    def imageDownload(self, id): pass


class FirebaseComm(Comm):
    # using secret token as authentication but if we want to change to using login use this instead
    # user=self.auth.sign_in_with_email_and_password('yourfirenation@gmail.com','yourfirenation')
    def __init__(self, testing=False):
        if testing:
            config = {
                "apiKey": "AIzaSyC9IG_3k-6pISqS1HO82GPVqm4bOo_aVb0",
                "authDomain": " yfn-aerospace-staging.firebaseapp.com",
                "databaseURL": "https://yfn-aerospace-staging.firebaseio.com",
                "storageBucket": "yfn-aerospace-staging.appspot.com"
            }
            self.token = 'WaPfb7ZK3nFH1RDBUzL71sPIr0LJGp9JSGKE0u1B'
        else:
            config = {
                "apiKey": "AIzaSyDAzrKDM0Mjw20BiQKSyL3G09cUUTDXTjE",
                "authDomain": " yfn-aerospace.firebaseapp.com",
                "databaseURL": "https://yfn-aerospace.firebaseio.com",
                "storageBucket": "yfn-aerospace.appspot.com"
            }
            self.token = 'GHtobDkSPrtoOtVAcPR4OF7dBXzMBEPAH5UALw45'
        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()
        self.storage = self.firebase.storage()

    def read(self, location, id):
        '''
        :param location: where it is stored
        :param id: ID what you are looking for
        :return: data at location or none
        '''
        result = self.db.child(location).child(id).get(self.token)
        print(result.val())
        return result.val()

    def write(self, location,id, data):
        '''
        :param location: where to store
        :param id: name of scan
        :param data: all data
        :return:
        '''
        if location is None:
            result = self.db.child(id).set(data=data, token=self.token)
        else:
            result = self.db.child(location).child(id).set(data=data, token=self.token)

    def delete(self, location, id):
        '''
        :param location:
        :param id:
        :return:
        '''
        self.db.child(location).child(id).remove(token=self.token)
    def imageStore(self,id,srcImage):
        '''
        :param id: id of scan
        :param srcImage: location of source image
        :return:
        '''
        auth = self.firebase.auth()
        user = auth.sign_in_with_email_and_password('yourfirenation@gmail.com', 'yourfirenation')
        self.storage.child('image').child(id+'.jpg').put(srcImage, token=user['idToken'])

    def imageDownload(self, id):
        auth = self.firebase.auth()
        user = auth.sign_in_with_email_and_password('yourfirenation@gmail.com', 'yourfirenation')
        print(self.storage.child('images').get_url(user['idToken']))
        # prints out url to image
        # print(self.storage.child('images/test.jpg').get_url(self.token))
        self.storage.child('image/test.jpg').download('downloaded.jpg', user['idToken'])
