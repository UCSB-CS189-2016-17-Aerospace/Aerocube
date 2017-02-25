import unittest
from externalComm.commClass import FirebaseComm
from externalComm.externalComm import *
import os.path


class TestFirebaseComm(unittest.TestCase):
    commTest = FirebaseComm(True)

    def test_write(self):
        self.commTest.write('test', '2', 'this is a test2')
        self.assertEqual(self.commTest.read('test', '2'), 'this is a test2')
        # testing delete
        self.commTest.delete('test', '2')
        self.assertIsNone(self.commTest.read('test', '2'))

    def test_read(self):
        self.assertEqual(self.commTest.read('test', '1'), 'this is a test')

    def test_imageStore(self):
        self.commTest.imageStore('test', 'externalComm/testimage.jpg')

    def test_imageDownload(self):
        self.commTest.imageDownload('test')
        self.assertTrue(os.path.isfile('externalComm/test.jpg'))


class TestExternalComm(unittest.TestCase):

    def test_process_store(self):
        external_write(database=FirebaseComm.NAME, location='test', scanID='4', data='this is a process test2', testing=True)
        self.assertEqual(external_read(database=FirebaseComm.NAME, location='test', scanID='4', testing=True), 'this is a process test2')
        # testing process delete
        external_delete(database=FirebaseComm.NAME, location='test', scanID='4', testing=True)
        self.assertIsNone(external_read(database=FirebaseComm.NAME, location='test', scanID='4', testing=True))

    def test_process_read(self):
        self.assertEqual(external_read(database=FirebaseComm.NAME, location='test', scanID='3', testing=True), 'this is a process test')

    def test_process_storeImage(self):
        external_store_img(database=FirebaseComm.NAME, scanID='processtest', srcImage='testimage.jpg', testing=True)

if __name__ == '__main__':
    unittest.main()
