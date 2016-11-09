import unittest
from commClass import FirebaseComm

class TestFirebaseComm(unittest.TestCase):
    def test_write(self):
        commTest=FirebaseComm(True)
        self.assertEqual(commTest.write('/test','this is a test'),{'1':'this is a test'})

    def test_read(self):
        commTest = FirebaseComm(True)
        self.assertEqual(commTest.read('/test',None),{'1':'this is a test'})
if __name__ == '__main__':
    unittest.main()