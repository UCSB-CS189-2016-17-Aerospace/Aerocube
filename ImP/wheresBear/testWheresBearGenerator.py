import unittest
import wheresBearGenerator

class TestSubFunctions(unittest.TestCase):
    def testCreateImage(self):
        wheresBearGenerator.createImage(wheresBearGenerator.Image.open('thing.png'),wheresBearGenerator.Image.new("RGB",(1000,1000)),'test.png')