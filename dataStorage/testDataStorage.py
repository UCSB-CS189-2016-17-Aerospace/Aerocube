import os
import json
import unittest
import dataStorage


class TestDataStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    test = json.dumps([1, [[1, 2, 3],
                           [3, 2, 1],
                           [2, 1, 1]]])

    def test_store(self):
        dataStorage.store(location="test.p", pickled=self.test)
        self.assertTrue(os.path.isfile('test.p'))

    def test_retrieve(self):
        result = dataStorage.retrieve(location="test.p")
        self.assertEquals(result, self.test)
