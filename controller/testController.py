import unittest
from .controller import Controller


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from logger import Logger
        Logger.prevent_external()

    def setUp(self):
        self._controller = Controller()

    def tearDown(self):
        self._controller = None

    def test_init(self):
        self.assertIsNotNone(self._controller.server)

if __name__ == '__main__':
    unittest.main()
