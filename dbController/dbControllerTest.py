from dbControllerConfig import generate_config, generate_data, parse_config
import unittest
import json

class TestdbControllerConfig(unittest.TestCase):
    config = {'location': 'test', 'method': 'firebase'}

    def test_generate_config(self):
        self.assertEqual(generate_config('test','firebase'),json.dumps(self.config))

    def test_parse_config(self):
        self.assertEqual(parse_config(json.dumps(self.config)),('test','firebase'))

    def test_generate_data(self):
        # TODO: define how data should look
        self.assertEqual(0,1)

