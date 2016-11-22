import unittest
from bundle import Bundle
import pickle


class TestBundle(unittest.TestCase):

    # Keys

    def test_valid_key(self):
        self.assertEqual(Bundle.is_valid_key('VALID_KEY'), True)

    def test_invalid_key_has_lowercase(self):
        self.assertEqual(Bundle.is_valid_key('invalid_key'), False)

    def test_invalid_key_has_number(self):
        self.assertEqual(Bundle.is_valid_key('1NVAL1D_K3Y'), False)

    def test_invalid_key_has_symbol(self):
        self.assertEqual(Bundle.is_valid_key('INV@LID_KEY'), False)

    # Numbers

    def test_bundle_invalid_number_set(self):
        bundle = Bundle()
        key = 'INVALID_NUMBER'
        invalid_num = ' '
        with self.assertRaises(AttributeError):
            bundle.insert_number(key, invalid_num)

    def test_bundle_valid_number(self):
        bundle = Bundle()
        key = 'VALID_NUMBER'
        valid_num = 42
        bundle.insert_number(key, valid_num)
        self.assertEqual(bundle.numbers(key), valid_num)

    # Strings

    def test_bundle_invalid_string_set(self):
        bundle = Bundle()
        key = 'INVALID_STRING'
        invalid_string = 42
        self.assertRaises(AttributeError, bundle.insert_string, key, invalid_string)

    def test_bundle_valid_string(self):
        bundle = Bundle()
        key = 'VALID_STRING'
        valid_string = 'This is a string'
        bundle.insert_string(key, valid_string)
        self.assertEqual(bundle.strings(key), valid_string)

    # Raws

    def test_bundle_raw_not_string(self):
        bundle = Bundle()
        key = 'INVALID_RAW'
        invalid_raw = ' not raw'
        self.assertRaises(AttributeError, bundle.insert_raw, key, invalid_raw)

    def test_bundle_raw_not_number(self):
        bundle = Bundle()
        key = 'INVALID_RAW'
        invalid_raw = 0
        self.assertRaises(AttributeError, bundle.insert_raw, key, invalid_raw)

    def test_bundle_raw(self):
        bundle = Bundle()
        key = 'INVALID_RAW'
        raw_value = pickle.dumps('asdf', pickle.HIGHEST_PROTOCOL)
        bundle.insert_raw(key, raw_value)
        bundled_val = bundle.raws(key=key)
        self.assertEqual(bundled_val, raw_value)

    # Iterables

    # TODO: Iterables Testing



if __name__ == '__main__':
    unittest.main()
