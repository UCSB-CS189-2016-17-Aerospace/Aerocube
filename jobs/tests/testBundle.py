import pickle
import unittest

from jobs.bundle import Bundle, BundleKeyError


class TestBundle(unittest.TestCase):

    def setUp(self):
        self._bundle = Bundle()

    def tearDown(self):
        self._bundle = None

    @classmethod
    def setUpClass(cls):
        cls._bundle = Bundle()
        cls._messages = Bundle._ERROR_MESSAGES
        # Keys
        cls._valid_key = 'VALID_KEY'
        cls._invalid_key_has_num = 'INVAL1D_K3Y'
        cls._invalid_key_has_symbol = 'INV@lID_K3Y'
        cls._invalid_key_has_lower = 'inVALID_KEY'
        # Numbers
        cls._valid_num = 42
        cls._invalid_num = 'not a num'
        # Strings
        cls._valid_string = 'Valid string'
        cls._invalid_string = 0
        # Iterables
        cls._valid_iterable = []
        cls._valid_iterable_string = ''
        cls._invalid_iterable = 0
        # Raws
        cls._valid_raw = pickle.dumps('valid raw')
        cls._invalid_raw_string = 'invalid raw'
        cls._invalid_raw_number = 0

    @classmethod
    def tearDownClass(cls):
        cls._bundle = None
        # Reset iterable
        cls._valid_iterable = []

    # Keys

    def test_valid_key(self):
        self.assertEqual(self._bundle.is_valid_key(self._valid_key), True)

    def test_invalid_key_has_lowercase(self):
        self.assertEqual(self._bundle.is_valid_key(self._invalid_key_has_lower), False)

    def test_invalid_key_has_number(self):
        self.assertEqual(self._bundle.is_valid_key(self._invalid_key_has_num), False)

    def test_invalid_key_has_symbol(self):
        self.assertEqual(self._bundle.is_valid_key(self._invalid_key_has_symbol), False)

    def test_bundle_access_valid_key_not_found(self):
        self.assertRaises(BundleKeyError, self._bundle.strings, self._valid_key)

    def test_bundle_access_invalid_key(self):
        self.assertRaises(AttributeError, self._bundle.strings, self._invalid_key_has_lower)

    # Numbers

    def test_bundle_invalid_number_set(self):
        with self.assertRaises(AttributeError):
            self._bundle.insert_number(self._valid_key, self._invalid_num)

    def test_bundle_valid_number(self):
        self._bundle.insert_number(self._valid_key, self._valid_num)
        self.assertEqual(self._bundle.numbers(self._valid_key), self._valid_num)

    # Strings

    def test_bundle_invalid_string_set(self):
        self.assertRaises(AttributeError, self._bundle.insert_string, self._valid_key, self._invalid_string)

    def test_bundle_valid_string(self):
        self._bundle.insert_string(self._valid_key, self._valid_string)
        self.assertEqual(self._bundle.strings(self._valid_key), self._valid_string)

    # Raws

    def test_bundle_invalid_raw_string(self):
        self.assertRaises(AttributeError, self._bundle.insert_raw, self._valid_key, self._invalid_raw_string)

    def test_bundle_invalid_raw_number(self):
        self.assertRaises(AttributeError, self._bundle.insert_raw, self._valid_key, self._invalid_raw_number)

    def test_bundle_valid_raw(self):
        self._bundle.insert_raw(self._valid_key, self._valid_raw)
        bundled_val = self._bundle.raws(self._valid_key)
        self.assertEqual(bundled_val, self._valid_raw)

    # Iterables

    def test_bundle_insert_iterable(self):
        self._bundle.insert_iterable(self._valid_key, self._valid_iterable)
        self.assertEqual(self._bundle.iterables(self._valid_key), self._valid_iterable)

    def test_bundle_insert_non_iterable(self):
        with self.assertRaises(AttributeError):
            self._bundle.insert_iterable(self._valid_key, self._invalid_iterable)

    def test_bundle_insert_string_counts_as_iterable(self):
        self._bundle.insert_iterable(self._valid_key, self._valid_iterable)
        self.assertEqual(self._bundle.iterables(self._valid_key), self._valid_iterable)

    def test_bundle_update_iterable(self):
        self._bundle.insert_iterable(self._valid_key, self._valid_iterable)
        self.assertEqual(self._bundle.iterables(self._valid_key), self._valid_iterable)
        # Test updated iterable
        updated_iterable = self._valid_iterable
        updated_iterable.append('item')
        self._bundle.insert_iterable(self._valid_key, updated_iterable)
        self.assertEqual(self._bundle.iterables(self._valid_key), self._valid_iterable)



if __name__ == '__main__':
    unittest.main()
