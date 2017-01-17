import unittest
import sys
from logbook import StreamHandler, Logger


class TestLogbook(unittest.TestCase):

    def setUp(self):
        """
        Initialize Handler for arbitrary streams
        with a parameter to log to the standard output.
        Logger can be any name. The logger itself is a log channel
        or a record dispatcher. In this case, the logger is set to
        channel logs to the standard output
        :return:
        """
        self._output_stream = StreamHandler(sys.stdout)
        self._ctx = self._output_stream.push_application()
        self._log = Logger('TestLog')

    def tearDown(self):
        """
        Close the output stream.
        :return:
        """
        self._output_stream.close()

    def test_logging_to_stdout(self):
        """
        Call all log methods
        :return:
        """
        self.assertIsNotNone(self._log, msg='Logger resource not found')
        self._log.warn('Warn')
        self._log.error('Error')
        self._log.critical('Critical Error')
        self._log.debug('Debug')
        # self._logger.exception('Exception') # Verifies an actual exception
        self._log.info('Information')
        self._log.trace('Trace')
