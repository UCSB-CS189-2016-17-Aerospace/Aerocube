import unittest
import sys
from logbook import StreamHandler, Logger


class TestLogbook(unittest.TestCase):

  def setUp(self):
    self._output_stream = StreamHandler(sys.stdout)
    self._ctx = self._output_stream.push_application()
    self._logger = Logger('TestLog')

  def test_logging_to_stdout(self):
    '''
      Call all log methods and review if the logs are
      outputted to the standard output.
    :return: None
    '''
    self.assertIsNotNone(self._logger)
    self._logger.warn('Warn')
    self._logger.error('Error')
    self._logger.critical('Critical Error')
    self._logger.debug('Debug')
    #self._logger.exception('Exception') # Verifies an actual exception
    self._logger.info('Information')
    self._logger.trace('Trace')