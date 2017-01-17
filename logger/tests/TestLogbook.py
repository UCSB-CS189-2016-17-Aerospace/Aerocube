import unittest
import sys
from logbook import StreamHandler, Logger, TestHandler


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
        # Setup logging handler for standard output purposes
        self._output_stream = StreamHandler(sys.stdout)
        self._ctx = self._output_stream.push_application()

        # Setup logging handler for testing purposes
        self._log_handler = TestHandler(bubble=True)
        self._log_handler.push_thread()

        # Initialize Logger
        # Apply both standard-output and test log-handlers to the newly, initialized logger
        self._log = Logger('TestLog')
        self._log.handlers = [self._log_handler, self._output_stream]

    def tearDown(self):
        """
        Close the output stream.
        :return:
        """
        self._output_stream.close()
        self._log_handler.pop_thread()
        self._log_handler.close()

    def test_logger_init(self):
        """
        Call all log methods
        :return:
        """
        self.assertIsNotNone(self._log, msg='Logger resource not found')

    def test_warn_logging(self):
        self.assertFalse(self._log_handler.has_warning('warn000.log'))
        self._log.warn('warn000.log')
        self.assertTrue(self._log_handler.has_warning)
        self._log.warn('warn001.log')
        self.assertTrue(self._log_handler.has_warning('warn000.log'))
        self.assertTrue(self._log_handler.has_warning('warn001.log'))
        self.assertFalse(self._log_handler.has_warning('warn002.log'))

    def test_error_logging(self):
        self.assertFalse(self._log_handler.has_errors)
        self.assertFalse(self._log_handler.has_error('error000.log'))
        self._log.error('error000.log')
        self.assertTrue(self._log_handler.has_error)
        self._log.error('error001.log')
        self.assertTrue(self._log_handler.has_error('error000.log'))
        self.assertTrue(self._log_handler.has_error('error001.log'))
        self.assertFalse(self._log_handler.has_error('error002.log'))
        self.assertTrue(self._log_handler.has_errors)

    def test_critical_logging(self):
        self.assertFalse(self._log_handler.has_criticals)
        self.assertFalse(self._log_handler.has_critical('critical000.log'))
        self._log.critical('critical000.log')
        self.assertTrue(self._log_handler.has_error)
        self._log.critical('critical001.log')
        self.assertTrue(self._log_handler.has_critical('critical000.log'))
        self.assertTrue(self._log_handler.has_critical('critical001.log'))
        self.assertFalse(self._log_handler.has_critical('critical002.log'))
        self.assertTrue(self._log_handler.has_criticals)

    def test_debug_logging(self):
        self.assertFalse(self._log_handler.has_debugs)
        self.assertFalse(self._log_handler.has_debug('debug000.log'))
        self._log.debug('debug000.log')
        self.assertTrue(self._log_handler.has_debug)
        self._log.debug('debug001.log')
        self.assertTrue(self._log_handler.has_debug('debug000.log'))
        self.assertTrue(self._log_handler.has_debug('debug001.log'))
        self.assertFalse(self._log_handler.has_debug('debug002.log'))
        self.assertTrue(self._log_handler.has_debugs)

    def test_info_logging(self):
        self.assertFalse(self._log_handler.has_infos)
        self.assertFalse(self._log_handler.has_info('info000.log'))
        self._log.info('info000.log')
        self.assertTrue(self._log_handler.has_info)
        self._log.info('info001.log')
        self.assertTrue(self._log_handler.has_info('info000.log'))
        self.assertTrue(self._log_handler.has_info('info001.log'))
        self.assertFalse(self._log_handler.has_info('info002.log'))
        self.assertTrue(self._log_handler.has_infos)

    def test_notice_logging(self):
        self.assertFalse(self._log_handler.has_notices)
        self.assertFalse(self._log_handler.has_notice('notice000.log'))
        self._log.notice('notice000.log')
        self.assertTrue(self._log_handler.has_notice)
        self._log.notice('notice001.log')
        self.assertTrue(self._log_handler.has_notice('notice000.log'))
        self.assertTrue(self._log_handler.has_notice('notice001.log'))
        self.assertFalse(self._log_handler.has_notice('notice002.log'))
        self.assertTrue(self._log_handler.has_notices)

    @unittest.skip('Trace test is skipped, because the log handler has no `has_trace` field\n')
    def test_trace_logging(self):
        self._log.trace('Trace')
