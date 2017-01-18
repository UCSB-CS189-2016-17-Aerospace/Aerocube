import unittest
from tempfile import NamedTemporaryFile
from logbook import FileHandler, Logger, TestHandler


class TestFileLogHandler(unittest.TestCase):

    def setUp(self):
        """
        Initialize Handler for arbitrary streams
        with a parameter to log to the standard output.
        Logger can be any name. The logger itself is a log channel
        or a record dispatcher. In this case, the logger is set to
        channel logs to the standard output
        :return:
        """
        # Create an arbitrary temporary file
        self._tempfile = NamedTemporaryFile(delete=True)

        # Setup logging handler for standard output purposes
        self._output_stream = FileHandler(
            filename=self._tempfile.name,
            encoding='UTF-8',
            bubble=True
        )
        self._ctx = self._output_stream.push_application()

        # Setup logging handler for testing purposes
        self._log_handler = TestHandler(bubble=True)
        self._log_handler.push_thread()

        # Initialize Logger
        # Apply both standard-output and test log-handlers to the newly, initialized logger
        self._log = Logger('TestLog')
        self._log.handlers = [self._log_handler, self._output_stream]

        # string helper
        self._tempfile_buffer = ''

    def tearDown(self):
        """
        Close the output stream.
        :return:
        """
        self._output_stream.close()
        self._log_handler.pop_thread()
        self._log_handler.close()
        self._tempfile.close()


    def test_logger_init(self):
        """
        Call all log methods
        :return:
        """
        self.assertIsNotNone(self._log, msg='Logger resource not found')

    def test_warn_logging(self):
        self._log.warn('warn000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('warn000.log' in self._tempfile_buffer)

        self._log.warn('warn001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('warn001.log' in self._tempfile_buffer)
        self.assertFalse('warn002.log' in self._tempfile_buffer)


    def test_error_logging(self):
        self._log.error('error000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('error000.log' in self._tempfile_buffer)

        self._log.error('error001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('error001.log' in self._tempfile_buffer)
        self.assertFalse('error002.log' in self._tempfile_buffer)

    def test_critical_logging(self):
        self._log.critical('critical000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('critical000.log' in self._tempfile_buffer)

        self._log.critical('critical001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('critical001.log' in self._tempfile_buffer)
        self.assertFalse('critical002.log' in self._tempfile_buffer)

    def test_debug_logging(self):
        self._log.debug('debug000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('debug000.log' in self._tempfile_buffer)

        self._log.debug('debug001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('debug001.log' in self._tempfile_buffer)
        self.assertFalse('debug002.log' in self._tempfile_buffer)

    def test_info_logging(self):
        self._log.info('info000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('info000.log' in self._tempfile_buffer)

        self._log.info('info001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('info001.log' in self._tempfile_buffer)
        self.assertFalse('info002.log' in self._tempfile_buffer)

    def test_notice_logging(self):
        self._log.notice('notice000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('notice000.log' in self._tempfile_buffer)

        self._log.notice('notice001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('notice001.log' in self._tempfile_buffer)
        self.assertFalse('notice002.log' in self._tempfile_buffer)

    def test_trace_logging(self):
        self._log.trace('trace000.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('trace000.log' in self._tempfile_buffer)

        self._log.trace('trace001.log')
        self._tempfile_buffer = str(self._tempfile.read())
        self.assertTrue('trace001.log' in self._tempfile_buffer)
        self.assertFalse('trace002.log' in self._tempfile_buffer)
