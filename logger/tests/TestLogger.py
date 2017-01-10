from logbook import Logger
import unittest
#from ../Logger import Logger

class TestLogger(unittest.TestCase):
  
  def setUp(self):
    # Declare the Logger
    self._log_handler = logbook.TestHandler()
    self._log_handler.push_thread()
    self._logger = Logger('TestLogger')

  def tearDown(self):
    self._log_handler.pop_thread()
    self._logger = None
    
  #@classmethod
  #def instantiateState(logger):
  
  
  def testLoggerWithWarning():
    with logbook.TestHandler() as handler:
      _logger.warn('A warning')
      assert logger.has_warning('A warning') 
