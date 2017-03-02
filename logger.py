import time
from enum import Enum

# This should not be changed in any commit, only used locally
global_log_disable = False


class LogType(Enum):
    success = 0
    warning = 1
    error = 2
    debug = 3

    def __init__(self, type_id):
        self._type_strings = {
            0: 'Success',
            1: 'Warning',
            2: 'Error',
            3: 'Debug'
        }
        self._type_id = type_id

    def __str__(self):
        return self._type_strings[self._type_id]


class Logger:
    """
    A logger class to be instantiated at the top of each non-test python module for debugging purposes
    :ivar _filename:
    :ivar _active:
    :ivar _external: Whether or not to save logs to an external location
    :cvar _prevent_external: Prevent any logs from being sent to firebase if true
    """
    _prevent_external = False

    def __init__(self, filename, active=False, external=True):
        self._filename = filename
        self._active = active
        self._external = external

    def _log(self, log_type, class_name=None, func_name='', msg='', id=None):
        if self._active and not global_log_disable:
            log_statement = '{}: {}'.format(log_type, self._filename)
            if class_name is not None:
                log_statement += '.{}'.format(class_name)
            log_statement += '.{}'.format(func_name)
            log_statement += ': {}'.format(msg)
            print(log_statement)
            if id is not None and self._external and not self._prevent_external:
                # Defer imports until needed to handle circular dependency issue
                from externalComm.externalComm import external_write
                from externalComm.commClass import FirebaseComm
                external_write(FirebaseComm.NAME, scanID=str(time.time()).split('.')[0], location='logs/{}'.format(id), data=log_statement, testing=True)

    @classmethod
    def prevent_external(cls):
        cls._prevent_external = True

    def err(self, class_name, func_name, msg, id):
        self._log(LogType.error, class_name, func_name, msg, id)

    def success(self, class_name, func_name, msg, id):
        self._log(LogType.success, class_name, func_name, msg, id)

    def warn(self, class_name, func_name, msg, id):
        self._log(LogType.warning, class_name, func_name, msg, id)

    def debug(self, class_name, func_name, msg, id):
        self._log(LogType.debug, class_name, func_name, msg, id)

    def enable(self):
        self._active = True

    def disable(self):
        self._active = False
