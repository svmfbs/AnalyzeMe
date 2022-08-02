""" This is my logging program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
from configparser import ConfigParser
from enum import Enum, auto
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
sys.dont_write_bytecode = True

class HandlerType(Enum):
    ''' This is enum class '''
    FILE = auto()
    STREAM = auto()

class MyLogger():
    __DIC_LEVEL = {"debug" : DEBUG, "info" : INFO, "warning" : WARNING, "error" : ERROR, "critical" : CRITICAL}
    __LOGFORMAT = '[%(asctime)s]%(filename)s(%(lineno)d): %(message)s'
    __DATEFORMAT = '%Y-%m-%d %H:%M:%S'
    mode = 'a'

    def __init__(self, handlertype, msglevel, mode, logfile=None):
        if logfile is None:
            self.logfile = os.path.join(os.getcwd(), 'logs/test.log')
        else:
            self.logfile = logfile
        self.mode = mode
        msg_level = self.__DIC_LEVEL[msglevel.lower()]
        handler = self.__get_handler(handlertype)
        handler.setLevel(msg_level)
        handler.setFormatter(Formatter(self.__LOGFORMAT, self.__DATEFORMAT))
        self._logger = getLogger(__name__)
        self._logger.setLevel(msg_level)
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def __get_handler(self, handlertype):
        if handlertype == HandlerType.FILE:
            return FileHandler(self.logfile, self.mode)
        elif handlertype == HandlerType.STREAM:
            return StreamHandler()
        else:
            print("# Error: wrong handler type")
            sys.exit()

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)


if __name__ == '__main__':
    # logger = MyLogger(HandlerType.STREAM, 'warning', 'w')
    logger = MyLogger(HandlerType.FILE, 'warning', 'w')
    logger.debug('# 1. This is debug.')
    logger.info('# 2. This is info.')
    logger.warning('# 3. This is warning.')
    logger.error('# 4. This is error.')
    logger.critical('# 5. This is critical.')

    configParser = ConfigParser()
    configParser.read('conf/globals.cfg')
    base_log_directory = configParser.get('directories','LOG_DIRECTORY')
    log_directory = os.path.join(base_log_directory, '')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    current_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logFilename = f'{log_directory}log_{current_date}.log'
    print(logFilename)
