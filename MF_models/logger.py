import os
import sys
import json 
import time
import tempfile
from collections import defaultdict

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class Logger:
    DEFAULT = None # a logger with no output files
    CURRENT = None # current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm




def log(*args, level=INFO):
    """
    write the sequence of args to the console and output files
    """
    get_current().log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

def set_level(level):
    get_current().set_level(level)

def get_dir():
    return get_current().get_dir()


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT 

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT 

def configure(dir=None, format_strs=None, comm=None):
    if not dir:
        dir = os.getenv('LOGDIR')
    if not dir:
        dir = os.path.join(tempfile.gettempdir(),
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_suffix = ''
    rank = 0

    if not format_strs:
        
        format_strs = os.getenv('LOG_FORMAT', 'stdout,log,csv').split(',')

    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    log('Logging to %s'%dir)
