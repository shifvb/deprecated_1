import logging
import time
import sys


def my_logger(log_filename: str):
    # get logger
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(message)s')
    # console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    # file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    # add the handler to the root logger
    _logger.addHandler(console)
    _logger.addHandler(file_handler)
    return _logger


if __name__ == '__main__':
    lg = my_logger("f:\\out.log")
    for i in range(10):
        lg.info("[INFO] [{}] blablalb".format(time.asctime()))
        time.sleep(0.4)
