import os
import logging
import sys


def my_logger(folder_name: str, file_name: str):
    # get logger
    _logger = logging.getLogger(os.path.join(folder_name, file_name))
    _logger.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(message)s')
    # console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    # file handler
    if not os.path.exists(folder_name):
        print("[DEBUG] Folder \"{}\" does not exist. create".format(folder_name), file=sys.stdout)
        os.makedirs(folder_name)
    file_handler = logging.FileHandler(os.path.join(folder_name, file_name))
    file_handler.setFormatter(formatter)
    # add the handler to the root logger
    _logger.addHandler(console)
    _logger.addHandler(file_handler)
    return _logger


class LossRecorder(object):
    def __init__(self):
        self.L1 = list()
        self.L2 = list()
        self.L3 = list()

    def record_loss(self, total_loss, ncc_loss, grad_loss):
        self.L1.append(total_loss)
        self.L2.append(ncc_loss)
        self.L3.append(grad_loss)

    def get_avg_losses(self):
        return (
            sum(self.L1) / len(self.L1),
            sum(self.L2) / len(self.L2),
            sum(self.L3) / len(self.L3),
        )
