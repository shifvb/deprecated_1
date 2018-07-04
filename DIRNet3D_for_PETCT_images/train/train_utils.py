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
    """用来记录loss的便捷类"""

    def __init__(self):
        self.L1 = list()
        self.L2 = list()
        self.L3 = list()

    def _clear(self):
        self.L1.clear()
        self.L2.clear()
        self.L3.clear()

    def record_loss(self, total_loss, ncc_loss, grad_loss):
        """添加loss"""
        self.L1.append(total_loss)
        self.L2.append(ncc_loss)
        self.L3.append(grad_loss)

    def get_losses(self):
        """返回平均loss， *同时清除当前已有loss记录*"""
        _losses = (sum(self.L1) / len(self.L1), sum(self.L2) / len(self.L2), sum(self.L3) / len(self.L3))
        self._clear()
        return _losses
