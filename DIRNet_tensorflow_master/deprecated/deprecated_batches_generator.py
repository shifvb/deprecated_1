import pickle
import random
import numpy as np


class Batches(object):  # 用来惰性加载batches文件的(按病人分开)
    def __init__(self, total_iter_num: int, batches_filenames: list):
        self._total_iter_num = total_iter_num
        self._batches_filenames = tuple(batches_filenames)
        self._step_length = int(self._total_iter_num / len(self._batches_filenames)) + 1
        self._curr_batches = None
        self._curr_index = None

    def get_batches(self, curr_iter_num: int):
        _index = curr_iter_num // self._step_length
        if self._curr_index != _index:
            self._curr_index = _index
            print("[INFO] lazy_loading {}...".format(self._batches_filenames[self._curr_index]))
            with open(self._batches_filenames[self._curr_index], 'rb') as f:
                self._curr_batches = pickle.load(f)
        return self._curr_batches


def sample_pair(bxs, bys, batch_size: int = 64):
    _bx, _by = [], []
    for _ in range(batch_size):
        _index = random.randint(0, len(bxs) - 1)
        _x, _y = bxs[_index], bys[_index]
        _min, _max = min(_x.min(), _y.min()), max(_x.max(), _y.max())
        _x = (_x - _min) / (_max - _min)
        _y = (_y - _min) / (_max - _min)
        # _x = _x / 255
        # _y = _y / 255
        _bx.append(_x)
        _by.append(_y)
    return np.stack(_bx, axis=0), np.stack(_by, axis=0)
