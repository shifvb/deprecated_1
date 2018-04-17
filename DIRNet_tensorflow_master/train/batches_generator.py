import pickle


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
