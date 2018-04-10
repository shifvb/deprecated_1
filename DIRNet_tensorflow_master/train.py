import os
import random
import pickle
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from DIRNet_tensorflow_master.models import DIRNet
from DIRNet_tensorflow_master.source.log import my_logger

logger = my_logger(r"f:\train.log")


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


def my_train():
    """暂时往里面训练一些图像"""

    def _sample_pair(bxs, bys, batch_size: int = 64):
        _bx, _by = [], []
        for _ in range(batch_size):
            _index = random.randint(0, len(bxs) - 1)
            _x, _y = bxs[_index], bys[_index]
            _min, _max = min(_x.min(), _y.min()), max(_x.max(), _y.max())
            _x = (_x - _min) / (_max - _min)
            _y = (_y - _min) / (_max - _min)
            _bx.append(_x)
            _by.append(_y)
        return np.stack(_bx, axis=0), np.stack(_by, axis=0)

    # config
    config = {
        "checkpoint_dir": "checkpoint",
        "image_size": [128, 128],
        "batch_size": 80,
        "learning_rate": 1e-4,
        "iteration_num": 10000,
        "temp_dir": "temp",
    }

    # 文件操作
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])
    # 加载数据
    batch_filenames = ['F:\\registration_patches\\ct_batches_train_0.pickle',
                       'F:\\registration_patches\\ct_batches_train_1.pickle',
                       'F:\\registration_patches\\ct_batches_train_2.pickle',
                       'F:\\registration_patches\\ct_batches_train_3.pickle',
                       'F:\\registration_patches\\ct_batches_train_4.pickle',
                       'F:\\registration_patches\\ct_batches_train_5.pickle',
                       'F:\\registration_patches\\ct_batches_train_6.pickle',
                       'F:\\registration_patches\\ct_batches_train_7.pickle',
                       'F:\\registration_patches\\ct_batches_train_8.pickle',
                       'F:\\registration_patches\\ct_batches_train_9.pickle',
                       'F:\\registration_patches\\ct_batches_train_10.pickle',
                       'F:\\registration_patches\\ct_batches_train_11.pickle',
                       'F:\\registration_patches\\ct_batches_train_12.pickle',
                       'F:\\registration_patches\\ct_batches_train_13.pickle',
                       'F:\\registration_patches\\ct_batches_train_14.pickle',
                       'F:\\registration_patches\\ct_batches_train_15.pickle',
                       'F:\\registration_patches\\ct_batches_train_16.pickle',
                       'F:\\registration_patches\\ct_batches_train_17.pickle',
                       'F:\\registration_patches\\ct_batches_train_18.pickle',
                       'F:\\registration_patches\\ct_batches_train_19.pickle',
                       'F:\\registration_patches\\ct_batches_train_20.pickle',
                       'F:\\registration_patches\\ct_batches_train_21.pickle',
                       'F:\\registration_patches\\ct_batches_train_22.pickle']
    batches = Batches(config["iteration_num"], batch_filenames)

    # 构建网络
    print("network constructing...")
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)

    # 开始训练
    print("training...")
    for i in range(config["iteration_num"]):
        batch_x, batch_y = _sample_pair(*(batches.get_batches(i)), config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        logger.info("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


if __name__ == "__main__":
    my_train()
