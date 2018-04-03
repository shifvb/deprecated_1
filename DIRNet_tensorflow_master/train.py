import os
import random
import pickle
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models import DIRNet
from DIRNet_tensorflow_master.config import get_config
from DIRNet_tensorflow_master.data import MNISTDataHandler


def main():
    # get configure
    config = get_config(is_train=True)

    # file operations
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])

    #
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    dh = MNISTDataHandler("MNIST_data", is_train=True)

    for i in range(config["iteration_num"]):
        batch_x, batch_y = dh.sample_pair(config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        print("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


def my_train():
    """暂时往里面训练一些512x512的图像"""
    # 加载数据
    batch_xs, batch_ys = pickle.load(open(r"F:\\registration\\ct_batches.pickle", 'rb'))

    def _sample_pair(bxs, bys, batch_size: int = 64):
        _bx, _by = [], []
        for _ in range(batch_size):
            _index = random.randint(0, len(bxs) - 1)
            _ = bxs[_index]
            _ = (_ - _.min()) / (_.max() - _.min())
            _bx.append(_)
            _ = bys[_index]
            _ = (_ - _.min()) / (_.max() - _.min())
            _by.append(_)
        return np.stack(_bx, axis=0), np.stack(_by, axis=0)

    # config
    config = {
        "checkpoint_dir": "checkpoint",
        "image_size": [512, 512],
        "batch_size": 2,
        "learning_rate": 1e-4,
        "iteration_num": 10000,
        "temp_dir": "temp",
    }

    # 文件操作
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])

    # 构建网络
    print("network constructing...")
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)

    # 开始训练
    print("training...")
    for i in range(config["iteration_num"]):
        batch_x, batch_y = _sample_pair(batch_xs, batch_ys, config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        print("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % 200 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


if __name__ == "__main__":
    my_train()
    # main()
