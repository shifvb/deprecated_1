import os
import numpy as np
import tensorflow as tf
from 日文论文实现.models.R1 import R1, R2, R3, ConvNetRegressor


def train():
    config_dict = config_folder_guard({
        # network settings
        "batch_size": 3,
        "img_height": 128,
        "img_width": 128,

        # train parameters
        "epoch_num": 10000,
        "learning_rate": 1e-5,

        # folder path
        "checkpoint_folder": "",  # todo remove it
        "validate_visualization_folder": "",  # todo change it
    })

    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=True, config=config_dict)
    for i in range(config_dict["epoch_num"]):
        batch_x, batch_y = get_train_batches(config_dict["batch_size"])  # todo change it
        loss = reg.fit(batch_x, batch_y)
        print("[INFO] epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % 1000 == 0:
            _validate_batch_x, _validate_batch_y = get_validate_batches(config_dict["batch_size"])
            reg.deploy(config_dict["validate_visualization_folder"], _validate_batch_x, _validate_batch_y)
            reg.save(sess, config_dict["checkpoint_folder"])


def test():
    config_dict = config_folder_guard({
        # network settings
        "batch_size": 3,
        "img_height": 512,
        "img_width": 512,

        # folder path
        "checkpoint_folder": "",  # todo remove it
        "test_visualization_folder": "",  # todo change it
    })
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=False, config=config_dict)
    reg.restore(sess, config_dict["checkpoint_folder"])
    batch_x, batch_y = get_test_batches(config_dict["batch_size"])
    reg.deploy(config_dict["test_visualization_folder"], batch_x, batch_y)


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    pass  # todo: change it, do some guard things
    return config_dict


def get_validate_batches(batch_size: int):
    batch_x, batch_y = None, None  # todo change it
    return batch_x, batch_y


def get_train_batches(batch_size: int):
    batch_x, batch_y = None, None  # todo change it
    batch_x = np.ones(shape=[batch_size, 128, 128, 1], dtype=np.float32)
    batch_y = np.zeros(shape=[batch_size, 128, 128, 1], dtype=np.float32)
    # batch_x = batch_x - batch_x.min() / (batch_x.max() - batch_x.min())
    # batch_y = batch_y - batch_y.min() / (batch_y.max() - batch_y.min())
    return batch_x, batch_y


def get_test_batches(batch_size: int):
    batch_x, batch_y = None, None  # todo change it
    return batch_x, batch_y


if __name__ == '__main__':
    train()
