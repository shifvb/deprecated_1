import os
import pickle
import random
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from DIRNet_tensorflow_master.data.log import my_logger
from DIRNet_tensorflow_master.train.batches_generator import Batches


def my_train():
    config = config_folder_guard({
        # train batch folder
        "batch_folder": r"F:\registration_patches\向水平或竖直方向移动8-13像素\train",
        # train parameters
        "image_size": [128, 128],
        "batch_size": 80,
        "learning_rate": 1e-5,
        "iteration_num": 20000,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\temp",
        "logger_dir": r"f:\registration_running_data\log",
        "logger_name": "train.log",
    })
    # logger
    logger = my_logger(folder_name=config["logger_dir"], file_name=config["logger_name"])

    # 加载数据
    batch_filenames = [os.path.join(config["batch_folder"], _) for _ in os.listdir(config["batch_folder"])]
    batches = Batches(config["iteration_num"], batch_filenames)

    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    for i in range(config["iteration_num"]):
        batch_x, batch_y = _sample_pair(*(batches.get_batches(i)), config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        logger.info("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


def config_folder_guard(config_dict: dict):
    """get config for training version 1"""
    if not os.path.exists(config_dict["checkpoint_dir"]):
        os.makedirs(config_dict["checkpoint_dir"])
    if not os.path.exists(config_dict["temp_dir"]):
        os.makedirs(config_dict["temp_dir"])
    return config_dict


def _sample_pair(bxs, bys, batch_size: int = 64):
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


if __name__ == "__main__":
    my_train()
