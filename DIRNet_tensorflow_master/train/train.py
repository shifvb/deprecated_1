import os
import pickle
import random
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from DIRNet_tensorflow_master.data.log import my_logger
from DIRNet_tensorflow_master.train.train_config import train_config_v1 as get_train_config
from DIRNet_tensorflow_master.train.batches_generator import Batches
from DIRNet_tensorflow_master.train.train_exchange_obj import TEO


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
            # _x = _x / 255
            # _y = _y / 255
            _bx.append(_x)
            _by.append(_y)
        return np.stack(_bx, axis=0), np.stack(_by, axis=0)

    # config
    config = get_train_config()

    # logger
    logger = my_logger(folder_name=config["logger_dir"], file_name=config["logger_name"])

    # 文件操作
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])

    # 加载数据
    batch_filenames = [os.path.join(config["batch_folder"], _) for _ in os.listdir(config["batch_folder"])]
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
        logger.info("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


if __name__ == "__main__":
    my_train()
