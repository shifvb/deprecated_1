import os
import pickle
import random
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from DIRNet_tensorflow_master.data.log import my_logger
from DIRNet_tensorflow_master.train.batches_generator import Batches, sample_pair


def my_train(config: dict):
    # 加载数据
    batch_filenames = [os.path.join(config["batch_folder"], _) for _ in os.listdir(config["batch_folder"])]
    batches = Batches(config["iteration_num"], batch_filenames)

    with tf.Session() as sess:
        reg = DIRNet(sess, config, "DIRNet", is_train=True)
        sess.run(tf.global_variables_initializer())
        for i in range(config["iteration_num"]):
            batch_x, batch_y = sample_pair(*(batches.get_batches(i)), config["batch_size"])
            loss = reg.fit(batch_x, batch_y)
            config["logger"].info("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
            if (i + 1) % 1000 == 0:
                reg.deploy(config["temp_dir"], batch_x, batch_y)
                reg.save(config["checkpoint_dir"])


def main():
    config = {
        # train batch folder
        "batch_folder": r"F:\registration_patches\向右移动11像素\train",
        # train parameters
        "image_size": [128, 128],
        "batch_size": 10,
        "learning_rate": 1e-5,
        "iteration_num": 20000,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\temp",
        # logger
        "logger_dir": r"f:\registration_running_data\log",
        "logger_name": "train.log",
    }
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    if not os.path.exists(config["logger_dir"]):
        os.makedirs(config["logger_dir"])
    config["logger"] = my_logger(folder_name=config["logger_dir"], file_name=config["logger_name"])
    my_train(config)


if __name__ == "__main__":
    main()
