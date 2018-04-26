import os
import pickle
import random
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from DIRNet_tensorflow_master.data.log import my_logger
from DIRNet_tensorflow_master.train.batches_generator import Batches, sample_pair
from learn_tensorflow.slice_input_producer_and_batch_examples.slice_input_producer_and_batch import \
    get_filenames_and_labels
from PIL import Image


def f(input_tensor):
    L = []
    for i in input_tensor:
        L.append(np.array(Image.open(i), dtype=np.uint8))
    return np.stack(L)


def my_train(config: dict):
    # 生成图片集和标签
    x_arr = r"F:\新建文件夹\resized_ct"
    x_arr = [os.path.join(x_arr, _) for _ in os.listdir(x_arr)]
    y_arr = r"F:\新建文件夹\shift_10_10_ct"
    y_arr = [os.path.join(y_arr, _) for _ in os.listdir(y_arr)]
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])
    batch_x = tf.py_func(f, [batch_x], tf.uint8)
    batch_y = tf.py_func(f, [batch_y], tf.uint8)
    batch_x = tf.reshape(batch_x, [config["batch_size"], 128, 128, 1])
    batch_y = tf.reshape(batch_y, [config["batch_size"], 128, 128, 1])
    batch_x = batch_x / 255
    batch_y = batch_y / 255

    with tf.Session() as sess:
        reg = DIRNet(sess, config, "DIRNet", is_train=True)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(config["iteration_num"]):
            _bx, _by = sess.run([batch_x, batch_y])
            loss = reg.fit(_bx, _by)
            config["logger"].info("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
            if (i + 1) % 100 == 0:
                reg.deploy(config["temp_dir"], _bx, _by)
                reg.save(config["checkpoint_dir"])
        coord.request_stop()
        coord.join(threads)


def main():
    config = {
        # train batch folder
        "batch_folder": r"F:\registration_patches\向右移动11像素\train",
        # train parameters
        "image_size": [128, 128],
        "batch_size": 10,
        "learning_rate": 1e-5,
        "iteration_num": 20000,
        "shuffle_batch": False,
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
