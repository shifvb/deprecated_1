import os
import numpy as np
from PIL import Image
import tensorflow as tf
from random import choice, random
from DIRNet_tensorflow_master.train.train import config_folder_guard
from DIRNet_tensorflow_master.models.models import DIRNet


def mnist_train():
    config = config_folder_guard({
        "image_size": [28, 28],
        "batch_size": 10,
        "learning_rate": 1e-5,
        "iter_num": 10000,
        "save_interval": 1000,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\validate",
        "log_dir": r"F:\registration_running_data\log"
    })

    # 声明训练集和验证集
    train_workspace = r"F:\registration_patches\mnist\train"
    valid_workspace = r"F:\registration_patches\mnist\test"
    _c = {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    }
    train_list, valid_list = [], []
    for i in range(10):
        _train_path = os.path.join(train_workspace, str(i))
        _valid_path = os.path.join(valid_workspace, str(i))
        train_list.append(gen_batches(_train_path, _train_path, _c))
        valid_list.append(gen_batches(_valid_path, _valid_path, _c))

    # 构建网络
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    for iter in range(config["iter_num"]):
        _tx, _ty = sess.run(train_list[choice(range(10))])
        _train_loss = reg.fit(_tx, _ty)
        print("[TRAIN] iter={:>6d}, loss={:.6f}".format(iter + 1, _train_loss))
        if (iter + 1) % config["save_interval"] == 0:
            for i in range(10):
                _vx, _vy = sess.run(valid_list[i])
                reg.deploy(config["temp_dir"], _vx, _vy, i * config["batch_size"])
            reg.save(config["checkpoint_dir"])


def gen_batches(x_dir: str, y_dir: str, config: dict):
    """warning：本版本被修改过！使用了np.shuffle进行打乱!!! 请使用原版！！！"""
    x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir)]
    y_arr = [os.path.join(y_dir, _) for _ in os.listdir(y_dir)]
    np.random.shuffle(x_arr)
    np.random.shuffle(y_arr)
    assert len(x_arr) == len(y_arr)
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])

    def _f(input_tensor, batch_size: int, img_height: int, img_width: int, channels: int):
        _ = np.stack([np.array(Image.open(img_name)) for img_name in input_tensor], axis=0) / 255
        return _.astype(np.float32).reshape([batch_size, img_height, img_width, channels])

    batch_x = tf.py_func(_f, [batch_x, config["batch_size"], *config["image_size"], 1], tf.float32)
    batch_y = tf.py_func(_f, [batch_y, config["batch_size"], *config["image_size"], 1], tf.float32)
    return batch_x, batch_y


if __name__ == '__main__':
    mnist_train()
