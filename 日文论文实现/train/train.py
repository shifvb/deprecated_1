import os
import numpy as np
import tensorflow as tf
from PIL import Image
from 日文论文实现.models.conv_regressor import ConvNetRegressor


def train():
    config_dict = config_folder_guard({
        # train parameters
        "batch_size": 10,
        "epoch_num": 100000,
        "image_size": [128, 128],
        "learning_rate": 1e-5,
        "shuffle_batch": True,

        # folder path
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "validate_dir": r"F:\registration_running_data\temp",
    })

    # 生成图片集和标签
    x_dir = r"F:\新建文件夹\resized_ct"
    y_dir = r"F:\新建文件夹\shift_10_10_ct"
    batch_x, batch_y = gen_batches(x_dir, y_dir, config_dict)

    # 构建网络
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=True, config=config_dict)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    for i in range(config_dict["epoch_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit(_bx, _by)
        print("[INFO] epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % 1000 == 0:
            reg.deploy(config_dict["validate_dir"], _bx, _by)
            reg.save(sess, config_dict["checkpoint_dir"])

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    pass  # todo: change it, do some guard things
    return config_dict


def get_validate_batches(batch_size: int):
    batch_x, batch_y = None, None  # todo change it
    return batch_x, batch_y


def gen_batches(x_dir: str, y_dir: str, config: dict):
    """
    给定x文件夹和y文件夹，生成batch tensor的函数
    :param x_dir: Moving Image文件夹绝对路径
    :param y_dir: Fixed Image 文件夹绝对路径
    :param config: config["shuffle_batch"]：是否shuffle batch
                    config["batch_size"]：batch大小
                    config["image_size"]：图像的height和width，tuple类型
    :return: Tensor('batch_x', dtype=float32, shape=[batch_size, img_height, img_width, 1])
            Tensor('batch_y', dtype=float32, shape=[batch_size, img_height, img_width, 1])
    """
    x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir)]
    y_arr = [os.path.join(y_dir, _) for _ in os.listdir(y_dir)]
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])

    def _f(input_tensor, batch_size: int, img_height: int, img_width: int, channels: int):
        _ = np.stack([np.array(Image.open(img_name)) for img_name in input_tensor], axis=0) / 255
        return _.astype(np.float32).reshape([batch_size, img_height, img_width, channels])

    batch_x = tf.py_func(_f, [batch_x, config["batch_size"], *config["image_size"], 1], tf.float32)
    batch_y = tf.py_func(_f, [batch_y, config["batch_size"], *config["image_size"], 1], tf.float32)
    return batch_x, batch_y


if __name__ == '__main__':
    train()
