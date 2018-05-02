import os
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from PIL import Image


def train():
    config = config_folder_guard({
        # train parameters
        "image_size": [128, 128],
        "batch_size": 80,
        "learning_rate": 1e-5,
        "iteration_num": 20000,
        "shuffle_batch": True,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\temp",
    })
    # 生成图片集和标签
    batch_x_dir = r"F:\registration_patches\train\resized_ct_image"
    batch_y_dir = r"F:\registration_patches\train\shift_10_10_ct_image"
    batch_x, batch_y = gen_batches(batch_x_dir, batch_y_dir, config)
    valid_x_dir = r"F:\registration_patches\validate\resized_ct"
    valid_y_dir = r"F:\registration_patches\validate\shift_10_10_ct"
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, config)
    # 开始训练
    with tf.Session() as sess:
        reg = DIRNet(sess, config, "DIRNet", is_train=True)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(config["iteration_num"]):
            _bx, _by = sess.run([batch_x, batch_y])
            loss = reg.fit(_bx, _by)
            config["logger"].info("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
            if (i + 1) % 1000 == 0:
                _valid_x, _valid_y = sess.run([valid_x, valid_y])
                reg.deploy(config["temp_dir"], _valid_x, _valid_y)
                reg.save(config["checkpoint_dir"])
        coord.request_stop()
        coord.join(threads)


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


def config_folder_guard(config: dict):
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    return config


if __name__ == "__main__":
    train()
