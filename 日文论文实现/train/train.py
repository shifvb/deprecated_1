import os
import numpy as np
import tensorflow as tf
from PIL import Image
from 日文论文实现.models.conv_regressor import ConvNetRegressor


def train():
    config = config_folder_guard({
        # train parameters
        "batch_size": 10,
        "epoch_num": 1000,
        "save_interval": 1000,
        "image_size": [128, 128],
        "learning_rate": 1e-5,
        "shuffle_batch": True,

        # folder path
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "valid_in_x_dir": r"F:\registration_patches\version_all\test\normolized_pt",
        "valid_in_y_dir": r"F:\registration_patches\version_all\test\resized_ct",
        "valid_out_dir_all": r"F:\registration_running_data\validate",
        "valid_out_dir_1": r"F:\registration_running_data\validate_1",
        "valid_out_dir_2": r"F:\registration_running_data\validate_2",
        "valid_out_dir_3": r"F:\registration_running_data\validate_3",
    })

    # 生成图片集
    batch_x_dir = r"F:\registration_patches\version_all\train\normalized_pt"
    batch_y_dir = r"F:\registration_patches\version_all\train\resized_ct"
    batch_x, batch_y = gen_batches(batch_x_dir, batch_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    })

    # 生成验证集
    valid_x_1, valid_y_1 = gen_batches(config["valid_in_x_dir"], config["valid_in_y_dir"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    valid_x_2, valid_y_2 = gen_batches(config["valid_in_x_dir"], config["valid_in_y_dir"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    valid_x_3, valid_y_3 = gen_batches(config["valid_in_x_dir"], config["valid_in_y_dir"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    valid_x, valid_y = gen_batches(config["valid_in_x_dir"], config["valid_in_y_dir"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })

    # 构建网络
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=True, config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Captain on the bridge!

    # 单独训练R1
    for i in range(config["epoch_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit_only_r1(_bx, _by)
        print("[INFO] (R1) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            reg.save(sess, config["checkpoint_dir"])
            for j in range(len(os.listdir(config["valid_in_x_dir"])) // config["batch_size"]):
                _vx_1, _vy_1 = sess.run([valid_x_1, valid_y_1])
                reg.deploy(config["valid_out_dir_1"], _vx_1, _vy_1, j * config["batch_size"])

    # 单独训练R2
    for i in range(config["epoch_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit_only_r2(_bx, _by)
        print("[INFO] (R2) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            reg.save(sess, config["checkpoint_dir"])
            for j in range(len(os.listdir(config["valid_in_x_dir"])) // config["batch_size"]):
                _vx_2, _vy_2 = sess.run([valid_x_2, valid_y_2])
                reg.deploy(config["valid_out_dir_2"], _vx_2, _vy_2, j * config["batch_size"])

    # 单独训练R3
    for i in range(config["epoch_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit_only_r3(_bx, _by)
        print("[INFO] (R3) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            reg.save(sess, config["checkpoint_dir"])
            for j in range(len(os.listdir(config["valid_in_x_dir"])) // config["batch_size"]):
                _vx_3, _vy_3 = sess.run([valid_x_3, valid_y_3])
                reg.deploy(config["valid_out_dir_3"], _vx_3, _vy_3, j * config["batch_size"])

    # 再统一训练R1 + R2 + R3
    for i in range(config["epoch_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit(_bx, _by)
        print("[INFO] epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            reg.save(sess, config["checkpoint_dir"])
            for j in range(len(os.listdir(config["valid_in_x_dir"])) // config["batch_size"]):
                _vx, _vy = sess.run([valid_x, valid_y])
                reg.deploy(config["valid_out_dir_all"], _vx, _vy, j * config["batch_size"])

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    if not os.path.exists(config_dict["checkpoint_dir"]):
        os.makedirs(config_dict["checkpoint_dir"])
    if not os.path.exists(config_dict["valid_out_dir_all"]):
        os.makedirs(config_dict["valid_out_dir_all"])
    if not os.path.exists(config_dict["valid_out_dir_1"]):
        os.makedirs(config_dict["valid_out_dir_1"])
    if not os.path.exists(config_dict["valid_out_dir_2"]):
        os.makedirs(config_dict["valid_out_dir_2"])
    if not os.path.exists(config_dict["valid_out_dir_3"]):
        os.makedirs(config_dict["valid_out_dir_3"])
    return config_dict


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
