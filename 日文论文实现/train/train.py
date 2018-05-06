import os
import numpy as np
import tensorflow as tf
from PIL import Image
from 日文论文实现.models.conv_regressor import ConvNetRegressor
from 日文论文实现.train.config_folder_guard import config_folder_guard


def train():
    # 设置参数
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
        # R1 path
        "train_in_x_dir_1": r"F:\registration_patches\version_all\train\normalized_pt",
        "train_in_y_dir_1": r"F:\registration_patches\version_all\train\resized_ct",
        "valid_in_x_dir_1": r"F:\registration_patches\version_all\train\normalized_pt",
        "valid_in_y_dir_1": r"F:\registration_patches\version_all\train\resized_ct",
        "valid_out_dir_1": r"F:\registration_running_data\validate_1",
        # R2 path
        "train_in_x_dir_2": r"F:\registration_running_data\validate_1",
        "train_in_y_dir_2": r"F:\registration_patches\version_all\train\resized_ct",
        "valid_in_x_dir_2": r"F:\registration_running_data\validate_1",
        "valid_in_y_dir_2": r"F:\registration_patches\version_all\train\resized_ct",
        "valid_out_dir_2": r"F:\registration_running_data\validate_2",
        # R3 path
        "train_in_x_dir_3": r"F:\registration_running_data\validate_2",
        "train_in_y_dir_3": r"F:\registration_patches\version_all\train\resized_ct",
        # "valid_in_x_dir_3": r"F:\registration_running_data\validate_2",
        "valid_in_x_dir_3": r"F:\registration_patches\version_all\test\normalized_pt",
        "valid_in_y_dir_3": r"F:\registration_patches\version_all\test\resized_ct",
        "valid_out_dir_3": r"F:\registration_running_data\validate_3",
        # # R1, R2, R3 path
        # "train_in_x_dir_all": r"F:\registration_running_data\validate_3",
        # "train_in_y_dir_all": r"F:\registration_patches\version_all\train\resized_ct",
        # "valid_in_x_dir_all": r"F:\registration_running_data\validate_3",
        # "valid_in_y_dir_all": r"F:\registration_patches\version_all\train\resized_ct",
        # "valid_out_dir_all": r"F:\registration_running_data\validate_all",

    })
    valid_iter_num = len(os.listdir(config["valid_in_y_dir_1"])) // config["batch_size"]
    test_iter_num = len(os.listdir(config["valid_in_x_dir_3"])) // config["batch_size"]

    # 生成图片集

    # 构建网络
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=True, config=config)

    # Captain on the bridge!

    # 单独训练R1
    train_x_1, train_y_1 = gen_batches(config["train_in_x_dir_1"], config["train_in_y_dir_1"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    })
    valid_x_1, valid_y_1 = gen_batches(config["valid_in_x_dir_1"], config["valid_in_y_dir_1"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    coord_1 = tf.train.Coordinator()
    threads_1 = tf.train.start_queue_runners(sess=sess, coord=coord_1)
    for i in range(config["epoch_num"]):
        _tx_1, _ty_1 = sess.run([train_x_1, train_y_1])
        loss = reg.fit_only_r1(_tx_1, _ty_1)
        print("[INFO] (R1) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            # reg.save(sess, config["checkpoint_dir"])
            pass
    for j in range(valid_iter_num):
        _vx_1, _vy_1 = sess.run([valid_x_1, valid_y_1])
        reg.deploy(config["valid_out_dir_1"], _vx_1, _vy_1, j * config["batch_size"])

    # 单独训练R2
    train_x_2, train_y_2 = gen_batches(config["train_in_x_dir_2"], config["train_in_y_dir_2"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True,
    }, img_filter_func=lambda _: "z1" in _)  # R2训练时，使用R1生成的配准结果作为待配准图像训练
    valid_x_2, valid_y_2 = gen_batches(config["valid_in_x_dir_2"], config["valid_in_y_dir_2"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    }, img_filter_func=lambda _: "z1" in _)  # R2生成的结果的待配准图像，是基于R1生成的配准结果
    coord_2 = tf.train.Coordinator()
    threads_2 = tf.train.start_queue_runners(sess=sess, coord=coord_2)
    for i in range(config["epoch_num"]):
        _tx_2, _ty_2 = sess.run([train_x_2, train_y_2])
        loss = reg.fit_only_r2(_tx_2, _ty_2)
        print("[INFO] (R2) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            # reg.save(sess, config["checkpoint_dir"])
            pass
    for j in range(valid_iter_num):
        _vx_2, _vy_2 = sess.run([valid_x_2, valid_y_2])
        reg.deploy(config["valid_out_dir_2"], _vx_2, _vy_2, j * config["batch_size"])

    # 单独训练R3
    train_x_3, train_y_3 = gen_batches(config["train_in_x_dir_3"], config["train_in_y_dir_3"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    }, img_filter_func=lambda _: "z2" in _)  # R3训练时，使用R2生成的配准结果作为待配准图像训练
    valid_x_3, valid_y_3 = gen_batches(config["valid_in_x_dir_3"], config["valid_in_y_dir_3"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })  # R3使用测试图像生成配准结果
    coord_3 = tf.train.Coordinator()
    threads_3 = tf.train.start_queue_runners(sess=sess, coord=coord_3)
    for i in range(config["epoch_num"]):
        _tx_3, _ty_3 = sess.run([train_x_3, train_y_3])
        loss = reg.fit_only_r3(_tx_3, _ty_3)
        print("[INFO] (R3) epoch={:>5}, loss={:.3f}".format(i, loss))
        if (i + 1) % config["save_interval"] == 0:
            # reg.save(sess, config["checkpoint_dir"])
            pass
    for j in range(test_iter_num):
        _vx_3, _vy_3 = sess.run([valid_x_3, valid_y_3])
        reg.deploy(config["valid_out_dir_3"], _vx_3, _vy_3, j * config["batch_size"])

    # # 再统一训练R1 + R2 + R3
    # train_x_all, train_y_all = gen_batches(config["train_in_x_dir_all"], config["train_in_y_dir_all"], {
    #     "batch_size": config["batch_size"],
    #     "image_size": config["image_size"],
    #     "shuffle_batch": True
    # })
    # valid_x_all, valid_y_all = gen_batches(config["valid_in_x_dir_all"], config["valid_in_y_dir_all"], {
    #     "batch_size": config["batch_size"],
    #     "image_size": config["image_size"],
    #     "shuffle_batch": False
    # })
    # coord_all = tf.train.Coordinator()
    # threads_all = tf.train.start_queue_runners(sess=sess, coord=coord_all)
    # for i in range(config["epoch_num"]):
    #     _tx_all, _ty_all = sess.run([train_x_all, train_y_all])
    #     loss = reg.fit(_tx_all, _ty_all)
    #     print("[INFO] epoch={:>5}, loss={:.3f}".format(i, loss))
    #     if (i + 1) % config["save_interval"] == 0:
    #         # reg.save(sess, config["checkpoint_dir"])
    #         pass
    # for j in range(valid_iter_num):
    #     _vx_all, _vy_all = sess.run([valid_x_all, valid_y_all])
    #     reg.deploy(config["valid_out_dir_all"], _vx_all, _vy_all, j * config["batch_size"])

    # 回收资源
    coord_1.request_stop()
    coord_1.join(threads_1)
    coord_2.request_stop()
    coord_2.join(threads_2)
    coord_3.request_stop()
    coord_3.join(threads_3)
    # coord_all.request_stop()
    # coord_all.join(threads_all)
    sess.close()


def gen_batches(x_dir: str, y_dir: str, config: dict, img_filter_func=None):
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
    # 获得待配准图像绝对路径列表
    if img_filter_func is None:
        x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir)]
    else:
        x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir) if img_filter_func(_)]
    # 获得参考图像绝对路径列表
    y_arr = [os.path.join(y_dir, _) for _ in os.listdir(y_dir)]
    # 对绝对路径列表进行排序
    x_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0].split("_")[0]))
    y_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0].split("_")[0]))
    # 如果参考图像数量和待配准图像数量不同，那么意味着出错了
    assert len(x_arr) == len(y_arr)

    # 构建输入队列 & batch
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])

    # 定义处理tensor的外部python函数
    def _f(input_tensor, batch_size: int, img_height: int, img_width: int, channels: int):
        _ = np.stack([np.array(Image.open(img_name)) for img_name in input_tensor], axis=0) / 255
        return _.astype(np.float32).reshape([batch_size, img_height, img_width, channels])

    # 应用外部python函数处理tensor
    batch_x = tf.py_func(_f, [batch_x, config["batch_size"], *config["image_size"], 1], tf.float32)
    batch_y = tf.py_func(_f, [batch_y, config["batch_size"], *config["image_size"], 1], tf.float32)

    # 返回batch
    return batch_x, batch_y


if __name__ == '__main__':
    train()
