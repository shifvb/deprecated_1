import os
import tensorflow as tf
from 日文论文实现.models.conv_regressor import ConvNetRegressor
from 日文论文实现.train.config_folder_guard import config_folder_guard
from 日文论文实现.train.gen_batches import gen_batches


def train():
    # 设置参数
    _train_y_dir = r"F:\registration_patches\version_all\train\resized_ct"
    config = config_folder_guard({
        # train parameters
        "batch_size": 10,
        "epoch_num": 10000,
        "save_interval": 1000,
        "image_size": [128, 128],
        "learning_rate": 1e-5,
        "shuffle_batch": True,

        # folder path
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        # R1 path
        "train_in_x_dir_1": r"F:\registration_patches\version_all\train\normalized_pt",
        "train_in_y_dir_1": _train_y_dir,
        "valid_in_x_dir_1": r"F:\registration_patches\version_all\train\normalized_pt",
        "valid_in_y_dir_1": _train_y_dir,
        "valid_out_dir_1": r"F:\registration_running_data\validate_1",
        # R2 path
        "train_in_x_dir_2": r"F:\registration_running_data\validate_1",
        "train_in_y_dir_2": _train_y_dir,
        "valid_in_x_dir_2": r"F:\registration_running_data\validate_1",
        "valid_in_y_dir_2": _train_y_dir,
        "valid_out_dir_2": r"F:\registration_running_data\validate_2",
        # R3 path
        "train_in_x_dir_3": r"F:\registration_running_data\validate_2",
        "train_in_y_dir_3": _train_y_dir,
        "valid_in_x_dir_3": r"F:\registration_running_data\validate_2",
        "valid_in_y_dir_3": _train_y_dir,
        "valid_out_dir_3": r"F:\registration_running_data\validate_3",
    })
    valid_iter_num = len(os.listdir(config["valid_in_y_dir_1"])) // config["batch_size"]

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
    }, img_filter_func=lambda _: "z2" in _)  # R3生成的结果的待配准图像，是基于R2生成的配准结果
    coord_3 = tf.train.Coordinator()
    threads_3 = tf.train.start_queue_runners(sess=sess, coord=coord_3)
    for i in range(config["epoch_num"]):
        _tx_3, _ty_3 = sess.run([train_x_3, train_y_3])
        loss = reg.fit_only_r3(_tx_3, _ty_3)
        print("[INFO] (R3) epoch={:>5}, loss={:.3f}".format(i, loss))
    for j in range(valid_iter_num):
        _vx_3, _vy_3 = sess.run([valid_x_3, valid_y_3])
        reg.deploy(config["valid_out_dir_3"], _vx_3, _vy_3, j * config["batch_size"])
    reg.save(sess, config["checkpoint_dir"])  # 存储网络

    # 回收资源
    coord_1.request_stop()
    coord_1.join(threads_1)
    coord_2.request_stop()
    coord_2.join(threads_2)
    coord_3.request_stop()
    coord_3.join(threads_3)
    sess.close()


if __name__ == '__main__':
    train()
