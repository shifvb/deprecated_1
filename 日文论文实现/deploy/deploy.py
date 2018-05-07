import os
import numpy as np
import tensorflow as tf
from 日文论文实现.models.conv_regressor import ConvNetRegressor
from 日文论文实现.train.gen_batches import gen_batches


def deploy():
    _deploy_y_dir = r"F:\registration_patches\version_all\test\resized_ct"
    config = config_folder_guard({
        # network settings
        "batch_size": 10,
        "image_size": [128, 128],

        # folder path
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        # R1
        "deploy_in_x_dir_1": r"F:\registration_patches\version_all\test\normalized_pt",
        "deploy_in_y_dir_1": _deploy_y_dir,
        "deploy_out_dir_1": r"F:\registration_running_data\deploy_1",
        # R2
        "deploy_in_x_dir_2": r"F:\registration_running_data\deploy_1",
        "deploy_in_y_dir_2": _deploy_y_dir,
        "deploy_out_dir_2": r"F:\registration_running_data\deploy_2",
        # R3
        "deploy_in_x_dir_3": r"F:\registration_running_data\deploy_2",
        "deploy_in_y_dir_3": _deploy_y_dir,
        "deploy_out_dir_3": r"F:\registration_running_data\deploy_3",
    })
    deploy_iter_num = len(os.listdir(config["deploy_in_y_dir_1"])) // config["batch_size"]

    # 构建网络
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=False, config=config)
    reg.restore(sess, config["checkpoint_dir"])

    # 生成R1结果
    deploy_x_1, deploy_y_1 = gen_batches(config["deploy_in_x_dir_1"], config["deploy_in_y_dir_1"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    coord_1 = tf.train.Coordinator()
    threads_1 = tf.train.start_queue_runners(sess=sess, coord=coord_1)
    for i in range(deploy_iter_num):
        _dx_1, _dy_1 = sess.run([deploy_x_1, deploy_y_1])
        reg.deploy(config["deploy_out_dir_1"], _dx_1, _dy_1, i * config["batch_size"])

    # 生成R2结果
    deploy_x_2, deploy_y_2 = gen_batches(config["deploy_in_x_dir_2"], config["deploy_in_y_dir_2"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    }, img_filter_func=lambda _: "z1" in _)  # R2 使用R1生成的配准结果作为待配准图像
    coord_2 = tf.train.Coordinator()
    threads_2 = tf.train.start_queue_runners(sess=sess, coord=coord_2)
    for i in range(deploy_iter_num):
        _dx_2, _dy_2 = sess.run([deploy_x_2, deploy_y_2])
        reg.deploy(config["deploy_out_dir_2"], _dx_2, _dy_2, i * config["batch_size"])

    # 生成R3结果
    deploy_x_3, deploy_y_3 = gen_batches(config["deploy_in_x_dir_3"], config["deploy_in_y_dir_3"], {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    }, img_filter_func=lambda _: "z2" in _)  # R3 使用R2生成的配准结果作为待配准图像
    coord_3 = tf.train.Coordinator()
    threads_3 = tf.train.start_queue_runners(sess=sess, coord=coord_3)
    for i in range(deploy_iter_num):
        _dx_3, _dy_3 = sess.run([deploy_x_3, deploy_y_3])
        reg.deploy(config["deploy_out_dir_3"], _dx_3, _dy_3, i * config["batch_size"])

    # 回收资源
    coord_1.request_stop()
    coord_1.join(threads_1)
    coord_2.request_stop()
    coord_2.join(threads_2)
    coord_3.request_stop()
    coord_3.join(threads_3)
    sess.close()

    # 计算ncc
    # result_list = []
    # result = 0  # todo: change / remove it
    # result_list.append(result)
    # loss_arr = np.array(result_list)
    # print(loss_arr.mean(axis=0))


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    if not os.path.exists(config_dict["checkpoint_dir"]):
        raise IOError("check point dir '{}' does not exist!".format(config_dict["checkpoint_dir"]))
    if not os.path.exists(config_dict["deploy_out_dir_1"]):
        os.makedirs(config_dict["deploy_out_dir_1"])
    if not os.path.exists(config_dict["deploy_out_dir_2"]):
        os.makedirs(config_dict["deploy_out_dir_2"])
    if not os.path.exists(config_dict["deploy_out_dir_3"]):
        os.makedirs(config_dict["deploy_out_dir_3"])
    return config_dict


if __name__ == '__main__':
    deploy()
