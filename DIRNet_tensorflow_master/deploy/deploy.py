import os
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from DIRNet_tensorflow_master.train.train import gen_batches


def deploy():
    config_dict = config_folder_guard({
        # network settings
        "batch_size": 10,
        "image_size": [128, 128],
        "shuffle_batch": False,

        # folder path
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "result_dir": r"F:\registration_running_data\result",
    })
    # 获取图片路径集
    deploy_x_dir = r"F:\registration_patches\version_3(pt-ct)\validate\normolized_pt"
    deploy_y_dir = r"F:\registration_patches\version_3(pt-ct)\validate\resized_ct"
    deploy_x, deploy_y = gen_batches(deploy_x_dir, deploy_y_dir, config_dict)

    # 构建网络
    sess = tf.Session()
    reg = DIRNet(sess, config_dict, "DIRNet", is_train=False)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 生成结果
    reg.restore(config_dict["checkpoint_dir"])
    result_list = []
    for i in range(len(os.listdir(deploy_y_dir)) // config_dict["batch_size"]):
        _dx, _dy = sess.run([deploy_x, deploy_y])
        result = reg.deploy(config_dict["result_dir"], _dx, _dy, i * config_dict["batch_size"])
        result_list.append(result)

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()

    # 计算ncc
    loss_arr = np.array(result_list)
    print(loss_arr.mean())


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    if not os.path.exists(config_dict["checkpoint_dir"]):
        raise IOError("check point dir '{}' does not exist!".format(config_dict["checkpoint_dir"]))
    if not os.path.exists(config_dict["result_dir"]):
        os.makedirs(config_dict["result_dir"])
    return config_dict


if __name__ == "__main__":
    deploy()
