import os
import numpy as np
import tensorflow as tf
from DIRNet_tensorflow_master.models.models import DIRNet
from PIL import Image


def train():
    config = config_folder_guard({
        # train parameters
        "image_size": [128, 128],
        "batch_size": 10,
        "learning_rate": 1e-5,
        "iteration_num": 10000,
        "save_interval": 1000,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\validate",
    })

    # 定义训练集和验证集
    batch_x_dir = r"F:\registration_patches\version_3(pt-ct)\train\normalized_pt"
    batch_y_dir = r"F:\registration_patches\version_3(pt-ct)\train\resized_ct"
    batch_x, batch_y = gen_batches(batch_x_dir, batch_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    })
    valid_x_dir = r"F:\registration_patches\version_3(pt-ct)\validate\normolized_pt"
    valid_y_dir = r"F:\registration_patches\version_3(pt-ct)\validate\resized_ct"
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })

    # 构建网络
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    for i in range(config["iteration_num"]):
        _bx, _by = sess.run([batch_x, batch_y])
        loss = reg.fit(_bx, _by)
        print("iter={:>6d}, loss={:.6f}".format(i + 1, loss))
        if (i + 1) % config['save_interval'] == 0:
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            reg.deploy(config["temp_dir"], _valid_x, _valid_y)
            reg.save(config["checkpoint_dir"])

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()


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


def config_folder_guard(config: dict):
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    return config


if __name__ == "__main__":
    train()
