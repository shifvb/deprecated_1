import os
import tensorflow as tf
from DIRNet_for_PETCT_images.models.models import DIRNet
from DIRNet_for_PETCT_images.train.logger import my_logger as logger
from DIRNet_for_PETCT_images.train.train_config_folder_guard import config_folder_guard
from DIRNet_for_PETCT_images.train.gen_batches import gen_batches


def train():
    # 定义网络参数
    config = config_folder_guard({
        # train parameters
        "image_size": [128, 128],
        "batch_size": 32,
        "learning_rate": 1e-4,
        # train data folder
        "ckpt_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\validate",
        "log_dir": r"F:\registration_running_data\log"
    })

    # 定义训练集和验证集
    train_x_dir = r"F:\registration_patches\CT-CT_75train_25valid\train\moving"
    train_y_dir = r"F:\registration_patches\CT-CT_75train_25valid\train\fixed"
    batch_x, batch_y = gen_batches(train_x_dir, train_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    })
    valid_x_dir = r"F:\registration_patches\CT-CT_75train_25valid\valid\moving"
    valid_y_dir = r"F:\registration_patches\CT-CT_75train_25valid\valid\fixed"
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })

    # 设定循环次数
    epoch_num = 50
    train_iter_num = 400
    valid_iter_num = len(os.listdir(valid_y_dir)) // config['batch_size']

    # 定义日志记录器
    train_log = logger(config["log_dir"], "train.log")
    valid_log = logger(config["log_dir"], "valid.log")

    # 构建网络
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    for epoch in range(epoch_num):
        # 放入训练集进行训练
        _train_L = []
        for i in range(train_iter_num):
            _bx, _by = sess.run([batch_x, batch_y])
            _loss_train = reg.fit(_bx, _by)
            _train_L.append(_loss_train)
        train_log.info("[TRAIN] epoch={:>6d}, loss={:.6f}".format(epoch + 1, sum(_train_L) / len(_train_L)))

        # 放入验证集进行验证
        _valid_L = []
        for j in range(valid_iter_num):
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            if (epoch + 1) % 2 == 0:
                _loss_valid = reg.deploy(config["temp_dir"], _valid_x, _valid_y, j * config["batch_size"])
            else:
                _loss_valid = reg.deploy(None, _valid_x, _valid_y)
            _valid_L.append(_loss_valid)
        valid_log.info("[VALID] epoch={:>6d}, loss={:.6f}".format(epoch + 1, sum(_valid_L) / len(_valid_L)))

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    train()
