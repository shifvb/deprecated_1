import os
import time
import numpy as np
import tensorflow as  tf
from DIRNet3D_for_PETCT_images.models.DIRNet_3d import DIRNet3D
from DIRNet3D_for_PETCT_images.train.train_utils import my_logger as logger
from DIRNet3D_for_PETCT_images.train.train_utils import LossRecorder


class TrainConfig(object):
    def __init__(self, batch_size, learning_rate,
                 train_x_dir, train_y_dir, valid_x_dir, valid_y_dir, temp_dir):
        # 网络参数设置
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 循环次数设置
        self.epoch_num = 50
        self.train_per_epoch = 400
        self.valid_iter_num = len(os.listdir(valid_y_dir)) // batch_size

        # 存储路径设置
        self.temp_dir = temp_dir
        self.train_logger = logger(r"F:\registration_running_data\log", "train.log")
        self.valid_logger = logger(r"F:\registration_running_data\log", "valid.log")
        self.loss_rec = LossRecorder()


def main():
    # 网络参数设置
    # todo: pathes
    cfg = TrainConfig(
        batch_size=32,
        learning_rate=1e-4,
        train_x_dir=r"",
        train_y_dir=r"",
        valid_x_dir=r"",
        valid_y_dir=r"",
        temp_dir=r"",
    )

    # 构建网络
    sess = tf.Session()
    net = DIRNet3D(img_shape=[1, 128, 128, 128, 3], sess=sess, is_train=True, learning_rate=cfg.learning_rate)

    # 开始训练
    for epoch in range(cfg.epoch_num):
        # 开始单个epoch训练
        for i in range(cfg.train_per_epoch):
            _tx, _ty = train_dataset.next_batch()  # 随机获得训练集
            cfg.loss_rec.record_loss(*net.fit(_tx, _ty))  # 记录训练loss
        # 记录训练loss日志
        cfg.train_logger.info("[TRAIN] epoch={:>6d}, loss={:.6f}, "
                              "ncc_loss={:.6f}, grad_loss={:.6f}".format(epoch, *cfg.loss_rec.get_losses()))

        # 放入验证集进行验证
        _v_img_path = cfg.temp_dir if epoch % 5 == 0 else None
        for j in range(cfg.valid_iter_num):
            _vx, _vy = valid_dataset.next_batch()  # 顺序获得训练集
            cfg.loss_rec.record_loss(*net.deploy(_vx, _vy, _v_img_path, j * cfg.batch_size))  # 记录验证loss
        # 记录验证loss日志
        cfg.valid_logger.info("[VALID] epoch={:>6d}, loss={:.6f}, "
                              "ncc_loss={:.6f}, grad_loss={:.6f}".format(epoch, *cfg.loss_rec.get_losses()))


if __name__ == '__main__':
    main()
