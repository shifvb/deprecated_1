import os
import time
import numpy as np
import tensorflow as tf
from DIRNet3D_for_PETCT_images.models.DIRNet_3d import DIRNet3D
from DIRNet3D_for_PETCT_images.train.train_utils import my_logger as logger
from DIRNet3D_for_PETCT_images.train.train_utils import LossRecorder
from DIRNet3D_for_PETCT_images.data.sample_data import MyBatch as Batch


class TrainConfig(object):
    def __init__(self, image_size, batch_size, learning_rate,
                 train_x_dir, train_y_dir, valid_x_dir, valid_y_dir, temp_dir):
        # 网络参数设置
        self.image_size = [batch_size] + image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 循环次数设置
        self.epoch_num = 50
        self.train_per_epoch = 400
        self.valid_iter_num = len(os.listdir(valid_y_dir)) // batch_size

        # 数据设置
        self.train_data_set = Batch(train_x_dir, train_y_dir, self.batch_size, True)
        self.valid_data_set = Batch(valid_x_dir, valid_y_dir, self.batch_size, False)

        # 存储路径设置
        self.temp_dir = temp_dir
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.train_logger = logger(r"F:\registration_running_data\log", "train.log")
        self.valid_logger = logger(r"F:\registration_running_data\log", "valid.log")
        self.loss_rec = LossRecorder()


def main():
    # 网络参数设置
    cfg = TrainConfig(
        image_size=[64, 64, 64, 1],
        batch_size=32,
        learning_rate=1e-4,
        train_x_dir=r"F:\KHJ\3D volume\pt_volume",
        train_y_dir=r"F:\KHJ\3D volume\ct_volume",
        valid_x_dir=r"F:\KHJ\3D volume\pt_volume_valid",
        valid_y_dir=r"F:\KHJ\3D volume\ct_volume_valid",
        temp_dir=r"F:\registration_running_data\validate",
    )

    # 构建网络
    sess = tf.Session()
    net = DIRNet3D(img_shape=cfg.image_size, sess=sess, is_train=True, learning_rate=cfg.learning_rate)
    sess.run(tf.global_variables_initializer())

    # 开始训练
    for epoch in range(cfg.epoch_num):
        # 开始单个epoch训练
        for i in range(cfg.train_per_epoch):
            _tx, _ty = cfg.train_data_set.next_batch()  # 随机获得训练集
            cfg.loss_rec.record_loss(*net.fit(_tx, _ty))  # 记录训练loss
        # 记录训练loss日志
        cfg.train_logger.info("[TRAIN] epoch={:>6d}, loss={:.6f}, "
                              "ncc_loss={:.6f}, grad_loss={:.6f}".format(epoch, *cfg.loss_rec.get_losses()))

        # 放入验证集进行验证
        _v_img_path = cfg.temp_dir if epoch % 5 == 0 else None
        for j in range(cfg.valid_iter_num):
            _vx, _vy = cfg.valid_data_set.next_batch()  # 顺序获得训练集
            cfg.loss_rec.record_loss(*net.deploy(_vx, _vy, _v_img_path, j * cfg.batch_size))  # 记录验证loss
        # 记录验证loss日志
        cfg.valid_logger.info("[VALID] epoch={:>6d}, loss={:.6f}, "
                              "ncc_loss={:.6f}, grad_loss={:.6f}".format(epoch, *cfg.loss_rec.get_losses()))

    # 释放资源
    sess.close()


if __name__ == '__main__':
    main()
