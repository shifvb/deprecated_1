import time
import numpy as np
import tensorflow as  tf
from DIRNet3D_for_PETCT_images.models.DIRNet_3d import DIRNet3D
from DIRNet3D_for_PETCT_images.train.train_utils import my_logger as logger
from DIRNet3D_for_PETCT_images.train.train_utils import LossRecorder


class TrainConfig(object):
    def __init__(self):
        self.learning_rate = 1e-4
        self.epoch_num = 50
        self.train_per_epoch = 400
        self.train_logger = logger(r"F:\registration_running_data\log", "train.log")
        self.valid_logger = logger(r"F:\registration_running_data\log", "valid.log")


def main():
    # 网络参数设置
    cfg = TrainConfig()

    # 构建网络
    sess = tf.Session()
    net = DIRNet3D(img_shape=[1, 128, 128, 128, 3], sess=sess, is_train=True, learning_rate=cfg.learning_rate)

    # 开始训练
    for epoch in range(cfg.epoch_num):
        # 开始单个epoch训练
        loss_recorder = LossRecorder()  # 初始化loss recorder
        for i in range(cfg.train_per_epoch):
            _tx, _ty = train_dataset.next_batch()
            loss_recorder.record_loss(*net.fit(_tx, _ty))  # 记录loss
        # 记录训练loss日志
        cfg.train_logger.info(
            "[TRAIN] epoch={:>6d}, loss={:.6f}, ncc_loss={:.6f}, grad_loss={:.6f}".format(
                epoch, *loss_recorder.get_avg_losses()
            )
        )
        # 单个epoch训练后放入验证集进行验证


if __name__ == '__main__':
    main()
