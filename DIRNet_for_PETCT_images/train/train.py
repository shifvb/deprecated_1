import os
import tensorflow as tf
from DIRNet_for_PETCT_images.models.models import DIRNet
from DIRNet_for_PETCT_images.train.logger import my_logger as logger
from DIRNet_for_PETCT_images.train.train_config_folder_guard import config_folder_guard
from DIRNet_for_PETCT_images.train.gen_batches import gen_batches
from DIRNet_for_PETCT_images.analyze.save_deformation_field_matrix import save_defvec


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
    })

    # 定义训练集和验证集
    train_x_dir = r"F:\registration_patches\PET-CT_75train_25valid\train\pt"
    train_y_dir = r"F:\registration_patches\PET-CT_75train_25valid\train\ct"
    batch_x, batch_y = gen_batches(train_x_dir, train_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": True
    })
    valid_x_dir = r"F:\registration_patches\PET-CT_75train_25valid\validate\pt"
    valid_y_dir = r"F:\registration_patches\PET-CT_75train_25valid\validate\ct"
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })
    defvec_x_dir = r"F:\registration_patches\PET-CT_75train_25valid\def_vec\pt"
    defvec_y_dir = r"F:\registration_patches\PET-CT_75train_25valid\def_vec\ct"
    # 生成batch Tensor
    defvec_x, defvec_y = gen_batches(defvec_x_dir, defvec_y_dir, {
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "shuffle_batch": False
    })

    # 设定循环次数
    epoch_num = 50
    train_iter_num = 400
    valid_iter_num = len(os.listdir(valid_y_dir)) // config['batch_size']

    # 定义日志记录器
    train_log = logger(r"F:\registration_running_data\log", "train.log")
    valid_log = logger(r"F:\registration_running_data\log", "valid.log")

    # 构建网络
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    for epoch in range(epoch_num):
        # 初始化暂存loss列表
        _tL_loss, _tL_ncc, _tL_grad = [], [], []
        # 开始训练集循环
        for i in range(train_iter_num):
            # 每100次循环，存储一下变形场矩阵
            if i % 100 == 0:
                save_defvec(defvec_x, defvec_y, epoch, i, sess, reg.deploy)
            # 放入训练集进行训练
            _tx, _ty = sess.run([batch_x, batch_y])
            _t_loss, _t_ncc_loss, _t_grad_loss = reg.fit(_tx, _ty)
            # 暂时存储loss
            _tL_loss.append(_t_loss)
            _tL_ncc.append(_t_ncc_loss)
            _tL_grad.append(_t_grad_loss)
        # 记录训练loss日志
        train_log.info(
            "[TRAIN] epoch={:>6d}, loss={:.6f}, ncc_loss={:.6f}, grad_loss={:.6f}".format(
                epoch + 1, sum(_tL_loss) / len(_tL_loss), sum(_tL_ncc) / len(_tL_ncc), sum(_tL_grad) / len(_tL_grad)
            )
        )

        # 初始化暂存loss列表
        _vL_loss, _vL_ncc, _vL_grad = [], [], []
        # 放入验证集进行验证
        for j in range(valid_iter_num):
            _vx, _vy = sess.run([valid_x, valid_y])
            # 每5个epoch，存储一下验证集图像（配准结果）
            if (epoch + 1) % 5 == 0:
                _v_loss, _v_ncc_loss, _v_grad_loss = reg.deploy(config["temp_dir"], _vx, _vy, j * config["batch_size"])
            else:
                _v_loss, _v_ncc_loss, _v_grad_loss = reg.deploy(None, _vx, _vy)
            # 暂时存储loss
            _vL_loss.append(_v_loss)
            _vL_ncc.append(_v_ncc_loss)
            _vL_grad.append(_v_grad_loss)
        # 记录验证loss日志
        valid_log.info(
            "[VALID] epoch={:>6d}, loss={:.6f}, ncc_loss={:.6f}, grad_loss={:.6f}".format(
                epoch + 1, sum(_vL_loss) / len(_vL_loss), sum(_vL_ncc) / len(_vL_ncc), sum(_vL_grad) / len(_vL_grad)
            )
        )

    # 回收资源
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    train()
