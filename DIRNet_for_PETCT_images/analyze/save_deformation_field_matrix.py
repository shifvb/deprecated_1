import os


def save_defvec(batch_x, batch_y, epoch: int, iteration: int, sess, deploy_func, save_dir: str = None):
    if save_dir is None:
        save_dir = r"F:\registration_running_data"

    # 准备存储变形场矩阵路径
    defvec_arr_dir = os.path.join(save_dir, "defvec_arrs")
    if not os.path.exists(defvec_arr_dir):
        os.mkdir(defvec_arr_dir)
    defvec_arr_name = "epoch_{:>02}_iter_{:>03}.pickle".format(epoch, iteration)
    defvec_arr_path = os.path.join(defvec_arr_dir, defvec_arr_name)

    # 准备存储变形场图像路径
    defvec_img_dir = os.path.join(save_dir, "defvec_imgs")
    if not os.path.exists(defvec_img_dir):
        os.mkdir(defvec_img_dir)
    defvec_img_name = "defvec_epoch_{:>02}_iter_{:>03}".format(epoch, iteration)
    defvec_img_path = os.path.join(defvec_img_dir, defvec_img_name)
    if not os.path.exists(defvec_img_path):
        os.mkdir(defvec_img_path)

    # 获取数据集
    defvec_x, defvec_y = sess.run([batch_x, batch_y])

    # 存储数据
    deploy_func(defvec_img_path, defvec_x, defvec_y, deform_vec_path=defvec_arr_path)
