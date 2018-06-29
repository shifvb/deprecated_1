import os
import numpy as np
import tensorflow as tf
from PIL import Image
from SpatialTransformer_modify_work.interpolate_3D.interp3d import interpolate_3d


def gen_images(save_path, n, h, w, d, c):
    # 删除
    if os.path.exists(save_path):
        [os.remove(os.path.join(save_path, _)) for _ in os.listdir(save_path)]
        os.rmdir(save_path)
    os.mkdir(save_path)
    # 生成 & 保存
    name = os.path.join(save_path, "batch_{}_depth_{}.jpg")
    for batch_num in range(n):
        for depth_num in range(d):
            # 生成每个slice/depth的array
            _arr = np.zeros(shape=[h, w, c], dtype=np.uint8)
            for row_num in range(h):
                for col_num in range(w):
                    if row_num ** 2 + col_num ** 2 < depth_num ** 2:
                        _L = [batch_num, batch_num, batch_num]
                        _L[depth_num % 3] += 1
                        _arr[row_num, col_num] = _L
                    else:
                        _arr[row_num, col_num] = [0, 0, 0]
            # 归一化
            if _arr.max() - _arr.min() == 0:
                _arr[:] = 0
            else:
                _arr = ((_arr - _arr.min()) / (_arr.max() - _arr.min()) * 255).astype(np.uint8)
            Image.fromarray(_arr, "RGB").save(name.format(batch_num, depth_num))


def load_arrs(load_dir, n):
    load_dir = os.path.abspath(load_dir)
    img_names = [os.path.join(load_dir, _) for _ in os.listdir(load_dir)]
    _L = []
    for batch_num in range(n):
        # 获取单个batch并排序
        batch_img_names = list(filter(lambda _: "batch_{}".format(batch_num) in _, img_names))
        batch_img_names.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0].split("_")[-1]))
        # 加载图像
        _arr = np.stack([np.array(Image.open(_)) for _ in batch_img_names], axis=2)  # [height, width, depth, channel]
        _L.append(_arr)
    _arrs = np.stack(_L, axis=0)  # [batch, height, width, depth, channel]
    return _arrs


def save_arrs(arrs, save_dir):
    # 删除
    if os.path.exists(save_dir):
        [os.remove(os.path.join(save_dir, _)) for _ in os.listdir(save_dir)]
        os.rmdir(save_dir)
    os.mkdir(save_dir)
    # 保存
    name = os.path.join(os.path.abspath(save_dir), "batch_{}_depth_{}.jpg")
    for batch_num in range(arrs.shape[0]):
        for depth_num in range(arrs.shape[3]):
            _arr = arrs[batch_num, :, :, depth_num, :]
            Image.fromarray(_arr).save(name.format(batch_num, depth_num))


def analog_test():
    img_size = [2, 25, 30, 20, 3]
    out_size = [2, 100, 120, 120, 3]

    # 生成测试图像
    gen_images("analog_img", *img_size)

    # 加载测试图像
    arrs = load_arrs("analog_img", img_size[0])
    arrs_tsr = tf.constant(arrs, dtype=tf.float32)

    # 生成插值图像
    save_arrs(arrs, "analog_img_out_origin")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(interpolate_3d(arrs_tsr, *img_size, *out_size[1:-1]))
        result = np.clip(result, 0, 255).astype(np.uint8)
    save_arrs(result, "analog_img_out")


def natural_test():
    img_size = [1, 540, 960, 3, 3]
    out_size = [1, 540, 960, 320, 3]

    # 加载自然图像
    arrs = load_arrs(r"nature_img", img_size[0])
    arrs_tsr = tf.constant(arrs, dtype=tf.float32)

    # 生成插值图像
    save_arrs(arrs, "nature_img_out_origin")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(interpolate_3d(arrs_tsr, *img_size, *out_size[1:-1]))
        result = np.clip(result, 0, 255).astype(np.uint8)
    save_arrs(result, "nature_img_out")


if __name__ == '__main__':
    analog_test()
    # natural_test()
