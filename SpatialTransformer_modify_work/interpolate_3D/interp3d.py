import os
import numpy as np
import tensorflow as tf
from PIL import Image


def gen_images(save_path):
    # 删除
    if os.path.exists(save_path):
        [os.remove(os.path.join(save_path, _)) for _ in os.listdir(save_path)]
        os.rmdir(save_path)
    os.mkdir(save_path)
    # 生成 & 保存
    name = os.path.join(save_path, "batch_{}_depth_{}.jpg")
    for batch_num in range(batch_size):
        for depth_num in range(img_depth):
            # 生成每个slice/depth的array
            _arr = np.zeros(shape=[img_height, img_width, img_channel], dtype=np.uint8)
            for row_num in range(img_height):
                for col_num in range(img_width):
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


def load_arrs(load_dir):
    load_dir = os.path.abspath(load_dir)
    img_names = [os.path.join(load_dir, _) for _ in os.listdir(load_dir)]
    _L = []
    for batch_num in range(batch_size):
        # 获取单个batch并排序
        batch_img_names = list(filter(lambda _: "batch_{}".format(batch_num) in _, img_names))
        batch_img_names.sort(key=lambda _: int(_.split(".")[0].split("_")[-1]))
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


def interpolate_3d(arrs):
    arrs_tsr = tf.constant(arrs, dtype=tf.float32)
    # [n, h, w, d, c] -> [n, h*s, w*s, d, c]
    arrs_tsr = tf.reshape(arrs_tsr, [batch_size, img_height, img_width, img_depth * img_channel])
    arrs_tsr = tf.image.resize_bicubic(arrs_tsr, [img_height * scale, img_width * scale], True)
    arrs_tsr = tf.reshape(arrs_tsr, [batch_size, img_height * scale, img_width * scale, img_depth, img_channel])
    #
    arrs_tsr = tf.reshape(arrs_tsr, [batch_size, img_height * scale * img_width * scale, img_depth, img_channel])
    arrs_tsr = tf.image.resize_bicubic(arrs_tsr, [img_height * scale * img_width * scale, img_depth * scale], True)
    arrs_tsr = tf.reshape(arrs_tsr, [batch_size, img_height * scale, img_width * scale, img_depth * scale, img_channel])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(arrs_tsr)
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def main():
    gen_images("img")
    arrs = load_arrs("img")
    save_arrs(arrs, "img_out_origin")
    save_arrs(interpolate_3d(arrs), "img_out")


if __name__ == '__main__':
    batch_size, img_height, img_width, img_depth, img_channel = 2, 25, 30, 20, 3
    scale = 4
    dest_height, dest_width, dest_depth = img_height * scale, img_width * scale, img_depth * scale
    main()
