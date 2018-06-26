import os
import numpy as np
import tensorflow as tf
from PIL import Image


def gen_images(save_path):
    name = os.path.join(save_path, "batch_{}_depth_{}.jpg")
    for batch_num in range(batch_size):
        for depth_num in range(image_depth):
            # 生成每个slice/depth的array
            _arr = np.zeros(shape=[image_height, image_width, image_channel], dtype=np.uint8)
            for row_num in range(image_height):
                for col_num in range(image_width):
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
    name = os.path.join(os.path.abspath(save_dir), "batch_{}_depth_{}.jpg")
    for batch_num in range(arrs.shape[0]):
        for depth_num in range(arrs.shape[3]):
            _arr = arrs[batch_num, :, :, depth_num, :]
            Image.fromarray(_arr).save(name.format(batch_num, depth_num))


def interpolate_3d(arrs):
    return arrs


def main():
    # gen_images("img")
    arrs = load_arrs("img")
    # arrs2 = interpolate_3d(arrs)
    save_arrs(arrs, "img_out_origin")
    # save_arrs(arrs2)


if __name__ == '__main__':
    batch_size, image_height, image_width, image_depth, image_channel = 2, 25, 30, 20, 3
    scale = 4
    target_height, target_width, target_depth = image_height * scale, image_width * scale, image_depth * scale
    main()
