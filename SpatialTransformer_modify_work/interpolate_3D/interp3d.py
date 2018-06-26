import os
import numpy as np
import tensorflow as tf
from PIL import Image


def gen_images(save_path):
    batch_size = 1
    image_height = 32
    image_width = 32
    image_depth = 32
    image_channel = 3

    # scale = 4
    #
    # target_height = image_height * scale
    # target_width = image_width * scale
    # target_depth = image_depth * scale

    name = os.path.join(save_path, "batch_{}_depth_{}.jpg")
    for batch_num in range(batch_size):
        for depth_num in range(image_depth):
            _arr = np.zeros(shape=[image_height, image_width, image_channel], dtype=np.uint8)
            for row_num in range(image_height):
                for col_num in range(image_width):
                    if row_num ** 2 + col_num ** 2 < depth_num ** 2:
                        _L = [1, 1, 1]
                        _L[depth_num % 3] = 2
                        _arr[row_num, col_num] = _L
            _arr = ((_arr - _arr.min()) / (_arr.max() - _arr.min()) * 255).astype(np.uint8)
            Image.fromarray(_arr, "RGB").save(name.format(batch_num, depth_num))


def main():
    gen_images("img")


if __name__ == '__main__':
    main()
