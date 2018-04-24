import os
import tensorflow as tf
from 日文论文实现.models.R1 import R1, R2, R3,ConvNetRegressor


def main():
    config_dict = {
        "batch_size": 3,
        "img_height": 512,
        "img_width": 512,
        "learning_rate": 1e-5,
    }
    net = ConvNetRegressor(None, is_train=True, config=config_dict)

    # xy = tf.placeholder(dtype=tf.float32, shape=[10, 512, 512, 2])
    # for x in r:
    #     print(x)


if __name__ == '__main__':
    main()
