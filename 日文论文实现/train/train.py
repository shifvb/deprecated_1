import os
import tensorflow as tf
from 日文论文实现.models.R1 import R1, R2, R3,ConvNetRegressor


def main():
    convnet_regressor = ConvNetRegressor(is_train=True)
    xy = tf.placeholder(dtype=tf.float32, shape=[10, 512, 512, 2])
    r = convnet_regressor(xy)
    for x in r:
        print(x)


if __name__ == '__main__':
    main()
