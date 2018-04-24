import os
import tensorflow as tf
from 日文论文实现.models.R1 import R1, R2


def main():
    r1 = R1("R1", is_train=True)
    r2 = R2("R2", is_train=True)
    xy = tf.placeholder(dtype=tf.float32, shape=[10, 512, 512, 2])
    r1_out = r1(xy)
    r2_out = r2(xy, r1_out)
    print(r2_out)


if __name__ == '__main__':
    main()
