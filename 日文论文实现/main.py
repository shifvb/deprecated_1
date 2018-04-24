import tensorflow as tf
import os

import numpy as np


def main():
    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=np.float32)
    arr = arr.reshape([1, 2, 4, 1])
    arr2 = np.array([
        [100, 1, 12, 13, 54, 677, 766, 5],
        [-536, 3, 34, 7, 123, 34, 999, 3],
        [100, 1, 12, 13, 54, 677, 776, 5],
        [100, 1, 12, 13, 54, 677, 766, 5],
    ], dtype=np.float32)
    arr2 = arr2.reshape([1, 4, 8, 1])

    xy = tf.placeholder(dtype=tf.float32, shape=[1, 2, 4, 1])
    xy2 = tf.placeholder(dtype=tf.float32, shape=[1, 4, 8, 1])
    xy_resized = tf.image.resize_nearest_neighbor(xy, [4, 8])
    z = tf.concat([xy_resized, xy2], axis=3)

    with tf.Session() as sess:
        r = sess.run(z, {xy: arr, xy2: arr2})
        print(r.shape)
        print(r[0,:,:,0])


if __name__ == '__main__':
    main()
