import tensorflow as tf


def interpolate_3d(arr, n, h, w, d, c, s):
    arr = tf.reshape(arr, [n, h, w, d * c])
    arr = tf.image.resize_bicubic(arr, [h * s, w * s], True)  # [n, h, w, d, c] -> [n, h*s, w*s, d, c]
    arr = tf.reshape(arr, [n, h * s * w * s, d, c])
    arr = tf.image.resize_bicubic(arr, [h * s * w * s, d * s], True)  # [n, h*s, w*s, d, c] -> [n, h*s, w*s, d*s, c]
    return tf.reshape(arr, [n, h * s, w * s, d * s, c])
