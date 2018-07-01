import tensorflow as tf


# todo: is it true for changing `[1,2,3]` to `[1,2,3,4]` when implementing 3d version?
def ncc(x, y):
    mean_x = tf.reduce_mean(x, [1, 2, 3, 4], keepdims=True)
    mean_y = tf.reduce_mean(y, [1, 2, 3, 4], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3, 4], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3, 4], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3, 4], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3, 4], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

# def mse(x, y):
#     return tf.reduce_mean(tf.square(x - y))
