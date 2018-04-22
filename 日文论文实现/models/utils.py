import tensorflow as tf


def conv2d(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
            x += b
        if af:
            x = af(x)
    return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)
