import os
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from DIRNet3D_for_PETCT_images.models.SpatialTransformer3D import SpatialTransformer3D


class DIRNet3D(object):
    def __init__(self, name, sess, ):
        pass

    def save(self, save_dir):  # todo: change
        self.vCNN.save(self.sess, os.path.join(save_dir, "model.checkpoint"))

    def load(self, load_dir):  # todo: change
        self.vCNN.save(self.sess, os.path.join(load_dir, "model.checkpoint"))

    @classmethod
    def _ncc(cls, x, y):
        mean_x = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
        mean_y = tf.reduce_mean(y, [1, 2, 3], keepdims=True)
        mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3], keepdims=True)
        mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3], keepdims=True)
        stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1, 2, 3], keepdims=True)
        stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1, 2, 3], keepdims=True)
        return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

    @classmethod
    def _mse(cls, x, y):
        return tf.reduce_mean(tf.square(x - y))


class _CNN(object):
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # conv_1
            x = self._conv2d(x, "conv1", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_1
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_2
            x = self._conv2d(x, "conv2", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_2
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_3
            x = self._conv2d(x, "conv3", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_3
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_4
            x = self._conv2d(x, "conv4", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_4
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_5
            x = self._conv2d(x, "conv5", 16, 1, 1, "SAME", True, tf.nn.elu, self.is_train)
            # conv_6
            x = self._conv2d(x, "conv6", 16, 1, 1, "SAME", True, tf.nn.elu, self.is_train)
            # conv_7
            x = self._conv2d(x, "conv7", 2, 1, 1, "SAME", False, None, self.is_train)

        if self.reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(self.var_list)
            self.reuse = True

        return x

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    @classmethod
    def _conv2d(cls, x, name, kernel_num, kernel_size, strides, padding, batch_normal, active_function, is_train):
        with tf.variable_scope(name):
            # convolution
            w = tf.get_variable('weight', [kernel_size, kernel_size, x.get_shape()[-1], kernel_num],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            x = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding)

            # batch normal
            if batch_normal:
                x = cls._batch_norm(x, "bn", is_train=is_train)
            else:
                b = tf.get_variable('biases', [kernel_num], initializer=tf.constant_initializer(0.))
                x += b

            # active function
            if active_function:
                x = active_function(x)

        return x

    @classmethod
    def _batch_norm(cls, x, name, momentum=0.9, epsilon=1e-5, is_train=True):
        return batch_norm(x, decay=momentum, updates_collections=None,
                          epsilon=epsilon, scale=True, is_training=is_train, scope=name)
