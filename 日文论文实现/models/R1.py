import tensorflow as tf
from 日文论文实现.models.utils import conv2d


class R1(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            x = tf.nn.avg_pool(x, [1, 128, 128, 1], [1, 128, 128, 1], "SAME")
            x = conv2d(x, "conv_1", 64, 3, 1, "SAME", True, tf.nn.elu, self._is_train)
            x = conv2d(x, "conv_2", 2, 3, 1, "SAME", False, None, self._is_train)
        if self._reuse is None:
            self._var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self._var_list)
            self._reuse = True
        return x

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)
