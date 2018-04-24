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


class R2(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x, R1_out):
        with tf.variable_scope(self._name, reuse=self._reuse):
            x = tf.nn.avg_pool(x, [1, 64, 64, 1], [1, 64, 64, 1], "SAME")
            # 将R1的输出插值 [batch_size, 4, 4, 2] -> [batch_size, 8, 8, 2]
            R1_out = tf.image.resize_nearest_neighbor(R1_out, [8, 8])
            # 将R1输出插值的结果concat到R2最大池化的结果上
            # [batch_size, 8, 8, 2] concat [batch_size, 8, 8, 2] -> [batch_size, 8, 8, 4]
            x = tf.concat([R1_out, x], axis=3)
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


class R3(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x, R2_out):
        with tf.variable_scope(self._name, reuse=self._reuse):
            x = tf.nn.avg_pool(x, [1, 32, 32, 1], [1, 32, 32, 1], "SAME")
            # 将R2的输出插值 [batch_size, 8, 8, 2] -> [batch_size, 16, 16, 2]
            R2_out = tf.image.resize_nearest_neighbor(R2_out, [16, 16])
            x = tf.concat([R2_out, x], axis=3)
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


class ConvNetRegressor(object):
    def __init__(self, is_train: bool):
        self._is_train = is_train

    def __call__(self, xy):  # xy:[batch_size, height(=512), width(=512), channel(=2)]
        self._r1 = R1("R1", is_train=self._is_train)
        self._r2 = R2("R2", is_train=self._is_train)
        self._r3 = R3("R3", is_train=self._is_train)
        r1_out = self._r1(xy)
        r2_out = self._r2(xy, r1_out)
        r3_out = self._r3(xy, r2_out)
        return r1_out, r2_out, r3_out

    def save(self, session, checkpoint_path):
        self._r1.save(session, checkpoint_path)
        self._r2.save(session, checkpoint_path)
        self._r3.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self._r1.restore(session, checkpoint_path)
        self._r2.restore(session, checkpoint_path)
        self._r3.restore(session, checkpoint_path)
