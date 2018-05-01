import os
import tensorflow as tf
from 日文论文实现.models.utils import conv2d
from 日文论文实现.models.WarpST import WarpST
from 日文论文实现.models.ops import ncc, save_image_with_scale


class R1(object):
    def __init__(self, name: str, is_train: bool):
        self._name = name
        self._is_train = is_train
        self._reuse = None

    def __call__(self, x):
        with tf.variable_scope(self._name, reuse=self._reuse):
            # x = tf.nn.avg_pool(x, [1, 128, 128, 1], [1, 128, 128, 1], "SAME")
            x = tf.nn.avg_pool(x, [1, 32, 32, 1], [1, 32, 32, 1], "SAME")
            x = conv2d(x, "conv_1", 64, 3, 1, "SAME", True, tf.nn.elu, self._is_train)
            x = conv2d(x, "conv_2", 2, 3, 1, "SAME", False, None, self._is_train)
        if self._reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
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
            # x = tf.nn.avg_pool(x, [1, 64, 64, 1], [1, 64, 64, 1], "SAME")
            x = tf.nn.avg_pool(x, [1, 16, 16, 1], [1, 16, 16, 1], "SAME")
            # 将R1的输出插值 [batch_size, 4, 4, 2] -> [batch_size, 8, 8, 2]
            R1_out = tf.image.resize_nearest_neighbor(R1_out, [8, 8])
            # 将R1输出插值的结果concat到R2最大池化的结果上
            # [batch_size, 8, 8, 2] concat [batch_size, 8, 8, 2] -> [batch_size, 8, 8, 4]
            x = tf.concat([R1_out, x], axis=3)
            x = conv2d(x, "conv_1", 64, 3, 1, "SAME", True, tf.nn.elu, self._is_train)
            x = conv2d(x, "conv_2", 2, 3, 1, "SAME", False, None, self._is_train)
        if self._reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
            self.saver = tf.train.Saver(self.var_list)
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
            # x = tf.nn.avg_pool(x, [1, 32, 32, 1], [1, 32, 32, 1], "SAME")
            x = tf.nn.avg_pool(x, [1, 8, 8, 1], [1, 8, 8, 1], "SAME")
            # 将R2的输出插值 [batch_size, 8, 8, 2] -> [batch_size, 16, 16, 2]
            R2_out = tf.image.resize_nearest_neighbor(R2_out, [16, 16])
            x = tf.concat([R2_out, x], axis=3)
            x = conv2d(x, "conv_1", 64, 3, 1, "SAME", True, tf.nn.elu, self._is_train)
            x = conv2d(x, "conv_2", 2, 3, 1, "SAME", False, None, self._is_train)
            if self._reuse is None:
                self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
                self.saver = tf.train.Saver(self.var_list)
                self._reuse = True
            return x

    def save(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path)

    def restore(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)


class ConvNetRegressor(object):
    def __init__(self, sess: tf.Session, is_train: bool, config: dict):
        self._sess = sess
        _is_train = is_train
        _batch_size = config["batch_size"]
        _img_height, _img_width = config["image_size"]
        _learning_rate = config['learning_rate']

        self._R1 = R1("R1", is_train=_is_train)
        self._R2 = R2("R2", is_train=_is_train)
        self._R3 = R3("R3", is_train=_is_train)
        self.x = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[_batch_size, _img_height, _img_width, 1])
        xy = tf.concat([self.x, self.y], axis=3)  # [batch_size, img_height, img_width, 2]
        r1_out = self._R1(xy)
        r2_out = self._R2(xy, r1_out)
        r3_out = self._R3(xy, r2_out)
        self._z1 = WarpST(self.x, r1_out, [_img_height, _img_width], name="WrapST_1")
        self._z2 = WarpST(self.x, r2_out, [_img_height, _img_width], name="WrapST_2")
        self._z3 = WarpST(self.x, r3_out, [_img_height, _img_width], name="WrapST_3")
        if _is_train:
            loss_1 = -ncc(self.y, self._z1)
            loss_2 = -ncc(self.y, self._z2)
            loss_3 = -ncc(self.y, self._z3)
            self.loss = 1 * loss_1 + 0.5 * loss_2 + 0.25 * loss_3
            _optimizer = tf.train.AdamOptimizer(_learning_rate)
            _var_list = self._R1.var_list + self._R2.var_list + self._R3.var_list
            self.train_step = _optimizer.minimize(self.loss, var_list=_var_list)
        self._sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss = self._sess.run(
            fetches=[self.train_step, self.loss],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )
        return loss

    def deploy(self, save_folder: str, x, y):
        z1, z2, z3 = self._sess.run([self._z1, self._z2, self._z3], feed_dict={self.x: x, self.y: y})
        for i in range(z1.shape[0]):
            save_image_with_scale(save_folder + "/{:02d}_x.png".format(i + 1), x[i, :, :, 0])
            save_image_with_scale(save_folder + "/{:02d}_y.png".format(i + 1), y[i, :, :, 0])
            save_image_with_scale(save_folder + "/{:02d}_z1.png".format(i + 1), z1[i, :, :, 0])
            save_image_with_scale(save_folder + "/{:02d}_z2.png".format(i + 1), z2[i, :, :, 0])
            save_image_with_scale(save_folder + "/{:02d}_z3.png".format(i + 1), z3[i, :, :, 0])

    def save(self, sess, save_folder: str):
        self._R1.save(sess, os.path.join(save_folder, "R1.ckpt"))
        self._R2.save(sess, os.path.join(save_folder, "R2.ckpt"))
        self._R3.save(sess, os.path.join(save_folder, "R3.ckpt"))

    def restore(self, sess, save_folder):
        self._R1.restore(sess, os.path.join(save_folder, "R1.ckpt"))
        self._R2.restore(sess, os.path.join(save_folder, "R2.ckpt"))
        self._R3.restore(sess, os.path.join(save_folder, "R3.ckpt"))
