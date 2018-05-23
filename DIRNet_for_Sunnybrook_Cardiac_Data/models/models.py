from DIRNet_for_Sunnybrook_Cardiac_Data.models.WarpST import WarpST
from DIRNet_for_Sunnybrook_Cardiac_Data.models.ops import *


class CNN(object):
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = conv2d(x, "conv1", 64, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            x = conv2d(x, "conv2", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            x = conv2d(x, "out1", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            x = conv2d(x, "out2", 2, 3, 1, "SAME", False, None, self.is_train)

        if self.reuse is None:
            self.var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(self.var_list)
            self.reuse = True

        return x

    def save(self, sess, ckpt_path):
        self.saver.save(sess, ckpt_path)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class DIRNet(object):
    def __init__(self, sess, config, name, is_train):
        self.sess = sess
        self.name = name
        self.is_train = is_train

        # moving / fixed images
        im_shape = [config["batch_size"]] + config["image_size"] + [1]
        self.x = tf.placeholder(tf.float32, im_shape)
        self.y = tf.placeholder(tf.float32, im_shape)
        self.xy = tf.concat([self.x, self.y], 3)

        self.vCNN = CNN("vector_CNN", is_train=self.is_train)

        # vector map & moved image
        self.v = self.vCNN(self.xy)
        self.z = WarpST(self.x, self.v, config["image_size"])

        # self.loss = mse(self.y, self.z)
        self.loss = -ncc(self.y, self.z)
        if self.is_train:
            self.optim = tf.train.AdamOptimizer(config["learning_rate"])
            self.train = self.optim.minimize(self.loss, var_list=self.vCNN.var_list)

    def fit(self, batch_x, batch_y):
        _, loss = self.sess.run([self.train, self.loss], {self.x: batch_x, self.y: batch_y})
        return loss

    def deploy(self, dir_path, x, y, img_name_start_idx=0):
        # 计算loss和配准结果
        loss, z = self.sess.run([self.loss, self.z], {self.x: x, self.y: y})
        # 如果不存储图像，只返回loss
        if dir_path is None:
            return loss
        # 存储图像
        for i in range(z.shape[0]):
            _idx = img_name_start_idx + i + 1
            save_image_with_scale(dir_path + "/{:02d}_x.png".format(_idx), x[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_y.png".format(_idx), y[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_z.png".format(_idx), z[i, :, :, 0])
        # 返回loss
        return loss

    def save(self, dir_path):
        self.vCNN.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.vCNN.restore(self.sess, dir_path + "/model.ckpt")
