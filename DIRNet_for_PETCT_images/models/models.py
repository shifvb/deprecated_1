import pickle
# from DIRNet_for_PETCT_images.models.WarpST import WarpST
from SpatialTransformer_modify_work.modifywork.SpatialTransformer import SpatialTransformer as WarpST
WarpST = WarpST()
from DIRNet_for_PETCT_images.models.ops import *
from DIRNet_for_PETCT_images.models.grad_regularization_loss import grad_xy_v2 as grad_xy


class CNN(object):
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # conv_1
            x = conv2d(x, "conv1", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_1
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_2
            x = conv2d(x, "conv2", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_2
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_3
            x = conv2d(x, "conv3", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_3
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_4
            x = conv2d(x, "conv4", 16, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
            # pool_4
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            # conv_5
            x = conv2d(x, "conv5", 16, 1, 1, "SAME", True, tf.nn.elu, self.is_train)
            # conv_6
            x = conv2d(x, "conv6", 16, 1, 1, "SAME", True, tf.nn.elu, self.is_train)
            # conv_7
            x = conv2d(x, "conv7", 2, 1, 1, "SAME", False, None, self.is_train)

        if self.reuse is None:
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
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
        self.grad_loss = grad_xy(self.v)
        self.ncc_loss = -ncc(self.y, self.z)
        self.loss = self.ncc_loss + self.grad_loss * 1e-3

        # declare train step
        if self.is_train:
            self.optim = tf.train.AdamOptimizer(config["learning_rate"])
            self.train = self.optim.minimize(self.loss, var_list=self.vCNN.var_list)

    def fit(self, batch_x, batch_y):
        # 训练，计算loss， ncc_loss, grad_loss
        _, loss, ncc_loss, grad_loss = self.sess.run(
            fetches=[self.train, self.loss, self.ncc_loss, self.grad_loss],
            feed_dict={self.x: batch_x, self.y: batch_y},
        )
        # 返回loss， ncc_loss, grad_loss
        return loss, ncc_loss, grad_loss

    def deploy(self, dir_path, batch_x, batch_y, img_name_start_idx=0, deform_vec_path=None):
        # 计算loss和配准结果
        z, loss, ncc_loss, grad_loss = self.sess.run(
            fetches=[self.z, self.loss, self.ncc_loss, self.grad_loss],
            feed_dict={self.x: batch_x, self.y: batch_y}
        )

        # 如果指定了存储变形场路径，那么存储变形场向量
        if deform_vec_path is not None:
            pickle.dump(self.sess.run(self.v, {self.x: batch_x, self.y: batch_y}), open(deform_vec_path, 'wb'))

        # 如果指定了存储图像路径，那么存储图像
        if dir_path is not None:
            for i in range(z.shape[0]):
                _idx = img_name_start_idx + i + 1
                save_image_with_scale(dir_path + "/{:02d}_x.png".format(_idx), batch_x[i, :, :, 0])
                save_image_with_scale(dir_path + "/{:02d}_y.png".format(_idx), batch_y[i, :, :, 0])
                save_image_with_scale(dir_path + "/{:02d}_z.png".format(_idx), z[i, :, :, 0])
        # 返回loss
        return loss, ncc_loss, grad_loss

    def save(self, dir_path):
        self.vCNN.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.vCNN.restore(self.sess, dir_path + "/model.ckpt")
