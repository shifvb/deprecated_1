from DIRNet_tensorflow_master.train.train_exchange_obj import TEO
from DIRNet_tensorflow_master.train.train_exchange_obj import TEO

# my grid start # todo remove it
_mygrid = _meshgrid(out_height, out_width)  # [2, h*w]
_mygrid = tf.reshape(_mygrid, [-1])  # [2*h*w]
_mygrid = tf.tile(_mygrid, tf.stack([num_batch]))  # [n*2*h*w]
_mygrid = tf.reshape(_mygrid, tf.stack([num_batch, 2, -1]))  # [n, 2, h*w]
_my_x_s = tf.slice(_mygrid, [0, 0, 0], [-1, 1, -1])
_my_y_s = tf.slice(_mygrid, [0, 1, 0], [-1, 1, -1])
_my_x_s_flat = tf.reshape(_my_x_s, [-1])
_my_y_s_flat = tf.reshape(_my_y_s, [-1])
_my_x_s_flat = tf.cast((_my_x_s_flat + 1) * (width_f) / 2, 'int32')
_my_y_s_flat = tf.cast((_my_y_s_flat + 1) * (height_f) / 2, 'int32')
TEO.x_diff = tf.cast(x0 - _my_x_s_flat, 'float32')  # todo remove it
TEO.y_diff = tf.cast(y0 - _my_y_s_flat, 'float32')  # todo remove it


class TEO(object):
    pass


# ncc: from 0.1 -> 0.8, so -ncc is from -0.1 -> -0.8, can be `minimized`
self.loss_term_1 = -ncc(self.y, self.z)
# todo: my version of loss
_mean_x = tf.reduce_mean(TEO.x_diff)
_variance_x = tf.reduce_mean(tf.square(TEO.x_diff - _mean_x))
_mean_y = tf.reduce_mean(TEO.y_diff)
_variance_y = tf.reduce_mean(tf.square(TEO.y_diff - _mean_y))
self.loss_term_2 = _variance_y + _variance_x
self.loss = self.loss_term_1 + self.loss_term_2
# self.sess.run(tf.variables_initializer(self.vCNN.var_list))
