import tensorflow as tf
from SpatialTransformer_modify_work.models.bicubic_interp import bicubic_interp_2d


class SpatialTransformer(object):
    """Deformable Transformer Layer with bicubic interpolation
    U : tf.float, [num_batch, height, width, num_channels].
        Input tensor to warp
    V : tf.float, [num_batch, height, width, 2]
        Warp map. It is interpolated to out_size.
    out_size: a tuple of two ints
        The size of the output of the network (height, width)
    ----------
    References :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py

    --------------------------------
    Modified by shifvb at 2018-06-17
    References:
        [1] https://github.com/iwyoo/DIRNet-tensorflow/blob/master/WarpST.py
        [2] https://github.com/voxelmorph/voxelmorph/blob/master/src/dense_3D_spatial_transformer.py
    Description:
        I just use the code from archive[1], but it seems to be a little bit poorly aligned.
        So I reference archive[2] and refactor the code by my comprehension .
    --------------------------------
    """

    def __call__(self, U, V):
        # deformation field
        V = bicubic_interp_2d(V, U.shape[1:3])  # [n, h, w, 2]
        dx = V[:, :, :, 0]  # [n, h, w]
        dy = V[:, :, :, 1]  # [n, h, w]
        return self._transform(U, dx, dy)

    def _transform(self, U, dx, dy):
        """
        transform (x, y)^T -> (x+vx, x+vy)^T
        :param U: Image
        :param dx: Delta in x direction (x方向偏移量)
        :param dy: Delta in y direction (y方向偏移量)
        :return: Registered result
        """
        batch_size = U.shape[0]
        height = U.shape[1]
        width = U.shape[2]

        # generate grid
        x_mesh, y_mesh = self._meshgrid(height, width)  # [h, w]
        x_mesh = tf.tile(tf.expand_dims(x_mesh, 0), [batch_size, 1, 1])  # [n, h, w]
        y_mesh = tf.tile(tf.expand_dims(y_mesh, 0), [batch_size, 1, 1])  # [n, h, w]

        # Convert dx and dy to absolute locations
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self._interpolate(U, x_new, y_new)

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, dtype='int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width):
        """
        generate grid
        generate x-grid (example of a 3x3 x-grid):
        [[-1. 0. 1.]
         [-1. 0. 1.]
         [-1. 0. 1.]]
         x_t = tf.matmul(
            tf.ones(shape=[height, 1]),
            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0])
         )

        generate y-grid (example of a 3x3 y-grid):
        [[-1.-1.-1.]
         [ 0. 0. 0.]
         [ 1. 1. 1.]]
         y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=[1, width])
         )

        This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                               np.linspace(-1, 1, height))
        :param height: height of grid
        :param width: width of grid
        :return: grid
        """
        return tf.meshgrid(tf.linspace(-1.0, 1.0, width), tf.linspace(-1.0, 1.0, height))

    def _interpolate(self, im, x, y):
        num_batch = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]
        out_height = x.shape[1]
        out_width = x.shape[2]

        print("[WARN] todo : remove -0.5 and -0.6")
        # scale indices from [-1, 1] to [0, width/height]
        x = tf.cast(tf.reshape(x, [-1]), 'float32')
        y = tf.cast(tf.reshape(y, [-1]), 'float32')
        x = (x + 1.0) * tf.to_float(width) / 2.0 - 0.5  # todo: remove -0.5
        y = (y + 1.0) * tf.to_float(height) / 2.0 - 0.6  # todo: remove -0.6

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = self._repeat(tf.range(num_batch) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        output = tf.reshape(output, [num_batch, height, width, channels])
        return output
