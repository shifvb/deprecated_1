import tensorflow as tf
import numpy as np
from PIL import Image
from SpatialTransformer_modify_work.interpolate.bicubic_interp import bicubic_interp_2d


def main():
    img = Image.open(r"C:\Users\anonymous\Desktop\1\lozman.png")
    img_arr = np.array(img)
    img_tsr = tf.reshape(tf.Variable(img_arr, dtype=tf.float32), [1, 20, 21, 1])

    # # 形变场向量
    # def_vec = np.array([
    #     [0, 0, 0],
    #     [0, 0.3, 0],
    #     [0, 0, 0]
    # ], dtype=np.float32).reshape([1, 3, 3, 1])
    # def_tsr = tf.Variable(def_vec, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = bicubic_interp_2d(img_tsr, (200, 210))
        r = sess.run(output)
        print(r[0, :, :, 0])


if __name__ == '__main__':
    main()
