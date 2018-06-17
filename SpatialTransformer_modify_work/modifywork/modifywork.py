import numpy as np
from PIL import Image
import tensorflow as tf
from SpatialTransformer_modify_work.modifywork.get_images_arr import get_images_arr
from SpatialTransformer_modify_work.modifywork.gen_diff_arr import gen_diff_arr
from SpatialTransformer_modify_work.modifywork.SpatialTransformer import SpatialTransformer


def main():
    # 图像数据
    img_arr = get_images_arr(r"F:\tmp2")[5, :, :, 0].reshape([1, 216, 384, 1])
    Image.fromarray(255 - img_arr[0, :, :, 0], "L").save("f:\\tmp3\\original.png")

    # 形变场向量
    def_vec_x = np.array([
        [0, 0, 0],
        [0, -0.2, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    def_vec_y = np.array([
        [0, 0, 0],
        [0, -0.3, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    def_vec = np.stack([def_vec_x, def_vec_y], axis=2).reshape([1, 3, 3, 2])
    def_tsr = tf.Variable(def_vec, dtype=tf.float32)

    # SpatialTransformer
    z = SpatialTransformer()(img_arr, def_tsr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(z)
        print(result.shape, result.dtype)
        result = result.astype(np.uint8)
        Image.fromarray(255 - result[0, :, :, 0], "L").save(r"F:\tmp3\transformed.png")


if __name__ == '__main__':
    main()
    gen_diff_arr(
        x_path=r"F:\tmp3\transformed.png",
        y_path=r"F:\tmp3\original.png",
        out_path=r"f:\tmp3\diff.png"
    )
