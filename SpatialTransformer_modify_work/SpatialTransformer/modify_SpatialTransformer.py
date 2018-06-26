import numpy as np
from PIL import Image
import tensorflow as tf
from SpatialTransformer_modify_work.imgprocess.get_images_arr import get_images_arr
from SpatialTransformer_modify_work.imgprocess.gen_diff_arr import gen_diff_arr
from SpatialTransformer_modify_work.SpatialTransformer.SpatialTransformer import SpatialTransformer


def main():
    # 图像数据
    img_arr = get_images_arr(r"img")
    Image.fromarray(255 - img_arr[0, :, :, 0], "L").save("img_out\\original.png")

    # 形变场向量
    def_vec_x = np.array([
        [0, 0, 0],
        [0, -0.3, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    def_vec_y = np.array([
        [0, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    def_vec = np.stack([def_vec_x, def_vec_y], axis=2).reshape([1, 3, 3, 2])
    def_tsr = tf.Variable(def_vec, dtype=tf.float32)

    # SpatialTransformer
    st = SpatialTransformer()
    z = st(img_arr, def_tsr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, ma, mi = sess.run([z, st.ma, st.mi])
        print(ma, mi)
        print(result.shape, result.dtype)
        result = result.astype(np.uint8)
        Image.fromarray(255 - result[0, :, :, 0], "L").save(r"img_out\transformed.png")


if __name__ == '__main__':
    main()
    gen_diff_arr(
        x_path=r"img_out\transformed.png",
        y_path=r"img_out\original.png",
        out_path=r"img_out\diff.png"
    )
