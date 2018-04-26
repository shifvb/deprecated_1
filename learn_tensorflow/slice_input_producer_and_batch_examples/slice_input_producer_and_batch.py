import os
import numpy as np
import tensorflow as tf
from PIL import Image


def get_filenames_and_labels(file_dir: str):  # 得到cats和dogs的图像路径(shape=[25000, ])和标签(shape=[25000, ])
    _cats, _label_cats, _dogs, _label_dogs = [], [], [], []
    # get cats abs filename & label
    cat_dir = os.path.join(file_dir, "Cat")
    for filename in os.listdir(cat_dir):
        _cats.append(os.path.join(cat_dir, filename))
        _label_cats.append(0)
    _cats.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0]))
    # get dogs abs filename & label
    dog_dir = os.path.join(file_dir, "Dog")
    for filename in os.listdir(dog_dir):
        _dogs.append(os.path.join(dog_dir, filename))
        _label_dogs.append(1)
    _dogs.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0]))
    return np.hstack([_cats, _dogs]), np.hstack([_label_cats, _label_dogs])


def f(input_tensor):
    L = []
    for i in input_tensor:
        L.append(np.array(Image.open(i), dtype=np.uint8))
    return np.stack(L)


def main():
    batch_size = 1
    shuffle = False
    epoch_num = 10000

    # 生成图片集和标签
    image_arr, label_arr = get_filenames_and_labels(r'F:\kagglecatsanddogs_3367a\PetImages')
    input_queue = tf.train.slice_input_producer([image_arr, label_arr], shuffle=shuffle)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)
    # 自己加的转换函数
    image_batch = tf.py_func(f, [image_batch], tf.uint8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # train
        for i in range(epoch_num):
            img, label = sess.run([image_batch, label_batch])
            print("iter={}, x={}, y={}".format(i, img.shape, label))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
