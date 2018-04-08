import os
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
import SimpleITK as sitk
from DIRNet_tensorflow_master.models import DIRNet
from DIRNet_tensorflow_master.config import get_config
from DIRNet_tensorflow_master.data import MNISTDataHandler


def main():
    # get configure
    config = get_config(is_train=False)

    # file operations
    if not os.path.exists(config["result_dir"]):
        os.mkdir(config["result_dir"])

    #
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=False)
    reg.restore(config["checkpoint_dir"])
    dh = MNISTDataHandler("MNIST_data", is_train=False)

    for i in range(10):
        result_i_dir = config["result_dir"] + "/{}".format(i)
        if not os.path.exists(result_i_dir):
            os.mkdir(result_i_dir)

        batch_x, batch_y = dh.sample_pair(config["batch_size"], i)
        reg.deploy(result_i_dir, batch_x, batch_y)


def my_test():
    """暂时往里面验证一些图像"""
    # 加载数据
    batch_x, batch_y = pickle.load(open(r"F:\\registration_patches\\ct_batches_test.pickle", 'rb'))
    batch_x = batch_x[100:200]
    batch_y = batch_y[100:200]
    batch_x = (batch_x - batch_x.min()) / (batch_x.max() - batch_x.min())
    batch_y = (batch_y - batch_y.min()) / (batch_y.max() - batch_y.min())
    # show image
    # sitk.Show(sitk.GetImageFromArray(batch_xs))
    # sitk.Show(sitk.GetImageFromArray(batch_ys))

    # config
    config_dict = {
        "checkpoint_dir": "checkpoint",
        "image_size": [128, 128],
        "batch_size": len(batch_x),
        "result_dir": "result",
    }

    # file operations
    if not os.path.exists(config_dict["result_dir"]):
        os.mkdir(config_dict["result_dir"])

    # create network
    sess = tf.Session()
    reg = DIRNet(sess, config_dict, "DIRNet", is_train=False)
    reg.restore(config_dict["checkpoint_dir"])

    # register
    reg.deploy(config_dict["result_dir"], batch_x, batch_y)


if __name__ == "__main__":
    # x = r"C:\Users\anonymous\Desktop\workspace\moving_image.jpg"
    # y = r"C:\Users\anonymous\Desktop\workspace\fixed_image.jpg"
    # x_arr = np.array(Image.open(x)).reshape([64, 64, 1])
    # y_arr = np.array(Image.open(y)).reshape([64, 64, 1])
    # Image.fromarray(x_arr[:, :, 0]).show()
    # Image.fromarray(y_arr[:, :, 0]).show()
    # x_arr = x_arr / 255
    # y_arr = y_arr / 255
    # register_img(x_arr, y_arr)
    my_test()
