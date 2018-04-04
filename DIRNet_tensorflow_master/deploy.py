import os
import tensorflow as tf
from PIL import Image
import numpy as np
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


def register_img(moving_image, fixed_image):
    config_dict = {
        "checkpoint_dir": "checkpoint",
        "image_size": [64, 64],
        "batch_size": 1,
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
    batch_x = np.array([moving_image])
    batch_y = np.array([fixed_image])
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
    main()
