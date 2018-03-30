import os
import tensorflow as tf
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


if __name__ == "__main__":
    main()
