import tensorflow as tf
from DIRNet_tensorflow_master.models import DIRNet
from DIRNet_tensorflow_master.config import get_config
from DIRNet_tensorflow_master.data import MNISTDataHandler
from DIRNet_tensorflow_master.ops import mkdir


def main():
    sess = tf.Session()
    config = get_config(is_train=True)
    mkdir(config["temp_dir"])
    mkdir(config["checkpoint_dir"])

    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    dh = MNISTDataHandler("MNIST_data", is_train=True)

    for i in range(config["iteration_num"]):
        batch_x, batch_y = dh.sample_pair(config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        print("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])


if __name__ == "__main__":
    main()
