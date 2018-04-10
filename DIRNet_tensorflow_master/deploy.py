import os
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
import SimpleITK as sitk
from DIRNet_tensorflow_master.models import DIRNet


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
    my_test()
